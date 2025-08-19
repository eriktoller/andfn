"""
Notes
-----
This module contains the DFN class.

The DFN class is the main class for the AnDFN package. It contains the fractures and elements of the DFN and methods to
solve the DFN.
"""

import numpy as np
import os
import pyvista as pv
import scipy as sp
import h5py
import time

from andfn import geometry_functions as gf
from .constants import Constants, dtype_constants
from .fracture import Fracture
from .hpc.hpc_solve import solve as hpc_solve
from .hpc.hpc_fracture import (
    get_flow_nets as hpc_get_flow_nets,
    get_heads as hpc_get_heads,
    sunflower_spiral
)
from .structures import STRUCTURES_COLOR
from .well import Well
from .impermeable_object import ImpermeableCircle, ImpermeableLine
from .const_head import ConstantHeadLine
from .intersection import Intersection
from .bounding import BoundingCircle
from .element import (
    element_dtype,
    fracture_dtype,
    element_index_dtype,
    fracture_index_dtype,
    element_dtype_hpc,
    fracture_dtype_hpc,
    ELEMENT_COLORS,
)

# Custom colormaps
from matplotlib.colors import LinearSegmentedColormap

import logging

logger = logging.getLogger("andfn")


def _constant_cmap(name, color, n_bin):
    """
    Creates a red colormap.

    Parameters
    ----------
    name : str
        The name of the colormap.
    color : str
        The color of the colormap.
    n_bin : int
        The number of bins in the colormap.

    Returns
    -------
    cmap : LinearSegmentedColormap
        The colormap.
    """
    # Create a red colormap
    return LinearSegmentedColormap.from_list(name, [color, color], N=n_bin)


def generate_connected_fractures(
    num_fracs,
    radius_factor,
    center_factor,
    ncoef_i,
    nint_i,
    ncoef_b,
    nint_b,
    frac_surface=None,
    se_factor=1.0,
):
    """
    Generates connected fractures and intersections.

    Parameters
    ----------
    num_fracs : int
        The number of fractures to generate.
    radius_factor : float
        The factor to multiply the radius by.
    center_factor : float
        The factor to multiply the center by.
    ncoef_i : int
        The number of coefficients to use for the intersection elements.
    nint_i : int
        The number of integration points to use for the intersection elements.
    ncoef_b : int
        The number of coefficients to use for the bounding elements.
    nint_b : int
        The number of integration points to use for the bounding elements.
    frac_surface : Fracture
        The fracture to use as the surface fracture.
    se_factor : float, optional
        The factor to use for shortening fo the elements. Default is 1.0 (no shortening).

    Returns
    -------
    frac_list : list
        The list of fractures.

    """
    print("Generating fractures...")
    fracs = gf.generate_fractures(
        num_fracs,
        radius_factor=radius_factor,
        center_factor=center_factor,
        ncoef=ncoef_b,
        nint=nint_b,
    )

    print("Analyzing intersections...")
    frac_list = gf.get_connected_fractures(
        fracs, se_factor, ncoef_i, nint_i, frac_surface
    )

    return frac_list


def get_lvs(lvs, omega_fn_list):
    """
    Gets the levels for the flow net.

    Parameters
    ----------
    lvs : int
        The number of levels for the equipotentials.
    omega_fn_list : list | np.ndarray
        The list of complex discharge values.

    Returns
    -------
    lvs_re : np.ndarray
        The levels for the equipotentials.
    lvs_im : np.ndarray
        The levels for the streamlines.
    """
    # Find the different in min and max for the stream function and equipotential
    omega_max_re, omega_min_re = (
        np.nanmax(np.real(omega_fn_list)),
        np.nanmin(np.real(omega_fn_list)),
    )
    omega_max_im, omega_min_im = (
        np.nanmax(np.imag(omega_fn_list)),
        np.nanmin(np.imag(omega_fn_list)),
    )
    delta_re = np.abs(omega_min_re - omega_max_re)
    delta_im = np.abs(omega_min_im - omega_max_im)
    # Create the levels for the equipotential contours
    lvs_re = np.linspace(omega_min_re, omega_max_re, lvs)
    # Create the levels for the stream function contours (using the same step size)
    step = delta_re / lvs
    n_steps = int(delta_im / step)
    lvs_im = np.linspace(omega_min_im, omega_max_im, n_steps)
    return lvs_re, lvs_im


def plot_line_3d(seg, f, pl, color, line_width):
    """
    Plots a line in 3D for a given fracture plane.

    Parameters
    ----------
    seg : np.ndarray
        The line to plot.
    f : Fracture
        The fracture plane.
    pl : pyvista.Plotter
        The plotter object.
    color : str | tuple
        The color of the line.
    line_width : float
        The line width of the line.
    """
    if seg.dtype is not np.dtype("complex"):
        x = seg[:, 0]
        y = seg[:, 1]
        contour_complex = x + 1j * y
    else:
        contour_complex = seg
    line_3d = gf.map_2d_to_3d(contour_complex, f)
    pl.add_mesh(pv.MultipleLines(line_3d), color=color, line_width=line_width)


def get_faces(pnts):
    """
    Gets the faces for a given set of points using Delaunay triangulation.

    Parameters
    ----------
    pnts : np.ndarray
        The points to get the faces for.

    Returns
    -------
    faces : np.ndarray
        The faces for the points.
    """
    # Create the faces for the points
    return pv.PolyData(pnts).delaunay_2d().faces


class DFN(Constants):
    def __init__(self, label, discharge_int=10, **kwargs):
        """
        Initializes the DFN class.

        Parameters
        ----------
        label : str or int
            The label of the DFN.
        discharge_int : int
            The number of points to use for the discharge integral.

        """
        super().__init__()
        self.label = label
        self.discharge_int = discharge_int
        self.fractures = []
        self.structures = []
        self.elements = None

        # Initialize the discharge matrix
        self.discharge_matrix = None
        self.discharge_elements = None
        self.discharge_elements_index = None
        self.lup = None
        self.discharge_error = 1

        # Initialize the structure array
        self.elements_struc_array = None
        self.elements_index_array = None
        self.fractures_struc_array = None
        self.fractures_index_array = None
        self.elements_struc_array_hpc = None
        self.fractures_struc_array_hpc = None

        # Set the kwargs
        self.set_kwargs(**kwargs)

    def __str__(self):
        """
        Returns the string representation of the DFN.

        Returns
        -------
        str
            The string representation of the DFN.
        """
        return f"DFN: {self.label}"

    def set_kwargs(self, **kwargs):
        """
        Changes the attributes of the DFN.

        Parameters
        ----------
        kwargs : **kwargs
            The attributes to change.
        """
        for key, value in kwargs.items():
            if key in dtype_constants.names:
                self.change_constants(**{key: value})
                continue
            setattr(self, key, value)

    ####################################################################################################################
    #                      Load and save                                                                               #
    ####################################################################################################################
    def save_dfn(self, filename):
        """
        Saves the DFN to a h5 file.

        Parameters
        ----------
        filename : str
            The name of the file to save the DFN to.
        """
        # Remove the .h5 if it is in the filename
        if filename[-3:] == ".h5":
            filename = filename[:-3]

        # Check if the elements and fractures have been consolidated
        if self.elements_struc_array is None or self.fractures_struc_array is None:
            self.consolidate_dfn()

        # Save the elements
        with h5py.File(f"{filename}.h5", "w") as hf:
            grp = [
                hf.create_group("elements/properties"),
                hf.create_group("fractures/properties"),
                hf.create_group("elements/index"),
                hf.create_group("fractures/index"),
            ]
            for j, array in enumerate(
                [self.elements_struc_array, self.fractures_struc_array]
            ):
                for name in array.dtype.names:
                    # create group
                    grp0 = grp[j].create_group(name)
                    # add data
                    for i, e in enumerate(array[name]):
                        grp0.create_dataset(f"{i}", data=e)

            for j, array in enumerate(
                [self.elements_index_array, self.fractures_index_array], start=2
            ):
                for name in array.dtype.names:
                    # check if the data is a string
                    if self.elements_index_array[name].dtype == "U100":
                        grp[j].create_dataset(name, data=array[name].astype("S"))
                        continue
                    grp[j].create_dataset(name, data=array[name])

        logger.info(f"Saved DFN to {filename}.h5")

    def load_dfn(self, filename):
        """
        Loads the DFN from a h5 file.

        Parameters
        ----------
        filename : str
            The name of the file to load the DFN from.
        """

        if filename[-3:] == ".h5":
            filename = filename[:-3]

        # Check if the file exists
        if not os.path.exists(f"{filename}.h5"):
            raise FileNotFoundError(f"The file {filename}.h5 does not exist.")

        logger.info(f"Loading DFN from {filename}.h5")

        with h5py.File(f"{filename}.h5", "r") as hf:
            # Load the fractures
            fracs = []
            for i in range(len(hf["fractures/index/label"])):
                fracs.append(
                    Fracture(
                        label=hf["fractures/index/label"][i].decode(),
                        _id=hf["fractures/index/_id"][i],
                        t=hf[f"fractures/properties/t/{i}"][()],
                        radius=hf[f"fractures/properties/radius/{i}"][()],
                        center=hf[f"fractures/properties/center/{i}"][()],
                        normal=hf[f"fractures/properties/normal/{i}"][()],
                        x_vector=hf[f"fractures/properties/x_vector/{i}"][()],
                        y_vector=hf[f"fractures/properties/y_vector/{i}"][()],
                        elements=False,
                        constant=hf[f"fractures/properties/constant/{i}"][()],
                    )
                )

            # Load the elements
            elements = []
            for i in range(len(hf["elements/index/label"])):
                if hf["elements/index/_type"][i] == 0:  # Intersection
                    elements.append(
                        Intersection(
                            label=hf["elements/index/label"][i].decode(),
                            _id=hf["elements/index/_id"][i],
                            endpoints0=hf[f"elements/properties/endpoints0/{i}"][()],
                            endpoints1=hf[f"elements/properties/endpoints1/{i}"][()],
                            ncoef=hf[f"elements/properties/ncoef/{i}"][()],
                            nint=hf[f"elements/properties/nint/{i}"][()],
                            frac0=fracs[hf[f"elements/properties/frac0/{i}"][()]],
                            frac1=fracs[hf[f"elements/properties/frac1/{i}"][()]],
                            q=hf[f"elements/properties/q/{i}"][()],
                            thetas=hf[f"elements/properties/thetas/{i}"][()],
                            coef=hf[f"elements/properties/coef/{i}"][()],
                            error=hf[f"elements/properties/error/{i}"][()],
                        )
                    )
                elif hf["elements/index/_type"][i] == 1:  # Bounding circle
                    elements.append(
                        BoundingCircle(
                            label=hf["elements/index/label"][i].decode(),
                            _id=hf["elements/index/_id"][i],
                            radius=hf[f"elements/properties/radius/{i}"][()],
                            center=hf[f"elements/properties/center/{i}"][()],
                            frac0=fracs[hf[f"elements/properties/frac0/{i}"][()]],
                            thetas=hf[f"elements/properties/thetas/{i}"][()],
                            coef=hf[f"elements/properties/coef/{i}"][()],
                            ncoef=hf[f"elements/properties/ncoef/{i}"][()],
                            nint=hf[f"elements/properties/nint/{i}"][()],
                            dpsi_corr=hf[f"elements/properties/dpsi_corr/{i}"][()],
                            error=hf[f"elements/properties/error/{i}"][()],
                        )
                    )
                elif hf["elements/index/_type"][i] == 2:  # Well
                    elements.append(
                        Well(
                            label=hf["elements/index/label"][i].decode(),
                            _id=hf["elements/index/_id"][i],
                            radius=hf[f"elements/properties/radius/{i}"][()],
                            center=hf[f"elements/properties/center/{i}"][()],
                            head=hf[f"elements/properties/head/{i}"][()],
                            frac0=fracs[hf[f"elements/properties/frac0/{i}"][()]],
                            q=hf[f"elements/properties/q/{i}"][()],
                            phi=hf[f"elements/properties/phi/{i}"][()],
                            error=hf[f"elements/properties/error/{i}"][()],
                        )
                    )
                elif hf["elements/index/_type"][i] == 3:  # Constant head line
                    elements.append(
                        ConstantHeadLine(
                            label=hf["elements/index/label"][i].decode(),
                            _id=hf["elements/index/_id"][i],
                            head=hf[f"elements/properties/head/{i}"][()],
                            endpoints0=hf[f"elements/properties/endpoints0/{i}"][()],
                            ncoef=hf[f"elements/properties/ncoef/{i}"][()],
                            nint=hf[f"elements/properties/nint/{i}"][()],
                            frac0=fracs[hf[f"elements/properties/frac0/{i}"][()]],
                            q=hf[f"elements/properties/q/{i}"][()],
                            phi=hf[f"elements/properties/phi/{i}"][()],
                            thetas=hf[f"elements/properties/thetas/{i}"][()],
                            coef=hf[f"elements/properties/coef/{i}"][()],
                            error=hf[f"elements/properties/error/{i}"][()],
                        )
                    )
                elif hf["elements/index/_type"][i] == 4:  # Impermeable circle
                    elements.append(
                        ImpermeableCircle(
                            label=hf["elements/index/label"][i].decode(),
                            _id=hf["elements/index/_id"][i],
                            radius=hf[f"elements/properties/radius/{i}"][()],
                            center=hf[f"elements/properties/center/{i}"][()],
                            frac0=fracs[hf[f"elements/properties/frac0/{i}"][()]],
                            ncoef=hf[f"elements/properties/ncoef/{i}"][()],
                            nint=hf[f"elements/properties/nint/{i}"][()],
                            thetas=hf[f"elements/properties/thetas/{i}"][()],
                            coef=hf[f"elements/properties/coef/{i}"][()],
                            error=hf[f"elements/properties/error/{i}"][()],
                        )
                    )
                elif hf["elements/index/_type"][i] == 5:  # Impermeable line
                    elements.append(
                        ImpermeableLine(
                            label=hf["elements/index/label"][i].decode(),
                            _id=hf["elements/index/_id"][i],
                            endpoints0=hf[f"elements/properties/focis/{i}"][()],
                            frac0=fracs[hf[f"elements/properties/frac0/{i}"][()]],
                            ncoef=hf[f"elements/properties/ncoef/{i}"][()],
                            nint=hf[f"elements/properties/nint/{i}"][()],
                            thetas=hf[f"elements/properties/thetas/{i}"][()],
                            coef=hf[f"elements/properties/coef/{i}"][()],
                            dpsi_corr=hf[f"elements/properties/dpsi_corr/{i}"][()],
                            error=hf[f"elements/properties/error/{i}"][()],
                        )
                    )

            # Add the fractures and elements to the DFN
            self.add_fracture(fracs)
            self.get_elements()

    def consolidate_dfn(self, hpc=False):
        # Check if the elements have been stored in the DFN
        if self.elements is None:
            self.get_elements()

        # Consolidate elements
        if hpc:
            e_dtype = element_dtype_hpc
            f_dtype = fracture_dtype_hpc
        else:
            e_dtype = element_dtype
            f_dtype = fracture_dtype
        elements_struc_array = np.empty(self.number_of_elements(), dtype=e_dtype)
        elements_index_array = np.empty(
            self.number_of_elements(), dtype=element_index_dtype
        )

        if hpc:
            for i, e in enumerate(self.elements):
                elements_struc_array[i], elements_index_array[i] = e.consolidate_hpc()
        else:
            for i, e in enumerate(self.elements):
                elements_struc_array[i], elements_index_array[i] = e.consolidate()

        # Consolidate fractures
        fractures_struc_array = np.empty(self.number_of_fractures(), dtype=f_dtype)
        fractures_index_array = np.empty(
            self.number_of_fractures(), dtype=fracture_index_dtype
        )

        if hpc:
            for i, f in enumerate(self.fractures):
                fractures_struc_array[i], fractures_index_array[i] = f.consolidate_hpc()
        else:
            for i, f in enumerate(self.fractures):
                fractures_struc_array[i], fractures_index_array[i] = f.consolidate()

        # Save to self
        if hpc:
            self.elements_struc_array_hpc = elements_struc_array
            self.fractures_struc_array_hpc = fractures_struc_array
        else:
            self.elements_struc_array = elements_struc_array
            self.fractures_struc_array = fractures_struc_array
        self.elements_index_array = elements_index_array
        self.fractures_index_array = fractures_index_array

    def unconsolidate_dfn(self, hpc=False):
        """
        Unconsolidates the DFN.

        Parameters
        ----------
        hpc : bool
            If True, the DFN is unconsolidated for the HPC.
        """

        # Unconsolidate fractures
        if hpc:
            for i, f in enumerate(self.fractures):
                f.unconsolidate_hpc(
                    self.fractures_struc_array_hpc[i], self.fractures_index_array[i]
                )
            for i, e in enumerate(self.elements):
                e.unconsolidate_hpc(
                    self.elements_struc_array_hpc[i],
                    self.elements_index_array[i],
                    self.fractures,
                )
        else:
            for i, f in enumerate(self.fractures):
                f.unconsolidate(
                    self.fractures_struc_array[i], self.fractures_index_array[i]
                )
            for i, e in enumerate(self.elements):
                e.unconsolidate(
                    self.elements_struc_array[i],
                    self.elements_index_array[i],
                    self.fractures,
                )

    ####################################################################################################################
    #                      DFN functions                                                                               #
    ####################################################################################################################

    def number_of_fractures(self):
        """
        Returns the number of fractures in the DFN.
        """
        return len(self.fractures)

    def number_of_elements(self):
        """
        Returns the number of elements in the DFN.
        """
        return len(self.elements)

    def get_elements(self):
        """
        Gets the elements from the fractures and add store them in the DFN.
        """
        elements = []
        for f in self.fractures:
            if f.elements is None or len(f.elements) == 0:
                continue
            for e in f.elements:
                if e not in elements:
                    elements.append(e)
        self.elements = elements
        logger.info(f"Added {len(self.elements)} elements to the DFN.")

        for e in self.elements:
            e.set_id(self.elements.index(e))

    def get_discharge_elements(self):
        """
        Gets the discharge elements from the fractures and add store them in the DFN.
        """
        # Check if the elements have been stored in the DFN
        if self.elements is None:
            self.get_elements()
        # Get the discharge elements
        self.discharge_elements = [
            e
            for e in self.elements
            if isinstance(e, Intersection)
            or isinstance(e, ConstantHeadLine)
            or isinstance(e, Well)
        ]

        self.discharge_elements_index = [e._id for e in self.discharge_elements]

    def get_dfn_discharge(self):
        # sum all discharges, except the intersections
        if self.discharge_elements is None:
            self.get_discharge_elements()
        q = 0
        for e in self.discharge_elements:
            if isinstance(e, Intersection):
                continue
            q += np.abs(e.q)
        return q / 2

    def add_fracture(self, new_fracture):
        """
        Adds a fracture to the DFN.

        Parameters
        ----------
        new_fracture : Fracture | list
            The fracture to add to the DFN.
        """
        if isinstance(new_fracture, list):
            if len(new_fracture) == 1:
                self.fractures.append(new_fracture[0])
                logger.info(f"Added {new_fracture[0]} fracture to the DFN.")
            else:
                self.fractures.extend(new_fracture)
                logger.info(f"Added {len(new_fracture)} fractures to the DFN.")
        else:
            self.fractures.append(new_fracture)
            logger.info(f"Added {new_fracture} fracture to the DFN.")
        # reset the discharge matrix and elements
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None
        self.lup = None

        for f in self.fractures:
            f.set_id(self.fractures.index(f))

    def delete_fracture(self, fracture):
        """
        Deletes a fracture from the DFN.

        Parameters
        ----------
        fracture : Fracture
            The fracture to delete from the DFN.
        """
        self.fractures.remove(fracture)
        # reset the discharge matrix and elements
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None

    def add_structure(self, new_structure):
        """
        Adds a structure to the DFN.

        Parameters
        ----------
        new_structure : ConstantHeadPrism | ImpermeablePrims | list
            The structure to add to the DFN.
        """
        if isinstance(new_structure, list):
            if len(new_structure) == 1:
                self.structures.append(new_structure[0])
                new_structure[0].frac_intersections(self.fractures)
                logger.info(f"Added {new_structure[0]} fracture to the DFN.")
            else:
                self.structures.extend(new_structure)
                for s in new_structure:
                    s.frac_intersections(self.fractures)
                logger.info(f"Added {len(new_structure)} fractures to the DFN.")
        else:
            self.structures.append(new_structure)
            new_structure.frac_intersections(self.fractures)
            logger.info(f"Added {new_structure} fracture to the DFN.")

    def import_fractures_from_file(
        self,
        path,
        radius_str,
        x_str,
        y_str,
        z_str,
        t_str,
        e_str=None,
        strike_str=None,
        dip_str=None,
        trend_str=None,
        plunge_str=None,
        starting_frac=None,
        remove_isolated=True,
    ):
        """
        Imports fractures from a csv file. More formatting options can be added later.

        Parameters
        ----------
        path : str
            The path to the file containing the fractures.
        radius_str : str
            The name of the column containing the radius of the fractures.
        x_str : str
            The name of the column containing the x coordinate of the center of the fractures.
        y_str : str
            The name of the column containing the y coordinate of the center of the fractures.
        z_str : str
            The name of the column containing the z coordinate of the center of the fractures.
        t_str : str
            The name of the column containing the transmissivity of the fractures.
        e_str : str, optional
            The name of the column containing the aperture of the fractures. The default is None.
        strike_str : str, optional
            The name of the column containing the strike of the fractures. The default is None.
        dip_str : str, optional
            The name of the column containing the dip of the fractures. The default is None.
        trend_str : str, optional
            The name of the column containing the trend of the fractures. The default is None.
        plunge_str : str, optional
            The name of the column containing the plunge of the fractures. The default is None.
        starting_frac : int, optional
            The fracture to use as the starting point for the connected fractures. The default is None.
        remove_isolated : bool, optional
            If True, removes isolated fractures from the DFN. The default is True.

        Returns
        -------
        None
            The fractures are added to the DFN.
        """

        # Check if pandas is installed
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required to import fractures from a file. Please install pandas."
            )

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

        data_file = pd.read_csv(path)
        frac = []
        for i in range(len(data_file)):
            radius = data_file[radius_str][i]
            if strike_str is not None and dip_str is not None:
                normal = gf.convert_strike_dip_to_normal(
                    data_file[strike_str][i], data_file[dip_str][i]
                )
            elif trend_str is not None and plunge_str is not None:
                normal = gf.convert_trend_plunge_to_normal(
                    data_file[trend_str][i], data_file[plunge_str][i]
                )
            else:
                raise ValueError("Either strike/dip or trend/plunge must be provided.")
            normal = normal / np.linalg.norm(normal)
            center = np.array(
                [data_file[x_str][i], data_file[y_str][i], data_file[z_str][i]]
            )
            transmissivity = data_file[t_str][i]
            if e_str is None:
                frac.append(
                    Fracture(
                        f"{i}",
                        transmissivity,
                        radius,
                        center,
                        normal,
                        ncoef=self.constants["NCOEF"],
                        nint=self.constants["NINT"],
                    )
                )
            else:
                aperture = data_file[e_str][i]
                frac.append(
                    Fracture(
                        f"{i}",
                        transmissivity,
                        radius,
                        center,
                        normal,
                        aperture,
                        ncoef=self.constants["NCOEF"],
                        nint=self.constants["NINT"],
                    )
                )

        if starting_frac is not None:
            fracs = gf.get_connected_fractures(
                frac,
                self.constants["SE_FACTOR"],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
                fracture_surface=frac[starting_frac],
            )
        else:
            fracs = gf.get_fracture_intersections(
                frac,
                self.constants["SE_FACTOR"],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
            )

        if remove_isolated:
            # Remove isolated fractures
            len_before = len(fracs)
            fracs = gf.remove_isolated_fractures(fracs)
            removed = len_before - len(fracs)
            if removed > 0:
                logger.info(
                    f"Removed {len_before - len(fracs)} isolated fractures from the DFN."
                )

        self.add_fracture(fracs)

    def generate_connected_dfn(
        self,
        num_fracs,
        radius_factor,
        center_factor,
        ncoef_i,
        nint_i,
        ncoef_b,
        nint_b,
        frac_surface=None,
    ):
        """
        Generates a connected DFN and adds it and the intersections to the DFN.

        Parameters
        ----------
        num_fracs : int
            Number of fractures to generate.
        radius_factor : float
            The factor to multiply the radius by.
        center_factor : float
            The factor to multiply the center by.
        ncoef_i : int
            The number of coefficients to use for the intersection elements.
        nint_i : int
            The number of integration points to use for the intersection elements.
        ncoef_b : int
            The number of coefficients to use for the bounding elements.
        nint_b : int
            The number of integration points to use for the bounding elements.
        frac_surface : Fracture
            The fracture to use as the surface fracture.
        """
        # Generate the connected fractures
        frac_list = generate_connected_fractures(
            num_fracs,
            radius_factor,
            center_factor,
            ncoef_i,
            nint_i,
            ncoef_b,
            nint_b,
            frac_surface,
        )
        # Add the fractures to the DFN
        self.add_fracture(frac_list)

    def get_fracture_intersections(self, ncoef=5, nint=10, new_frac=None):
        """
        Finds the intersections between the fractures in the DFN and adds them to the DFN.

        Parameters
        ----------
        ncoef : int
            The number of coefficients to use for the intersection elements.
        nint : int
            The number of integration points to use for the intersection elements
        new_frac : Fracture
            The fracture to calculate the intersections for.
        """
        # Compute intersections between fractures only for frac_surface
        if new_frac is not None:
            fr = new_frac
            for k in range(len(self.fractures)):
                fr2 = self.fractures[k]
                if fr == fr2:
                    continue
                endpoints0, endpoints1 = gf.fracture_intersection(fr, fr2)
                if endpoints0 is not None:
                    i0 = Intersection(
                        f"{fr.label}_{fr2.label}",
                        endpoints0,
                        endpoints1,
                        fr,
                        fr2,
                        ncoef,
                        nint,
                    )
                    fr.add_element(i0)
                    fr2.add_element(i0)

        # Compute intersections between all fractures
        if new_frac is None:
            for i in range(len(self.fractures)):
                fr = self.fractures[i]
                for k in range(i + 1, len(self.fractures)):
                    fr2 = self.fractures[k]
                    if fr == fr2:
                        continue
                    endpoints0, endpoints1 = gf.fracture_intersection(fr, fr2)
                    if endpoints0 is not None:
                        i0 = Intersection(
                            f"{fr.label}_{fr2.label}",
                            endpoints0,
                            endpoints1,
                            fr,
                            fr2,
                            ncoef,
                            nint,
                        )
                        fr.add_element(i0)
                        fr2.add_element(i0)

    def get_dfn_center(self):
        """
        Gets the center of the DFN.
        """
        center = np.array([0.0, 0.0, 0.0])
        for f in self.fractures:
            center += f.center
        return center / len(self.fractures)

    def set_constant_head_boundary(
        self,
        center,
        normal,
        radius,
        head,
        label="Constant Head Boundary",
        ncoef=5,
        nint=10,
    ):
        """
        Adds a constant head boundary to the DFN.

        Parameters
        ----------
        center : np.ndarray
            The center of the constant head boundary.
        normal : np.ndarray
            The normal of the constant head boundary.
        radius : float
            The radius of the constant head boundary.
        head : float
            The head of the constant head boundary.
        label : str
            The label of the constant head boundary.
        """
        gf.set_head_boundary(
            self.fractures,
            ncoef,
            nint,
            head,
            center,
            radius,
            normal,
            label,
            self.constants["SE_FACTOR"],
        )

    ####################################################################################################################
    #                      Solve functions                                                                             #
    ####################################################################################################################

    def build_discharge_matrix(self):
        """
        Builds the discharge matrix for the DFN and adds it to the DFN.

        """
        self.get_discharge_elements()
        size = len(self.discharge_elements) + self.number_of_fractures()

        # Create a sparse matrix
        # create the row, col and data arrays
        rows = []
        cols = []
        data = []

        # Add the discharge for each discharge element
        row = 0
        for e in self.discharge_elements:
            if isinstance(e, Intersection):
                z0 = e.z_array(self.discharge_int, e.frac0)
                z1 = e.z_array(self.discharge_int, e.frac1)
                for ee in e.frac0.get_discharge_elements():
                    if ee == e:
                        continue
                    # add the discharge term to the matrix for each element in the first fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        rows.append(row)
                        cols.append(pos)
                        data.append(
                            e.frac0.head_from_phi(ee.discharge_term(z0, e.frac0))
                        )
                    else:
                        rows.append(row)
                        cols.append(pos)
                        data.append(e.frac0.head_from_phi(ee.discharge_term(z0)))
                for ee in e.frac1.get_discharge_elements():
                    if ee == e:
                        continue
                    # add the discharge term to the matrix for each element in the second fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        rows.append(row)
                        cols.append(pos)
                        data.append(
                            e.frac1.head_from_phi(-ee.discharge_term(z1, e.frac1))
                        )
                    else:
                        rows.append(row)
                        cols.append(pos)
                        data.append(e.frac1.head_from_phi(-ee.discharge_term(z1)))
                pos_f0 = self.fractures.index(e.frac0)
                rows.append(row)
                cols.append(len(self.discharge_elements) + pos_f0)
                data.append(e.frac0.head_from_phi(1))
                pos_f1 = self.fractures.index(e.frac1)
                rows.append(row)
                cols.append(len(self.discharge_elements) + pos_f1)
                data.append(e.frac1.head_from_phi(-1))
            else:
                z = e.z_array(self.discharge_int)
                for ee in e.frac0.get_discharge_elements():
                    if ee == e:
                        continue
                    # add the discharge term to the matrix for each element in the fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        rows.append(row)
                        cols.append(pos)
                        data.append(ee.discharge_term(z, e.frac0))
                    else:
                        rows.append(row)
                        cols.append(pos)
                        data.append(ee.discharge_term(z))
                pos_f = self.fractures.index(e.frac0)
                rows.append(row)
                cols.append(len(self.discharge_elements) + pos_f)
                data.append(1)
            row += 1

        # Add the constants for each fracture
        for f in self.fractures:
            # fill the matrix for the fractures
            for e in f.elements:
                if e in self.discharge_elements:
                    # add the discharge term to the matrix for each element in the fracture
                    pos = self.discharge_elements.index(e)
                    if isinstance(e, Intersection):
                        if e.frac0 == f:
                            rows.append(row)
                            cols.append(pos)
                            data.append(1)
                        else:
                            rows.append(row)
                            cols.append(pos)
                            data.append(-1)
                    else:
                        rows.append(row)
                        cols.append(pos)
                        data.append(1)
            row += 1

        # create the csr sparse matrix
        matrix = sp.sparse.csc_matrix((data, (rows, cols)), shape=(size, size))

        self.discharge_matrix = matrix

    def solve(self):
        """
        Solves the DFN on a HPC.

        To change the solver constants, use the set_kwargs method.
        """
        logger.info("\n")
        logger.info("---------------------------------------")
        logger.info("Starting HPC solve...")
        logger.info("---------------------------------------")
        logger.info("Collecting elements and fractures...")
        self.get_elements()
        logger.info("Consolidating DFN...")
        self.consolidate_dfn(hpc=True)
        logger.info("Building discharge matrix...")
        self.build_discharge_matrix()

        logger.info(f"Number of elements: {len(self.elements)}")
        logger.info(f"Number of fractures: {len(self.fractures)}")
        logger.info(
            f"Number of entries in discharge matrix: {self.discharge_matrix.getnnz()}"
        )
        self.print_solver_constants()
        self.elements_struc_array = hpc_solve(
            self.fractures_struc_array_hpc,
            self.elements_struc_array_hpc,
            self.discharge_matrix,
            self.discharge_int,
            self.constants,
        )
        self.unconsolidate_dfn(hpc=True)

    ####################################################################################################################
    #                    Plotting functions                                                                            #
    ####################################################################################################################

    def initiate_plotter(
        self,
        window_size=(800, 800),
        grid=False,
        lighting="light kit",
        title=True,
        off_screen=False,
        scale=1.0,
        axis=True,
        notebook=False,
    ):
        """
        Initiates the plotter for the DFN.

        Parameters
        ----------
        window_size : tuple
            The size of the plot window.
        grid : bool
            Whether to add a grid to the plot.
        lighting : str
            The type of lighting to use.
        title : bool or str
            Whether to add a title to the plot.
        off_screen : bool
            Whether to plot off-screen.
        scale : float
            The scale of the plot.
        axis : bool
            Whether to add the axis to the plot.
        notebook : bool
            Whether to plot in a notebook. Set this to true when using Jupyter notebooks.

        Returns
        -------
        pl : pyvista.Plotter
            The plotter object.
        """
        logger.info("---------------------------------------")
        logger.info("Starting plotter...")
        logger.info("---------------------------------------")
        pl = pv.Plotter(
            window_size=window_size,
            lighting=lighting,
            off_screen=off_screen,
            notebook=notebook,
            title="AnDFN",
        )
        if axis:
            _ = pl.add_axes(
                line_width=2 * scale,
                cone_radius=0.3 + 0.1 * (1 - 1 / scale),
                shaft_length=0.7 + 0.3 * (1 - 1 / scale),
                tip_length=0.3 + 0.1 * (1 - 1 / scale),
                ambient=0.5,
                label_size=(0.2 / scale, 0.08 / scale),
                xlabel="X (E)",
                ylabel="Y (N)",
                zlabel="Z",
            )
        if grid:
            _ = pl.show_grid()
        # _ = pl.add_bounding_box(line_width=5, color='black', outline=False, culling='back')
        if isinstance(title, str):
            _ = pl.add_text(
                title,
                font_size=10 * scale,
                position="upper_left",
                color="k",
                shadow=True,
            )
            return pl
        if title:
            _ = pl.add_text(
                f"DFN: {self.label}",
                font_size=10 * scale,
                position="upper_left",
                color="k",
                shadow=True,
            )
        return pl

    def get_flow_fractures(self, cond=2e-1):
        q_dfn = self.get_dfn_discharge()
        fracs = []
        for i, f in enumerate(self.fractures):
            if f.get_total_discharge() / q_dfn > cond:
                fracs.append(f)
        return fracs

    def plot_input(self, pl=None, line_width=3.0):
        """
        Plots the input of the DFN, i.e. the fractures and elements.

        Parameters
        ----------
        pl : pyvista.Plotter, optional
            The plotter object to use. If None, a new plotter is created and shown.
        line_width : float
            The line width of the elements in the plot. Default is 3.0.

        Returns
        -------
        None
            The function plots the input of the DFN.
        """
        show = False
        if pl is None:
            pl = self.initiate_plotter()
            show = True
        self.plot_fractures(pl)
        labels = {}
        for s in self.structures:
            s.plot(pl)
            labels[f" {s.__class__.__name__}"] = STRUCTURES_COLOR[s._structure_type]
        if self.elements is not None:
            for e in self.elements:
                if e._type == 1: # Bounding circle
                    continue
                e.plot(pl, line_width=line_width, color=None)
                labels[f" {e.__class__.__name__}"] = ELEMENT_COLORS[e._type]

        # Add the legend
        # First convert the dict to a list of tuples
        labels = list(labels.items())
        pl.add_legend(
            labels,
            face="rectangle",
            loc="upper left",
        )

        if show:
            pl.show()

    def plot_fractures(
        self,
        pl,
        num_side=50,
        filled=True,
        color="#FFFFFF",
        opacity=1.0,
        show_edges=True,
        line_width=2.0,
        fracs=None,
    ):
        """
        Plots the fractures in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        num_side : int
            The number of sides to use for the fractures.
        filled : bool
            Whether to fill the fractures.
        color : str | tuple
            The color of the fractures.
        opacity : float
            The opacity of the fractures.
        show_edges : bool
            Whether to show the edges of the fractures.
        line_width : float
            The line width of the lines.
        fracs : list
            The list of fractures to plot. If None, all fractures are plotted.

        Returns
        -------
        fracs : list
            The list of fractures that have been plotted.
        """
        print_prog = False
        if fracs is None:
            fracs = self.fractures
            print_prog = True
        for i, f in enumerate(fracs):
            # plot the fractures
            pl.add_mesh(
                pv.Polygon(
                    center=f.center, radius=f.radius, normal=f.normal, n_sides=num_side, fill=filled
                ),
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                line_width=line_width,
            )
            if print_prog:
                logger.debug(f"Plotting fractures: {i + 1} / {len(self.fractures)}")

    def plot_fractures_flow_net(
        self, pl, lvs=20, n_points=2000, n_boundary_points=50, line_width=2, opacity=1.0, fill=True, contour_re=True, contour_im=True
    ):
        """
        Plots the flow net for the fractures in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        lvs : int
            The number of levels to contour for the flow net.
        n_points : int
            The number of points to use for the flow net.
        n_boundary_points : int
            The number of boundary points to use for the bounding circles.
        line_width : float
            The line width of the flow net.
        opacity : float
            The opacity of the fractures in the flownet.
        fill : bool
            Whether to fill the fractures in the flownet.
        contour_re : bool
            Whether to plot the equipotential lines (real part).
        contour_im : bool
            Whether to plot the streamlines (imaginary part).
        """


        # Check if the fractures have been consolidated
        if self.fractures_struc_array_hpc is None:
            self.consolidate_dfn(hpc=True)

        # Calculate the flow net for each fracture
        omegas, pnts_3d = hpc_get_flow_nets(
            self.fractures_struc_array_hpc,
            n_points,
            n_boundary_points,
            self.elements_struc_array_hpc,
        )

        # Calculate the levels for the contours
        lvs_re, lvs_im = get_lvs(lvs, omegas)

        # Create the PyVista meshes and plot for all fractures
        for i, pnts in enumerate(pnts_3d):
            logger.debug(f"Plotting flow net: {i + 1} / {len(self.fractures)}")
            surf = pv.PolyData(pnts)
            # Apply 2D Delaunay triangulation
            mesh = surf.delaunay_2d()
            # Add the mesh with scalar coloring
            if fill:
                pl.add_mesh(
                    mesh,
                    color="FFFFFF",
                    opacity=opacity,
                    show_edges=False,
                    line_width=line_width,
                )

            # Add contour lines, i.e. equipotential lines and streamlines
            contours_re = mesh.contour(isosurfaces=lvs_re, scalars=np.real(omegas[i]))
            if contours_re.n_points > 0 and contour_re:
                pl.add_mesh(
                    contours_re,
                    color="FF0000",
                    line_width=line_width,
                    opacity=opacity,
                )
            contours_im = mesh.contour(isosurfaces=lvs_im, scalars=np.imag(omegas[i]))
            if contours_im.n_points > 0 and contour_im:
                pl.add_mesh(
                    contours_im,
                    color="0000FF",
                    line_width=line_width,
                    opacity=opacity,
                )

        logger.debug("")

    def plot_fractures_head(
        self,
        pl,
        lvs=20,
        n_points=2000,
        n_boundary_points=50,
        line_width=2,
        opacity=1.0,
        color_map="jet",
        limits=None,
        contour=True,
        colorbar=True,
    ):
        """
        Plots the flow net for the fractures in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        lvs : int
            The number of levels to contour for the flow net.
        n_points : int
            The number of points to use for the flow net.
        n_boundary_points : int
            The number of boundary points to use for the bounding circles.
        line_width : float
            The line width of the flow net.
        opacity : float
            The opacity of the fractures in the flownet.
        color_map : str
            The color map to use for the flow net. For a constant color, use a string with the same value for the color.
        limits : list | tuple
            Custom limits for the flow net, overwrites the calculated limits.
        contour : bool
            Whether to plot the contour lines.
        colorbar : bool
            Whether to plot the color bar.
        """

        # Start timer
        start = time.time()

        # Check if the fractures have been consolidated
        if self.fractures_struc_array_hpc is None:
            self.consolidate_dfn(hpc=True)

        # Calculate the hydraulic head for each fracture and get the mapped 3d points
        heads, pnts_3d = hpc_get_heads(
            self.fractures_struc_array_hpc,
            n_points,
            n_boundary_points,
            self.elements_struc_array_hpc,
        )

        # Get the limits for the color map
        if limits is None:
            limits = [np.nanmin(heads), np.nanmax(heads)]
            logger.debug(f"Calculated limits for the color map: {limits}")

        # Calculate the levels for the contours
        lvs = np.linspace(limits[0], limits[1], lvs)

        # Create the PyVista meshes and plot for all fractures
        # Get the faces for the Delaunay triangulation (since we use the same triangulation for all fractures)
        s0 = time.time()
        faces = get_faces(pnts_3d[0])
        s1 = time.time()
        print(f"Creating faces took {s1 - s0:.2f} seconds.")

        meshes = []
        for i, pnts in enumerate(pnts_3d):
            poly = pv.PolyData(pnts, faces)
            poly.point_data["head"] = heads[i]
            meshes.append(poly)
        s2 = time.time()
        mesh = pv.merge(meshes)
        pl.add_mesh(
            mesh,
            scalars="head",
            cmap=color_map,
            opacity=opacity,
            show_edges=False,
            line_width=line_width,
            scalar_bar_args={"title": "Hydraulic Head"},
            name=f"head_{i}",
            clim=limits,
        )
        # Add contour lines, i.e. equipotential lines
        s3 = time.time()
        if contour:
            contours = mesh.contour(isosurfaces=lvs, scalars="head")
            if contours.n_points > 0:
                pl.add_mesh(
                    contours,
                    color="black",
                    line_width=line_width,
                    opacity=opacity,
                    clim=limits,
                )

        # Remove the color bar if not needed
        if not colorbar:
            pl.remove_scalar_bar()
        # Print the time it took to plot the flow net
        end = time.time()
        logger.info(f"Plotting hydraulic head took {end - start:.2f} seconds.")
        logger.debug("")

    def plot_elements(self, pl, color=None, elements=None, line_width=3.0, const_elements=False):
        """
        Plots the elements in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        color : str
            The color of the elements. If None, the default color is used.
        elements : list
            The list of elements to plot. If None, all elements are plotted.
        line_width : float
            The line width of the elements.
        """
        # Check if the elements have been stored in the DFN
        assert self.elements is not None and len(self.elements) > 0, (
            "The elements have not been stored in the DFN. Use the get_elements method."
        )
        # Plot the elements
        if elements is None:
            elements = self.elements
        if not isinstance(elements, list):
            elements = [elements]
        if const_elements:
            elements = [e for e in elements if isinstance(e, (ConstantHeadLine, Well))]
        for i, e in enumerate(elements):
            e.plot(pl, line_width=line_width, color=color)
            logger.debug(f"Plotting elements: {i + 1} / {len(self.elements)}")
        logger.debug("")

    def plot_sparse_matrix(self, save=False, name="sparse_matrix.png"):
        """
        Plots the sparse matrix of the DFN.

        Parameters
        ----------
        save : bool
            Whether to save the plot.
        name : str
            The name of the plot.

        Returns
        -------
        None
        """
        # Check if matplotlib is installed
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required to plot the sparse matrix. Please install matplotlib."
            )

        # Check if the discharge matrix has been built
        if self.discharge_matrix is None:
            self.build_discharge_matrix()

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        # set background color of plot surface
        fig.set_facecolor("black")
        ax.set_facecolor("black")
        # set tick and title color and axis box
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.title.set_color("white")
        for spine in ["top", "left", "right", "bottom"]:
            ax.spines[spine].set_color("white")
        ax.spy(self.discharge_matrix, markersize=0.5, color="white")
        # Equal axis
        ax.set_aspect("equal")

        num_entries = self.discharge_matrix.nnz
        num_zeros = self.discharge_matrix.shape[0] * self.discharge_matrix.shape[1]
        # title
        ax.set_title(
            f"Sparse matrix of the DFN\nNumber of fractures: {self.number_of_fractures()}, Number of elements: {self.number_of_elements()}"
            f"\nNumber of entries: {num_entries}, Number of zeros: "
            f"{num_zeros - num_entries}\nFilled percentage: {num_entries / num_zeros * 100:.2f}%"
        )
        if save:
            plt.savefig(name)
        plt.show()

    def plot_ncoef(self, save=False, name="ncoef.png"):
        """
        Plots the number of coefficients for the elements in the DFN.

        Parameters
        ----------
        save : bool
            Whether to save the plot.
        name : str
            The name of the plot.

        Returns
        -------
        None
        """
        # Check if matplotlib is installed
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required to plot the ncoef plot. Please install matplotlib."
            )

        ncoef = []
        for e in self.elements:
            ncoef.append(e.ncoef)
        fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
        nbins = np.ceil((max(ncoef) - min(ncoef)) / 5).astype(int)
        counts, edges, bars = ax.hist(
            ncoef, bins=nbins, color="grey", edgecolor="black"
        )
        ax.set_title("Number of coefficients for the elements in the DFN")
        ax.set_xlabel("Number of coefficients")
        ax.set_ylabel("Number of elements")
        ax.set_yscale("log")
        ax.bar_label(bars)
        if save:
            plt.savefig(name)
        plt.show()

    ####################################################################################################################
    #                    Streamline tracking functions                                                                 #
    ####################################################################################################################
    def plot_streamline_tracking(
        self,
        pl,
        z0,
        frac,
        ds=1e-2,
        max_length=1000,
        line_width=2.0,
        elevation=0.0,
        remove_false=True,
        color="black",
        backward=False,
    ):
        """
        Plots the streamline tracking for a given fracture.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        z0 : complex | np.ndarray
            The starting point for the streamline tracking.
        frac : Fracture
            The fracture where to start the streamline tracking.
        ds : float
            The step size for the streamline tracking.
        max_length : int
            The maximum length of the streamline.
        line_width : float
            The line width of the streamlines.
        elevation : float | np.ndarray
            The elevation of the starting point.
        remove_false : bool
            Whether to remove the streamlines that exit the DFN on flase locations.
        """
        if isinstance(z0, complex):
            z0 = np.array([z0])

        if isinstance(elevation, float):
            elevation = np.array([elevation])

        logger.info("---------------------------------------")
        logger.info("Starting streamline tracking...")
        logger.info("---------------------------------------")
        logger.debug(f"Number of starting points: {len(z0)}")
        logger.debug(f"Number of elevations: {len(elevation)}")

        streamlines = []
        streamlines_frac = []
        velocities = []
        elements = []
        for i, z in enumerate(z0):  # type: int, complex
            for j, e in enumerate(elevation):
                logger.debug(f"Tracing streamline: {i + 1} / {len(z0)}")
                streamline, streamline_frac, velocity, element = (
                    self.streamline_tracking(z, frac, e, ds, max_length, backward)
                )
                streamlines.append(streamline)
                streamlines_frac.append(streamline_frac)
                velocities.append(velocity)
                elements.append(element)

        # Concatenate streamlines list to ndarray
        for i, s in enumerate(streamlines):
            streamline_3d = []
            if len(s) == 0:
                continue
            if elements[i] is False and remove_false:
                continue
            for ii, ss in enumerate(s):
                psi_3d = gf.map_2d_to_3d(np.array(ss), streamlines_frac[i][ii])
                streamline_3d.append(psi_3d)
            streamline_3d = np.concatenate(streamline_3d)
            pl.add_mesh(
                pv.MultipleLines(streamline_3d), color=color, line_width=line_width
            )

        return streamlines, streamlines_frac, velocities, elements

    def streamline_tracking(self, z0, frac, elevation, ds, max_length, backward):
        """
        Function that tracks the streamlines in a fracture.

        Parameters
        ----------
        z0 : complex
            Starting point for streamline tracking
        elevation : float
            Elevation of the starting point
        frac: Fracture
            The fracture where to start the streamline tracking
        ds : float
            Step size for the streamline tracking
        max_length : int
            Maximum length of the streamline
        backward : bool
            Whether to track the streamline backward or forward

        Returns
        -------
        streamlines: np.ndarray
            Array with streamline
        """
        # Crete empty ndarray
        streamline = []
        streamline_frac = []
        velocity = []

        # Start the tracking process
        cond = True
        element = False
        z_start = z0  # set the start of the streamline on the element, while the computations for this point is made
        # just outside the boundary of the element
        while cond:
            # set the ds proportional to the fracture size
            ds_frac = frac.radius / ds
            # get the current number of points
            length = sum([len(s) for s in streamline])
            # Start the tracking process
            psi = [z_start]
            w = [np.abs(frac.calc_w(z0))]
            discharge_elements = frac.get_discharge_elements()

            # get the next points
            z1 = self.runge_kutta(z0, frac, ds_frac, backward)
            if np.isnan(np.real(z1)) or np.isnan(np.imag(z1)):
                break
            z3, element = self.check_streamline_exit(
                z0, z1, discharge_elements, frac, backward
            )
            while z3 is False:
                psi.append(z1)
                w.append(np.abs(frac.calc_w(z1)))
                z0 = z1
                z1 = self.runge_kutta(z0, frac, ds_frac, backward)
                if np.isnan(np.real(z1)) or np.isnan(np.imag(z1)):
                    z3 = z0
                    break
                z3, element = self.check_streamline_exit(
                    z0, z1, discharge_elements, frac, backward
                )
                if len(psi) > max_length - length:
                    z3 = z1
                    break
            psi.append(z3)
            w.append(np.abs(frac.calc_w(z3)))

            streamline.append(psi)
            streamline_frac.append(frac)
            velocity.append(w)

            if isinstance(element, Intersection):
                z3d = gf.map_2d_to_3d(z3, frac)
                frac_old = frac
                if frac == element.frac0:
                    frac = element.frac1
                else:
                    frac = element.frac0
                z0, z_start, elevation = self.get_exit_intersection(
                    z3d, element, frac, frac_old, elevation
                )
            else:
                cond = False

        return streamline, streamline_frac, velocity, element

    @staticmethod
    def check_streamline_exit(z0, z1, discharge_elements, frac, backward):
        """
        Function that checks if the streamline has exited the DFN
        """
        for e in discharge_elements:
            if isinstance(e, Intersection):
                # if ((e.q < 0 and e.frac0 == frac) or (e.q > 0 and e.frac1 == frac)) ^ backward:
                #    continue
                z2 = e.check_chi_crossing(z0, z1, frac)
            else:
                # if (e.q < 0) ^ backward:
                #    continue
                z2 = e.check_chi_crossing(z0, z1)
            if np.isnan(np.real(z2)):
                return z1, False
            if z2 is not False:
                return z2, e
        return False, False

    @staticmethod
    def get_exit_intersection(z3d, element, frac, frac_old, elevation, dchi=1e-6):
        if frac == element.frac0:
            endpoints = element.endpoints0
        else:
            endpoints = element.endpoints1
        z = gf.map_3d_to_2d(z3d, frac)
        chi0 = gf.map_z_line_to_chi(z, endpoints)
        chi1 = np.conj(chi0)
        z0 = gf.map_chi_to_z_line(chi0 * (1 + dchi), endpoints)
        z1 = gf.map_chi_to_z_line(chi1 * (1 + dchi), endpoints)
        w0 = frac.calc_w(z0)
        w1 = frac.calc_w(z1)

        # Magnitude
        abs_w0 = np.abs(w0)
        abs_w1 = np.abs(w1)

        # check angles between w0, w1 and z-z0, z-z1, using the dot product
        divide = abs_w0 / (abs_w0 + abs_w1)
        pointz0 = gf.map_2d_to_3d(z0, frac)

        # map on direction of normal
        nz0 = np.dot((pointz0 - z3d), frac_old.normal)
        if nz0 < 0:
            up = z1
            down = z0
        else:
            up = z0
            down = z1
            divide = 1 - divide

        # Check if elevation is below the divide
        if elevation < divide:
            elevation /= divide  # new elevation
            return down, z0, elevation * 0 + 0.5

        elevation = (elevation - divide) / (1 - divide)
        return up, z0, elevation * 0 + 0.5

    @staticmethod
    def runge_kutta(z0, frac, ds, backward, tolerance=1e-6, max_it=10):
        """
        Runge-Kutta method for streamline tracing.

        Parameters
        ----------
        z0 : complex
            The initial point.
        frac : Fracture
            The fracture where the streamline is traced.
        ds : float
            The step size.
        backward : bool
            Whether to trace the streamline backward.
        tolerance : float
            The tolerance for the error.
        max_it : int
            The maximum number of iterations.


        Returns
        -------
        z1 : complex
            The point at the end of the streamline.
        """
        if backward:
            ds = -ds
        w0 = frac.calc_w(z0)
        if np.isnan(np.real(w0)):
            return np.nan + np.nan * 1j
        z1 = z0 + np.conj(w0) / np.abs(w0) * ds
        dz = np.abs(z1 - z0)
        it = 0
        while dz > tolerance and it < max_it:
            w1 = frac.calc_w(z1)
            if np.isnan(np.real(w1)):
                break
            z2 = z0 + np.conj(w0 + w1) / np.abs(w0 + w1) * ds
            dz = np.abs(z2 - z1)
            z1 = z2
            it += 1

        return z1

    @staticmethod
    def get_travel_time_and_length(streamline, velocity):
        """
        Returns the travel time for a streamline.

        Parameters
        ----------
        streamline : list | ndarray
            The streamline.
        velocity : ndarray
            The velocity along the streamline.

        Returns
        -------
        time : float
            The travel time for the streamline.
        length : float
            The length of the streamline.
        """
        if isinstance(streamline[0], list):
            streamline = np.concat(streamline)
            velocity = np.concatenate(velocity)
        if isinstance(streamline, list):
            streamline = np.array(streamline)
        if isinstance(velocity, list):
            velocity = np.array(velocity)
        time = np.sum(np.abs(streamline[1:] - streamline[:-1]) / velocity[:-1])
        length = np.sum(np.abs(streamline[1:] - streamline[:-1]))
        return time, length
