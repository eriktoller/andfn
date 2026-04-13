"""
Notes
-----
This module contains the DFN class.

The DFN class is the main class for the AnDFN package. It contains the fractures and elements of the DFN and methods to
solve the DFN.
"""

import numpy as np
import numba as nb
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
    MAX_ELEMENTS,
    MAX_NCOEF,
)

# Custom colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation

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


@nb.njit(parallel=True, cache=True)
def consolidate_fractures_numba(
    fracture_struc_array,
    fracture_index_array,
    ids,
    tvals,
    radii,
    centers,
    normals,
    xvecs,
    yvecs,
    elements_ids,
    nelements,
    constants,
):
    nf = ids.shape[0]

    for i in nb.prange(nf):
        fracture_struc_array["_id"][i] = ids[i]
        fracture_struc_array["t"][i] = tvals[i]
        fracture_struc_array["radius"][i] = radii[i]
        fracture_struc_array["center"][i] = centers[i]
        fracture_struc_array["normal"][i] = normals[i]
        fracture_struc_array["x_vector"][i] = xvecs[i]
        fracture_struc_array["y_vector"][i] = yvecs[i]

        n = nelements[i]
        fracture_struc_array["elements"][i, :n] = elements_ids[i, :n]
        fracture_struc_array["nelements"][i] = n
        fracture_struc_array["constant"][i] = constants[i]

        fracture_index_array["_id"][i] = ids[i]


@nb.njit(parallel=True, cache=True)
def consolidate_elements_numba(
    elements_struc_array,
    elements_index_array,
    element_id,
    element_type,
    frac0,
    frac1,
    radius,
    center,
    head,
    phi,
    q,
    ncoef,
    nint,
    error,
    endpoints0,
    endpoints1,
    thetas,
    coef,
    old_coef,
    dpsi_corr,
):
    """
    Parallel consolidation of elements into structured arrays.
    All inputs are Struct-of-Arrays (SoA).
    """

    ne = element_id.shape[0]

    for i in nb.prange(ne):
        # ---- scalar assignments ----
        elements_struc_array["_id"][i] = element_id[i]
        elements_struc_array["_type"][i] = element_type[i]
        elements_struc_array["frac0"][i] = frac0[i]
        elements_struc_array["frac1"][i] = frac1[i]
        elements_struc_array["radius"][i] = radius[i]
        elements_struc_array["center"][i] = center[i]
        elements_struc_array["head"][i] = head[i]
        elements_struc_array["phi"][i] = phi[i]
        elements_struc_array["q"][i] = q[i]
        elements_struc_array["ncoef"][i] = ncoef[i]
        elements_struc_array["nint"][i] = nint[i]
        elements_struc_array["error"][i] = error[i]

        # ---- endpoints ----
        elements_struc_array["endpoints0"][i, 0] = endpoints0[i, 0]
        elements_struc_array["endpoints0"][i, 1] = endpoints0[i, 1]
        elements_struc_array["endpoints1"][i, 0] = endpoints1[i, 0]
        elements_struc_array["endpoints1"][i, 1] = endpoints1[i, 1]

        # ---- coefficient arrays ----
        nc = ncoef[i]

        for k in range(nc):
            elements_struc_array["coef"][i, k] = coef[i, k]
            elements_struc_array["old_coef"][i, k] = old_coef[i, k]

        for k in range(2 * nc):
            elements_struc_array["thetas"][i, k] = thetas[i, k]
            elements_struc_array["dpsi_corr"][i, k] = dpsi_corr[i, k]

        # ---- index array (numeric only!) ----
        elements_index_array["_id"][i] = element_id[i]  # element id
        elements_index_array["_type"][i] = element_type[i]  # type


class DFN(Constants):
    def __init__(self, label, discharge_int=50, **kwargs):
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

        self.ntype_element = np.zeros(6, dtype=int)  # number of elements of each type

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

        # Initialize the cell tree
        self._tree = None

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

        The following constants can be changed:
            - RHO: Density of water in kg/m^3
            - G: Gravitational acceleration in m/s^2
            - PI: Pi
            - SE_FACTOR: Shortening element length factor
            - MAX_ITERATIONS: Maximum number of iterations
            - MAX_ERROR: Maximum error
            - MAX_NCOEF: Maximum number of coefficients
            - COEF_INCREASE: Coefficient increase factor
            - COEF_RATIO: Coefficient ratio
            - MAX_ELEMENTS: Maximum number of elements
            - NCOEF: Number of coefficients
            - NINT: Number of integration points
            - NUM_THREADS: Number of threads to use for Numba (default -1 = use all available threads)

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
    def save_dfn(self, filename, overwrite=False):
        """
        Saves the DFN to a h5 file.

        Parameters
        ----------
        filename : str
            The name of the file to save the DFN to.
        """
        # Check the filename extension
        ext = os.path.splitext(filename)[1]
        if ext == "":
            filename += ".h5"
        elif ext not in [".h5"]:
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported extensions are .h5."
            )

        # Check if the file already exists
        if os.path.exists(f"{filename}") and not overwrite:
            logger.warning(
                f"The file {filename} already exists. To overwrite the existing file set the argument 'overwrite' to True."
            )
            return

        # Check if the elements and fractures have been consolidated
        if self.elements_struc_array is None or self.fractures_struc_array is None:
            self.consolidate_dfn()

        # Save the elements
        with h5py.File(f"{filename}", "w") as hf:
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

        logger.info(f"Saved DFN to {filename}")

    def load_dfn(self, filename):
        """
        Loads the DFN from a h5 file.

        Parameters
        ----------
        filename : str
            The name of the file to load the DFN from.
        """
        # Check the filename extension
        ext = os.path.splitext(filename)[1]
        if ext == "":
            filename += ".h5"
        elif ext not in [".h5"]:
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported extensions are .h5."
            )

        # Check if the file exists
        if not os.path.exists(f"{filename}"):
            raise FileNotFoundError(f"The file {filename} does not exist.")

        logger.info(f"Loading DFN from {filename}")

        with h5py.File(f"{filename}", "r") as hf:
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
                        aperture=hf[f"fractures/properties/aperture/{i}"][()]
                        if f"fractures/properties/aperture/{i}" in hf
                        else 1e-6,
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

    def consolidate_dfn2(self, hpc=False):
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
        fractures_struc_array = np.empty(self.number_of_fractures, dtype=f_dtype)
        fractures_index_array = np.empty(
            self.number_of_fractures, dtype=fracture_index_dtype
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

    def consolidate_dfn(self, hpc=False):
        if self.elements is None:
            self.get_elements()

        elements = self.elements
        fractures = self.fractures

        ne = len(elements)
        nf = self.number_of_fractures

        if hpc:
            e_dtype = element_dtype_hpc
            f_dtype = fracture_dtype_hpc
        else:
            e_dtype = element_dtype
            f_dtype = fracture_dtype

        elements_struc_array = np.empty(ne, dtype=e_dtype)
        elements_index_array = np.empty(ne, dtype=element_index_dtype)

        for name in elements_struc_array.dtype.names:
            if np.issubdtype(element_dtype_hpc[name], np.int_):
                elements_struc_array[name][0] = -1
            elif np.issubdtype(element_dtype_hpc[name], np.float64):
                elements_struc_array[name][0] = np.nan
            elif np.issubdtype(element_dtype_hpc[name], np.complex128):
                elements_struc_array[name][0] = np.nan + 1j * np.nan
            elif name == "thetas" or name == "dpsi_corr":
                elements_struc_array[name][0] = np.zeros(
                    MAX_NCOEF * 2, dtype=np.float64
                )
            elif name == "coef" or name == "old_coef":
                elements_struc_array[name][0] = np.zeros(MAX_NCOEF, dtype=np.complex128)
            elif name == "endpoints0" or name == "endpoints1":
                elements_struc_array[name][0] = np.full(
                    2, np.nan + 1j * np.nan, dtype=np.complex128
                )

        ne = len(elements)

        element_id = np.empty(ne, dtype=np.int64)
        element_type = np.empty(ne, dtype=np.int64)
        element_frac0 = np.empty(ne, dtype=np.int64)
        element_frac1 = np.empty(ne, dtype=np.int64)
        element_radius = np.empty(ne, dtype=np.float64)
        element_head = np.empty(ne, dtype=np.float64)
        element_phi = np.empty(ne, dtype=np.float64)
        element_q = np.empty(ne, dtype=np.float64)
        element_ncoef = np.empty(ne, dtype=np.int64)
        element_nint = np.empty(ne, dtype=np.int64)
        element_error = np.empty(ne, dtype=np.float64)
        element_center = np.empty(ne, dtype=np.complex128)
        element_endpoints0 = np.empty((ne, 2), dtype=np.complex128)
        element_endpoints1 = np.empty((ne, 2), dtype=np.complex128)
        element_thetas = np.empty((ne, 2 * MAX_NCOEF), dtype=np.float64)
        element_coef = np.empty((ne, MAX_NCOEF), dtype=np.complex128)
        element_old_coef = np.empty((ne, MAX_NCOEF), dtype=np.complex128)
        element_dpsi_corr = np.empty((ne, 2 * MAX_NCOEF), dtype=np.float64)

        for i, e in enumerate(elements):
            element_id[i] = e._id
            element_type[i] = e._type
            element_frac0[i] = e.frac0._id if e.frac0 is not None else -1
            element_frac1[i] = (
                e.frac1._id if hasattr(e, "frac1") and e.frac1 is not None else -1
            )
            element_radius[i] = e.radius if hasattr(e, "radius") else 0.0
            element_head[i] = e.head if hasattr(e, "head") else 0.0
            element_phi[i] = e.phi if hasattr(e, "phi") else 0.0
            element_q[i] = e.q if hasattr(e, "q") else 0.0
            element_ncoef[i] = e.ncoef if hasattr(e, "ncoef") else 0
            element_nint[i] = e.nint if hasattr(e, "nint") else 0
            element_error[i] = e.error if hasattr(e, "error") else 0.0
            element_center[i] = e.center if hasattr(e, "center") else 0.0
            element_endpoints0[i] = e.endpoints0 if hasattr(e, "endpoints0") else 0.0
            element_endpoints1[i] = e.endpoints1 if hasattr(e, "endpoints1") else 0.0
            element_coef[i, : len(e.coef)] = e.coef if hasattr(e, "coef") else 0.0

        consolidate_elements_numba(
            elements_struc_array,
            elements_index_array,
            element_id,
            element_type,
            element_frac0,
            element_frac1,
            element_radius,
            element_center,
            element_head,
            element_phi,
            element_q,
            element_ncoef,
            element_nint,
            element_error,
            element_endpoints0,
            element_endpoints1,
            element_thetas,
            element_coef,
            element_old_coef,
            element_dpsi_corr,
        )

        fractures_struc_array = np.empty(nf, dtype=f_dtype)
        fractures_index_array = np.empty(nf, dtype=fracture_index_dtype)

        nf = len(fractures)

        ids = np.empty(nf, dtype=np.int64)
        tvals = np.empty(nf, dtype=np.float64)
        radii = np.empty(nf, dtype=np.float64)
        centers = np.empty((nf, 3), dtype=np.float64)
        normals = np.empty((nf, 3), dtype=np.float64)
        xvecs = np.empty((nf, 3), dtype=np.float64)
        yvecs = np.empty((nf, 3), dtype=np.float64)
        constants = np.empty(nf, dtype=np.float64)
        labels = np.empty(nf, dtype=np.str_)

        elements_ids = np.zeros((nf, MAX_ELEMENTS), dtype=np.int64)
        nelements = np.zeros(nf, dtype=np.int64)

        for i, f in enumerate(fractures):
            ids[i] = f._id
            tvals[i] = f.t
            radii[i] = f.radius
            centers[i] = f.center
            normals[i] = f.normal
            xvecs[i] = f.x_vector
            yvecs[i] = f.y_vector
            constants[i] = f.constant
            labels[i] = f.label

            ne = len(f.elements)
            nelements[i] = ne
            for k in range(ne):
                elements_ids[i, k] = f.elements[k]._id

        consolidate_fractures_numba(
            fractures_struc_array,
            fractures_index_array,
            ids,
            tvals,
            radii,
            centers,
            normals,
            xvecs,
            yvecs,
            elements_ids,
            nelements,
            constants,
        )

        if hpc:
            self.elements_struc_array_hpc = elements_struc_array
            self.fractures_struc_array_hpc = fractures_struc_array
        else:
            self.elements_struc_array = elements_struc_array
            self.fractures_struc_array = fractures_struc_array

        self.elements_index_array = elements_index_array
        self.fractures_index_array = fractures_index_array

        logger.info(
            f"Consolidated DFN with {nf} fractures and {len(elements_struc_array)} elements."
        )

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
    @property
    def tree(self):
        """
        Gets the cell tree for the DFN.

        Returns
        -------
        tree : scipy.spatial.KDTree | None
            The cell tree for the DFN.
        """
        if self._tree is None and self.number_of_fractures > 0:
            self._tree = sp.spatial.KDTree(np.array([f.center for f in self.fractures]))
        return self._tree

    @property
    def number_of_fractures(self):
        """
        Returns the number of fractures in the DFN.
        """
        return len(self.fractures)

    def number_of_elements(self, element_type=None):
        """
        Gets the number of elements in the DFN.

        Parameters
        ----------
        element_type : type, optional
            The type of elements to count. If None, counts all elements. The default is None.

        Returns
        -------
        int
            The number of elements in the DFN.
        """
        if self.elements is None:
            self.get_elements()

        if element_type is None:
            return len(self.elements)
        else:
            if isinstance(element_type, str):
                element_type = {
                    "intersection": Intersection,
                    "bounding circle": BoundingCircle,
                    "well": Well,
                    "constant head line": ConstantHeadLine,
                    "impermeable circle": ImpermeableCircle,
                    "impermeable line": ImpermeableLine,
                }.get(element_type.lower(), None)
                if element_type is None:
                    raise ValueError(
                        f"Unsupported element type: {element_type}. Supported types are 'intersection', 'bounding circle', 'well', 'constant head line', 'impermeable circle', and 'impermeable line'."
                    )
            if element_type not in [
                Intersection,
                BoundingCircle,
                Well,
                ConstantHeadLine,
                ImpermeableCircle,
                ImpermeableLine,
            ]:
                raise ValueError(
                    f"Unsupported element type: {element_type}. Supported types are 'intersection', 'bounding', 'well', 'const_head_line', 'imp_circle', and 'imp_line'."
                )
            if not isinstance(element_type, type):
                raise ValueError(
                    f"Element type must be a type. Got {type(element_type)} instead."
                )
            return len([e for e in self.elements if isinstance(e, element_type)])

    def get_elements_org(self):
        """
        Gets the elements from the fractures and add store them in the DFN.
        """
        # reset the discharge matrix and elements
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None
        self.lup = None

        elements = []
        for f in self.fractures:
            if f.elements is None or len(f.elements) == 0:
                continue
            for e in f.elements:
                if e not in elements:
                    elements.append(e)
        # sort the elements by their type
        intersections = [e for e in elements if getattr(e, "_type", None) == 0]
        bounding = [e for e in elements if getattr(e, "_type", None) == 1]
        wells = [e for e in elements if getattr(e, "_type", None) == 2]
        const_head_lines = [e for e in elements if getattr(e, "_type", None) == 3]
        imp_circle = [e for e in elements if getattr(e, "_type", None) == 4]
        imp_line = [e for e in elements if getattr(e, "_type", None) == 5]

        elements = (
            bounding + imp_circle + imp_line + const_head_lines + wells + intersections
        )
        self.elements = elements
        self.ntype_element = np.array(
            [
                len(intersections),
                len(bounding),
                len(wells),
                len(const_head_lines),
                len(imp_circle),
                len(imp_line),
            ],
            dtype=int,
        )
        logger.info(f"Added {len(self.elements)} elements to the DFN.")

        for e in self.elements:
            e.set_id(self.elements.index(e))

    def get_elements(self):
        """
        Gets the elements from the fractures and stores them in the DFN.
        """

        # reset
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None
        self.lup = None

        # ---- collect unique elements (O(n)) ----
        elements = []
        element_set = set()

        for f in self.fractures:
            fes = f.elements
            if not fes:
                continue
            for e in fes:
                if e not in element_set:
                    element_set.add(e)
                    elements.append(e)

        # ---- classify in ONE pass ----
        intersections = []
        bounding = []
        wells = []
        const_head_lines = []
        imp_circle = []
        imp_line = []

        for e in elements:
            t = e._type
            if t == 0:
                intersections.append(e)
            elif t == 1:
                bounding.append(e)
            elif t == 2:
                wells.append(e)
            elif t == 3:
                const_head_lines.append(e)
            elif t == 4:
                imp_circle.append(e)
            elif t == 5:
                imp_line.append(e)

        # ---- final ordering (unchanged semantics) ----
        self.elements = (
            bounding + imp_circle + imp_line + const_head_lines + wells + intersections
        )

        self.ntype_element = np.array(
            [
                len(intersections),
                len(bounding),
                len(wells),
                len(const_head_lines),
                len(imp_circle),
                len(imp_line),
            ],
            dtype=int,
        )

        logger.info(f"Added {len(self.elements)} elements to the DFN.")

        # ---- assign ids in O(n) ----
        for i, e in enumerate(self.elements):
            e.set_id(i)

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
        fracture : Fracture | list
            The fracture to delete from the DFN.
        """
        # Delete all the elements that are connected to the fracture
        if isinstance(fracture, Fracture):
            fracture = [fracture]

        for f in fracture:
            f.delete_all_elements()

        # Remove fractures in one operation
        self.fractures = [f for f in self.fractures if f not in fracture]

        # Reset dependent structures
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None

        # Reassign IDs efficiently
        for i, f in enumerate(self.fractures):
            f.set_id(i)

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

        # Update the elements
        self.get_elements()

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
        remove_tolerance=-1,
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
        remove_tolerance : float, optional
            The tolerance to use when removing isolated fractures. The default is -1 (no tolerance).

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
        if strike_str is not None and dip_str is not None:
            orientation_method = gf.convert_strike_dip_to_normal
            st_str = strike_str
            dp_str = dip_str
        elif trend_str is not None and plunge_str is not None:
            orientation_method = gf.convert_trend_plunge_to_normal
            st_str = trend_str
            dp_str = plunge_str
        else:
            raise ValueError("Either strike/dip or trend/plunge must be provided.")
        if e_str is not None and e_str not in data_file.columns:
            raise ValueError(f"Aperture column '{e_str}' not found in the data file.")

        # Extract the data from the file
        radius_arr = data_file[radius_str].to_numpy()
        st_arr = data_file[st_str].to_numpy()
        dp_arr = data_file[dp_str].to_numpy()
        center_arr = data_file[[x_str, y_str, z_str]].to_numpy()
        transmissivity_arr = data_file[t_str].to_numpy()
        aperture_arr = data_file[e_str].to_numpy()

        normals = np.array(
            [orientation_method(st, dp) for st, dp in zip(st_arr, dp_arr)]
        )

        frac = [
            Fracture(
                f"{i}",
                transmissivity_arr[i],
                radius_arr[i],
                center_arr[i],
                normals[i],
                aperture_arr[i],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
            )
            for i in range(len(data_file))
        ]
        # sort the fracture by radius, starting with the largest
        frac.sort(key=lambda f: f.radius, reverse=True)
        centers = np.array([f.center for f in frac])
        tree = sp.spatial.KDTree(centers)

        if starting_frac is not None:
            fracs = gf.get_connected_fractures(
                frac,
                self.constants["SE_FACTOR"],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
                fracture_surface=frac[starting_frac],
                tolerance=remove_tolerance,
            )
        else:
            fracs = gf.get_fracture_intersections(
                frac,
                self.constants["SE_FACTOR"],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
                tolerance=remove_tolerance,
                tree=tree,
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

    def get_fracture_intersections(
        self, ncoef=5, nint=10, new_frac=None, se_factor=None
    ):
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
        se_factor : float
            The shortening element factor to use for the intersection elements.
        """

        if se_factor is None:
            se_factor = self.constants["SE_FACTOR"]

        # Compute intersections between fractures only for frac_surface
        if new_frac is not None:
            fr = new_frac
            for k in range(len(self.fractures)):
                fr2 = self.fractures[k]
                if fr == fr2:
                    continue
                endpoints0, endpoints1 = gf.fracture_intersection(fr, fr2)
                if endpoints0 is not None:
                    endpoints0 = gf.shorten_line(endpoints0, se_factor)
                    endpoints1 = gf.shorten_line(endpoints1, se_factor)
                    Intersection(
                        f"{fr.label}_{fr2.label}",
                        endpoints0,
                        endpoints1,
                        fr,
                        fr2,
                        ncoef,
                        nint,
                    )

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
                        endpoints0 = gf.shorten_line(endpoints0, se_factor)
                        endpoints1 = gf.shorten_line(endpoints1, se_factor)
                        Intersection(
                            f"{fr.label}_{fr2.label}",
                            endpoints0,
                            endpoints1,
                            fr,
                            fr2,
                            ncoef,
                            nint,
                        )

        # Update the elements in the DFN
        self.get_elements()

    @property
    def center(self):
        """
        Gets the center of the DFN.
        """
        center = np.array([0.0, 0.0, 0.0])
        for f in self.fractures:
            center += f.center
        return center / len(self.fractures)

    @property
    def size(self):
        """
        Gets the size of the DFN.
        """
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        min_z = np.inf
        max_z = -np.inf

        for f in self.fractures:
            c = f.center
            r = f.radius
            min_x = min(min_x, c[0] - r)
            max_x = max(max_x, c[0] + r)
            min_y = min(min_y, c[1] - r)
            max_y = max(max_y, c[1] + r)
            min_z = min(min_z, c[2] - r)
            max_z = max(max_z, c[2] + r)

        return np.array([max_x - min_x, max_y - min_y, max_z - min_z])

    def smallest_fracture_radius(self):
        """
        Gets the smallest fracture radius in the DFN.
        """
        min_radius = np.inf
        for f in self.fractures:
            if f.radius < min_radius:
                min_radius = f.radius
        return min_radius

    def set_constant_head_boundary(
        self,
        center,
        normal,
        radius,
        head,
        label="Constant Head Boundary",
        ncoef=5,
        nint=10,
        se_factor=None,
        tolerance=-1,
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
        ncoef : int
            The number of coefficients to use for the constant head boundary elements.
        nint : int
            The number of integration points to use for the constant head boundary elements.
        se_factor : float
            The shortening element factor to use for the constant head boundary elements.
        tolerance : float, optional
            The tolerance to use when checking if the constant head boundary intersects with the fractures. The default is -1 (no tolerance).
        """
        if se_factor is None:
            se_factor = self.constants["SE_FACTOR"]
        gf.set_head_boundary(
            self.fractures,
            ncoef,
            nint,
            head,
            center,
            radius,
            normal,
            label,
            se_factor,
            tolerance,
        )

    def set_impermeable_boundary(
        self,
        center,
        normal,
        radius,
        label="Impermeable Boundary",
        ncoef=5,
        nint=10,
        se_factor=None,
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
        ncoef : int
            The number of coefficients to use for the constant head boundary elements.
        nint : int
            The number of integration points to use for the constant head boundary elements.
        se_factor : float
            The shortening element factor to use for the constant head boundary elements.
        """
        if se_factor is None:
            se_factor = self.constants["SE_FACTOR"]
        gf.set_impermeable_boundary(
            self.fractures,
            ncoef,
            nint,
            center,
            radius,
            normal,
            label,
            se_factor,
        )

    def shorten_elements(self, factor=None):
        """
        Shortens the elements in the DFN based on the SE_FACTOR constant. Only applicable to line elements.

        Parameters
        ----------
        factor : float, optional
            The factor to shorten the elements by. If None, uses the SE_FACTOR constant. If provided, overrides the SE_FACTOR constant.
        """
        if self.elements is None:
            self.get_elements()

        if factor is not None:
            self.constants["SE_FACTOR"] = factor

        for e in self.elements:
            if isinstance(e, (ConstantHeadLine, ImpermeableLine)):
                endpoints = e.endpoints0
                new_endpoints = gf.shorten_line(endpoints, self.constants["SE_FACTOR"])
                e.endpoints0 = new_endpoints
            elif isinstance(e, Intersection):
                endpoints0 = e.endpoints0
                endpoints1 = e.endpoints1
                new_endpoints0 = gf.shorten_line(
                    endpoints0, self.constants["SE_FACTOR"]
                )
                new_endpoints1 = gf.shorten_line(
                    endpoints1, self.constants["SE_FACTOR"]
                )
                e.endpoints0 = new_endpoints0
                e.endpoints1 = new_endpoints1

    def remove_small_elements(self, min_length):
        """
        Removes elements from the DFN that are smaller than the specified minimum length.

        Parameters
        ----------
        min_length : float
            The minimum length of the elements to keep.
        """
        if self.elements is None:
            self.get_elements()

        cnt = 0
        for f in self.fractures:
            elements_to_remove = []
            for e in f.elements:
                if isinstance(e, Intersection):
                    length = np.linalg.norm(e.endpoints0[1] - e.endpoints0[0])
                    if length < min_length:
                        elements_to_remove.append(e)
                elif isinstance(e, (ConstantHeadLine, ImpermeableLine)):
                    length = np.linalg.norm(e.endpoints0[1] - e.endpoints0[0])
                    if length < min_length:
                        elements_to_remove.append(e)
            for e in elements_to_remove:
                f.delete_element(e)
                del e
                cnt += 1
        logger.info(f"Removed {cnt} elements smaller than {min_length} units.")

        # Check if fractures have no elements and remove them
        flen = len(self.fractures)
        fracs = gf.remove_isolated_fractures(self.fractures)
        self.fractures = []
        self.add_fracture(fracs)
        rlen = flen - len(self.fractures)
        if rlen > 0:
            logger.info(f"Removed {rlen} fractures with no elements.")

        # Update the elements in the DFN
        self.get_elements()

    def check_connectivity(self):
        """
        Checks the connectivity of the DFN and removes isolated fractures.
        """
        logger.info("Checking connectivity of the DFN...")
        # connectivity, remove_fracs = gf.check_connectivity(self.fractures)
        self.consolidate_dfn(hpc=True)
        connectivity, rf = gf.check_connectivity_hpc(
            self.fractures_struc_array_hpc, self.elements_struc_array_hpc
        )
        logger.info(
            f"Connectivity check complete. DFN is {'connected' if connectivity else 'not connected'}."
        )
        remove_fracs = [self.fractures[i] for i in rf]
        if not connectivity:
            logger.info(
                f"Removing {len(remove_fracs)} isolated fractures from the DFN."
            )
            self.delete_fracture(remove_fracs)

        # Update the elements in the DFN
        self.get_elements()

    ####################################################################################################################
    #                      Solve functions                                                                             #
    ####################################################################################################################

    def build_discharge_matrix(self):
        """
        Builds the discharge matrix for the DFN and adds it to the DFN.

        """
        self.get_discharge_elements()
        size = len(self.discharge_elements) + self.number_of_fractures

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

    def solve(self, unconsolidate=False):
        """
        Solves the DFN on a HPC.

        To change the solver constants, use the set_kwargs method.

        Parameters
        ----------
        unconsolidate : bool
            If True, the DFN is unconsolidated after the solve. Default is False.
        """
        logger.info("\n")
        logger.info("---------------------------------------")
        logger.info("Starting HPC solve...")
        logger.info("---------------------------------------")
        logger.info("Collecting elements and fractures...")
        t0 = time.time()
        if self.elements is None:
            self.get_elements()
        t1 = time.time()
        logger.debug(f"Time to collect elements and fractures: {t1 - t0:.2f} seconds")
        logger.info("Consolidating DFN...")
        self.consolidate_dfn(hpc=True)
        t2 = time.time()
        logger.debug(f"Time to consolidate DFN: {t2 - t1:.2f} seconds")
        # logger.info("Building discharge matrix...")
        # self.build_discharge_matrix()
        # t3 = time.time()
        # logger.debug(f"Time to build discharge matrix: {t3 - t2:.2f} seconds")
        logger.info("DFN properties:")
        logger.info(f"Number of fractures: {len(self.fractures)}")
        logger.info(f" Number of elements: {len(self.elements)}")
        self.print_solver_constants()
        self.elements_struc_array = hpc_solve(
            self.fractures_struc_array_hpc,
            self.elements_struc_array_hpc,
            self.discharge_int,
            self.constants,
            self.ntype_element,
        )
        if unconsolidate:
            logger.info("Unconsolidating DFN...")
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

    def plot_input(
        self, pl=None, line_width=3.0, point_size=5.0, opacity_fractures=0.2
    ):
        """
        Plots the input of the DFN, i.e. the fractures and elements.

        Parameters
        ----------
        pl : pyvista.Plotter, optional
            The plotter object to use. If None, a new plotter is created and shown.
        line_width : float
            The line width of the elements in the plot. Default is 3.0.
        point_size : float
            The point size of the elements in the plot. Default is 5.0.
        opacity_fractures : float
            The opacity of the fractures in the plot. Default is 0.2.

        Returns
        -------
        None
            The function plots the input of the DFN.
        """
        show = False
        if pl is None:
            pl = self.initiate_plotter()
            show = True
        self.plot_fractures(pl, opacity=opacity_fractures, line_width=line_width)
        labels = {}
        for s in self.structures:
            s.plot(pl)
            labels[f" {s.__class__.__name__}"] = STRUCTURES_COLOR[s._structure_type]
        if self.elements is not None:
            for e in self.elements:
                if e._type == 1:  # Bounding circle
                    continue
                e.plot(pl, line_width=line_width, color=None, point_size=point_size)
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
                    center=f.center,
                    radius=f.radius,
                    normal=f.normal,
                    n_sides=num_side,
                    fill=filled,
                ),
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                line_width=line_width,
            )
            if print_prog:
                logger.debug(f"Plotting fractures: {i + 1} / {len(self.fractures)}")

    def plot_fractures_flow_net(
        self,
        pl,
        lvs=20,
        n_points=2000,
        n_boundary_points=50,
        line_width=2,
        opacity=1.0,
        fill=True,
        contour_re=True,
        contour_im=True,
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

    def plot_fractures_head_org(
        self,
        pl,
        lvs=20,
        n_layers=10,
        line_width=2,
        opacity=1.0,
        color_map="viridis",
        limits=None,
        contour=True,
        colorbar=True,
        debug=False,
        fractures=None,
    ):
        """
        Plots the flow net for the fractures in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        lvs : int | bool
            The number of levels to contour for the flow net.
        n_layers : int
            The number of layers to use for the flow net. This determines how many points are used
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
        debug : bool
            This will only plot fractures with a head outside the limits.
        fractures : list
            The list of fractures to plot. If None, all fractures are plotted.
        """

        # Start timer
        start = time.time()

        # Asset debug and limits
        if debug:
            assert limits is not None, "For debug mode, limits must be provided."
            min_lim, max_lim = limits
            limits = None

        # Check if the fractures have been consolidated
        if self.fractures_struc_array_hpc is None:
            self.consolidate_dfn(hpc=True)

        # Only compute fracture that are given in fracs
        if fractures is not None:
            fracture_dtype_hpc = self.fractures_struc_array_hpc.dtype
            self.fractures_struc_array_hpc_fracs = np.empty(
                len(fractures), dtype=fracture_dtype_hpc
            )
            for i, f in enumerate(fractures):
                idx = self.fractures.index(f)
                self.fractures_struc_array_hpc_fracs[i] = (
                    self.fractures_struc_array_hpc[idx]
                )
        else:
            self.fractures_struc_array_hpc_fracs = self.fractures_struc_array_hpc

        # Calculate the hydraulic head for each fracture and get the mapped 3d points
        h = 1 / (n_layers + 1)
        partitions = int(2 * np.pi / h / n_layers)
        z_array, base_faces = generate_disk(partitions, n_layers)
        heads, pnts_3d = hpc_get_heads(
            self.fractures_struc_array_hpc_fracs,
            self.elements_struc_array_hpc,
            z_array,
        )

        # Get the limits for the color map
        if limits is None:
            limits = [np.nanmin(heads), np.nanmax(heads)]
            logger.debug(f"Calculated limits for the color map: {limits}")

        # Calculate the levels for the contours
        if lvs is not False:
            lvs = np.linspace(limits[0], limits[1], lvs)

        # Create the PyVista meshes and plot for all fractures
        # Get the faces for the Delaunay triangulation (since we use the same triangulation for all fractures)
        s0 = time.time()
        faces = get_faces(pnts_3d[0])
        s1 = time.time()
        print(f"Creating faces took {s1 - s0:.2f} seconds.")

        meshes = []
        if debug:
            logger.debug(
                "Debug mode: Only plotting fractures with head outside limits."
            )
            for i, pnts in enumerate(pnts_3d):
                if np.nanmin(heads[i]) < min_lim or np.nanmax(heads[i]) > max_lim:
                    logger.debug(f"Fracture id={i} has head outside limits, plotting.")
                    poly = pv.PolyData(pnts, faces)
                    poly.point_data["head"] = heads[i]
                    meshes.append(poly)
        else:
            for i, pnts in enumerate(pnts_3d):
                poly = pv.PolyData(pnts, faces)
                poly.point_data["head"] = heads[i]
                meshes.append(poly)
        mesh = pv.merge(meshes)
        pl.add_mesh(
            mesh,
            scalars="head",
            cmap=color_map,
            opacity=opacity,
            show_edges=False,
            line_width=line_width,
            scalar_bar_args=dict(title="Hydraulic Head", shadow=True),
            name=f"head_{i}",
            clim=limits,
        )
        # Add contour lines, i.e. equipotential lines
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

    def plot_fractures_head(
        self,
        pl,
        lvs=20,
        n_layers=10,
        line_width=2,
        opacity=1.0,
        color_map="viridis",
        limits=None,
        contour=True,
        colorbar=True,
        debug=False,
        fractures=None,
    ):

        start = time.time()

        # --- Debug handling ---
        if debug:
            assert limits is not None, "For debug mode, limits must be provided."
            min_lim, max_lim = limits
            limits = None

        # --- Ensure consolidation ---
        if self.fractures_struc_array_hpc is None:
            self.consolidate_dfn(hpc=True)

        # --- Select fractures efficiently ---
        fracs_arr = self.fractures_struc_array_hpc
        if fractures is not None:
            fracture_index = {f: i for i, f in enumerate(self.fractures)}
            idx = np.fromiter(
                (fracture_index[f] for f in fractures),
                dtype=np.int64,
                count=len(fractures),
            )
            fracs_arr = fracs_arr[idx]

        # --- Compute heads & points ---
        h = 1 / (n_layers + 1)
        partitions = int(2 * np.pi / h / n_layers)
        z_array, base_faces = generate_disk(partitions, n_layers)
        heads, pnts_3d = hpc_get_heads(
            fracs_arr, self.elements_struc_array_hpc, z_array
        )

        # --- Debug filtering BEFORE mesh creation ---
        if debug:
            mask = np.array(
                [(np.nanmin(h) < min_lim) or (np.nanmax(h) > max_lim) for h in heads],
                dtype=bool,
            )
            heads = heads[mask]
            pnts_3d = pnts_3d[mask]

        if heads.size == 0:
            return

        # --- Color limits ---
        if limits is None:
            limits = [np.nanmin(heads), np.nanmax(heads)]

        # --- Contour levels ---
        if lvs is not False:
            lvs = np.linspace(limits[0], limits[1], lvs)

        # --- Build ONE mesh (major speedup) ---

        nf, npts, _ = pnts_3d.shape

        points = pnts_3d.reshape(nf * npts, 3)
        head_vals = heads.reshape(nf * npts)

        # base_faces = get_faces(pnts_3d[0])  # already VTK-style

        faces = np.tile(base_faces, nf)

        # indices are at positions 1,2,3, 5,6,7, 9,10,11, ...
        idx = np.arange(len(faces)) % 4 != 0

        repeat_offsets = np.repeat(np.arange(nf) * npts, len(base_faces))
        faces[idx] += repeat_offsets[idx]

        mesh = pv.PolyData(points, faces)
        mesh.point_data["head"] = head_vals

        # --- Plot mesh ---
        pl.add_mesh(
            mesh,
            scalars="head",
            cmap=color_map,
            opacity=opacity,
            show_edges=False,
            line_width=line_width,
            scalar_bar_args=dict(title="Hydraulic Head", shadow=True),
            clim=limits,
            name="head",
        )

        # --- Contours ---
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

        if not colorbar:
            pl.remove_scalar_bar()

        logger.info(f"Plotting hydraulic head took {time.time() - start:.2f} seconds.")

    def plot_elements(
        self, pl, color=None, elements=None, line_width=3.0, const_elements=False
    ):
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
        const_elements : bool
            Whether to only plot the constant head elements. Default is False.
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

    def plot_sparse_matrix(
        self, save=False, filename="sparse_matrix.png", black_bg=True
    ):
        """
        Plots the sparse matrix of the DFN.

        Parameters
        ----------
        save : bool
            Whether to save the plot.
        filename : str
            The name of the plot.
        black_bg : bool
            Whether to use a black background.

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
        if black_bg:
            fig.set_facecolor("black")
            ax.set_facecolor("black")
            tick_color = "white"
            title_color = "white"
            spine_color = "white"
            spy_color = "white"
        else:
            fig.set_facecolor("white")
            ax.set_facecolor("white")
            tick_color = "black"
            title_color = "black"
            spine_color = "black"
            spy_color = "black"
        # set tick and title color and axis box
        ax.tick_params(axis="x", colors=tick_color)
        ax.tick_params(axis="y", colors=tick_color)
        ax.title.set_color(title_color)
        for spine in ["top", "left", "right", "bottom"]:
            ax.spines[spine].set_color(spine_color)
        ax.spy(self.discharge_matrix, markersize=0.5, color=spy_color)
        # Equal axis
        ax.set_aspect("equal")

        num_entries = self.discharge_matrix.nnz
        num_zeros = self.discharge_matrix.shape[0] * self.discharge_matrix.shape[1]
        # title
        ax.set_title(
            f"Sparse matrix of the DFN\nNumber of fractures: {self.number_of_fractures}, Number of elements: {self.number_of_elements()}"
            f"\nNumber of entries: {num_entries}, Number of zeros: "
            f"{num_zeros - num_entries}\nFilled percentage: {num_entries / num_zeros * 100:.2f}%"
        )
        if save:
            plt.savefig(filename)
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

        self.unconsolidate_dfn(hpc=True)

        ncoef = []
        for e in self.elements:
            ncoef.append(e.ncoef)
        fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
        nbins = np.ceil((max(ncoef) - min(ncoef)) / 5).astype(int)
        if nbins < 1:
            nbins = 1
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
                print(f"\rTracing streamline: {i} / {len(z0)}", end="")
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
            ds_frac = frac.radius * ds
            # get the current number of points
            length = sum([len(s) for s in streamline])
            # Start the tracking process
            psi = [z_start]
            w = [frac.calc_velocity(z0)]
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
                w.append(frac.calc_velocity(z1))
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
            w.append(frac.calc_velocity(z3))

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
                #   continue
                z2 = e.check_chi_crossing(z0, z1, frac)
            else:
                # if (e.q < 0) ^ backward:
                #   continue
                z2 = e.check_chi_crossing(z0, z1)
            if z2 is not False:
                return z2, e
        return False, False

    @staticmethod
    def get_exit_intersection(z3d, element, frac, frac_old, elevation, dchi=1e-4):
        if frac == element.frac0:
            endpoints = element.endpoints0
        else:
            endpoints = element.endpoints1
        z = gf.map_3d_to_2d(z3d, frac)
        # z2 = gf.map_3d_to_2d(z3d, frac_old)
        chi0 = gf.map_z_line_to_chi(z, endpoints)
        chi1 = np.conj(chi0)
        # chi20 = gf.map_z_line_to_chi(z2, endpoints)
        # chi21 = np.conj(chi20)
        z0 = gf.map_chi_to_z_line(chi0 * (1 + dchi), endpoints)
        z1 = gf.map_chi_to_z_line(chi1 * (1 + dchi), endpoints)
        # z2 = gf.map_chi_to_z_line(chi20 * (1 + dchi), endpoints)
        # z3 = gf.map_chi_to_z_line(chi21 * (1 + dchi), endpoints)
        w0 = frac.calc_w(z0)
        w1 = frac.calc_w(z1)
        # w2 = frac_old.calc_w(z2)
        # w3 = frac_old.calc_w(z3)

        # Magnitude
        abs_w0 = np.abs(w0)
        abs_w1 = np.abs(w1)
        # abs_w2 = np.abs(w2)
        # abs_w3 = np.abs(w3)

        # check angles between w0, w1 and z-z0, z-z1, using the dot product
        divide = abs_w0 / (abs_w0 + abs_w1)
        pointz0 = gf.map_2d_to_3d(z0, frac)

        # if divide > 0.5 + np.random.rand() * 0.1:
        #  return z0, z0, elevation * 0 + 0.5
        # else:
        #   return z1, z1, elevation * 0 + 0.5

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
            return down, z0, elevation  # * 0 + 0.5

        elevation = (elevation - divide) / (1 - divide)
        return up, z0, elevation  # * 0 + 0.5

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
        if np.abs(z1) > frac.radius:
            z1 *= frac.radius / (np.abs(z1) * (1 + 1e-5))
        dz = 1e99
        it = 0
        while dz > tolerance and it < max_it:
            w1 = frac.calc_w(z1)
            if np.isnan(np.real(w1)):
                break
            z2 = z0 + np.conj(w0 + w1) / np.abs(w0 + w1) * ds
            if np.abs(z2) > frac.radius:
                z2 *= frac.radius / (np.abs(z2) * (1 + 1e-5))
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
            time = 0
            length = 0
            for i, s in enumerate(streamline):
                if isinstance(s, list):
                    streamline[i] = np.array(s)
                if isinstance(velocity[i], list):
                    velocity[i] = np.array(velocity[i])
                t, le = _get_length_time_fracture(streamline[i], velocity[i])
                time += t
                length += le
            return time, length
        if isinstance(streamline, list):
            streamline = np.array(streamline)
        if isinstance(velocity, list):
            velocity = np.array(velocity)
        # remove nans
        mask = ~np.isnan(streamline) & ~np.isnan(velocity)
        streamline = streamline[mask]
        velocity = velocity[mask]
        t, le = _get_length_time_fracture(streamline, velocity)

        return t, le


def _get_length_time_fracture(streamline, velocity):
    mask = ~np.isnan(streamline) & ~np.isnan(velocity)
    streamline = streamline[mask]
    velocity = velocity[mask]
    time = np.sum(np.abs(streamline[1:] - streamline[:-1]) / velocity[:-1])
    length = np.sum(np.abs(streamline[1:] - streamline[:-1]))

    return time, length


def generate_disk(partitions: int, depth: int):
    """
    Generate a triangular mesh for the unit circle.

    Parameters
    ----------
    partitions: int
        Number of triangles around the origin.
    depth: int
        Number of "layers" of triangles around the origin.

    Returns
    -------
    z: np.ndarray of complex numbers with shape ``(n_points,)``
        The coordinates of the points in the mesh, where the real part is the x-coordinate and the imaginary part is the y-coordinate.
    faces : np.ndarray of integers with shape ``(n_triangles, 4)``
        The faces of the mesh, where each row contains the number of vertices in the face (which is 3 for triangles) followed by the indices of the vertices in the mesh.
    """
    N = depth + 1
    n_per_level = partitions * np.arange(N)
    n_per_level[0] = 1

    delta_angle = (2 * np.pi) / np.repeat(n_per_level, n_per_level)
    index = np.repeat(np.insert(n_per_level.cumsum()[:-1], 0, 0), n_per_level)
    angles = delta_angle.cumsum()
    angles = angles - angles[index] + 0.5 * np.pi
    radii = np.repeat(np.linspace(0.0, 1.0, N), n_per_level)

    x = np.cos(angles) * radii
    y = np.sin(angles) * radii
    triang = Triangulation(x, y)
    triangles = triang.triangles
    faces = np.column_stack([np.full(triangles.shape[0], 3), triangles]).ravel()
    return x + 1j * y, faces
