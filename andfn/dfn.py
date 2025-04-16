"""
Notes
-----
This module contains the DFN class.

The DFN class is the main class for the AnDFN package. It contains the fractures and elements of the DFN and methods to
solve the DFN.
"""

from datetime import datetime

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import scipy as sp
import h5py

from andfn import geometry_functions as gf
from .constants import Constants, dtype_constants
from .fracture import Fracture
from .hpc.hpc_solve import solve as hpc_solve
from .hpc.hpc_fracture import get_flow_nets as hpc_get_flow_nets, get_heads as hpc_get_heads
from .well import Well
from .impermeable_object import ImpermeableCircle, ImpermeableLine
from .const_head import ConstantHeadLine
from .intersection import Intersection
from .bounding import BoundingCircle
from .element import element_dtype, fracture_dtype, element_index_dtype, fracture_index_dtype, element_dtype_hpc, \
    fracture_dtype_hpc, MAX_NCOEF, MAX_ELEMENTS


def generate_connected_fractures(num_fracs, radius_factor, center_factor, ncoef_i, nint_i, ncoef_b, nint_b, frac_surface=None):
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

    Returns
    -------
    frac_list : list
        The list of fractures.

    """
    print('Generating fractures...')
    fracs = gf.generate_fractures(num_fracs, radius_factor=radius_factor, center_factor=center_factor, ncoef=ncoef_b,
                                  nint=nint_b)

    print('Analyzing intersections...')
    frac_list = gf.get_connected_fractures(fracs, ncoef_i, nint_i, frac_surface)

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
    omega_max_re, omega_min_re = np.nanmax(np.real(omega_fn_list)), np.nanmin(np.real(omega_fn_list))
    omega_max_im, omega_min_im = np.nanmax(np.imag(omega_fn_list)), np.nanmin(np.imag(omega_fn_list))
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
    if seg.dtype is not np.dtype('complex'):
        x = seg[:, 0]
        y = seg[:, 1]
        contour_complex = x + 1j * y
    else:
        contour_complex = seg
    line_3d = gf.map_2d_to_3d(contour_complex, f)
    pl.add_mesh(pv.MultipleLines(line_3d), color=color, line_width=line_width)


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
        return f'DFN: {self.label}'

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
                self.constants[key] = value
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
        if filename[-3:] == '.h5':
            filename = filename[:-3]

        # Check if the elements and fractures have been consolidated
        if self.elements_struc_array is None or self.fractures_struc_array is None:
            self.consolidate_dfn()

        # Save the elements
        with h5py.File(f'{filename}.h5', 'w') as hf:
            grp = [hf.create_group('elements/properties'), hf.create_group('fractures/properties'),
                   hf.create_group('elements/index'), hf.create_group('fractures/index')]
            for j, array in enumerate([self.elements_struc_array, self.fractures_struc_array]):
                for name in array.dtype.names:
                    # create group
                    grp0 = grp[j].create_group(name)
                    # add data
                    for i, e in enumerate(array[name]):
                        grp0.create_dataset(f'{i}', data=e)

            for j, array in enumerate([self.elements_index_array, self.fractures_index_array], start=2):
                for name in array.dtype.names:
                    # check if the data is a string
                    if self.elements_index_array[name].dtype == 'U100':
                        grp[j].create_dataset(name, data=array[name].astype('S'))
                        continue
                    grp[j].create_dataset(name, data=array[name])


        print(f'Saved DFN to {filename}.h5')

    def load_dfn(self, filename):
        """
        Loads the DFN from a h5 file.

        Parameters
        ----------
        filename : str
            The name of the file to load the DFN from.
        """

        if filename[-3:] == '.h5':
            filename = filename[:-3]

        with h5py.File(f'{filename}.h5', 'r') as hf:
            # Load the fractures
            fracs = []
            for i in range(len(hf['fractures/index/label'])):
                fracs.append(
                    Fracture(
                    label=hf['fractures/index/label'][i].decode(),
                    id_=hf['fractures/index/id_'][i],
                    t=hf[f'fractures/properties/t/{i}'][()],
                    radius=hf[f'fractures/properties/radius/{i}'][()],
                    center=hf[f'fractures/properties/center/{i}'][()],
                    normal=hf[f'fractures/properties/normal/{i}'][()],
                    x_vector=hf[f'fractures/properties/x_vector/{i}'][()],
                    y_vector=hf[f'fractures/properties/y_vector/{i}'][()],
                    elements=False,
                    constant=hf[f'fractures/properties/constant/{i}'][()]
                ))

            # Load the elements
            elements = []
            for i in range(len(hf['elements/index/label'])):
                if hf['elements/index/type_'][i] == 0:  # Intersection
                    elements.append(
                        Intersection(
                            label=hf['elements/index/label'][i].decode(),
                            id_=hf['elements/index/id_'][i],
                            endpoints0=hf[f'elements/properties/endpoints0/{i}'][()],
                            endpoints1=hf[f'elements/properties/endpoints1/{i}'][()],
                            ncoef=hf[f'elements/properties/ncoef/{i}'][()],
                            nint=hf[f'elements/properties/nint/{i}'][()],
                            frac0=fracs[hf[f'elements/properties/frac0/{i}'][()]],
                            frac1=fracs[hf[f'elements/properties/frac1/{i}'][()]],
                            thetas=hf[f'elements/properties/thetas/{i}'][()],
                            coef=hf[f'elements/properties/coef/{i}'][()],
                            error=hf[f'elements/properties/error/{i}'][()]
                        )
                    )
                elif hf['elements/index/type_'][i] == 1:  # Bounding circle
                    elements.append(
                        BoundingCircle(
                            label=hf['elements/index/label'][i].decode(),
                            id_=hf['elements/index/id_'][i],
                            radius=hf[f'elements/properties/radius/{i}'][()],
                            center=hf[f'elements/properties/center/{i}'][()],
                            frac0=fracs[hf[f'elements/properties/frac0/{i}'][()]],
                            thetas=hf[f'elements/properties/thetas/{i}'][()],
                            coef=hf[f'elements/properties/coef/{i}'][()],
                            ncoef=hf[f'elements/properties/ncoef/{i}'][()],
                            nint=hf[f'elements/properties/nint/{i}'][()],
                            dpsi_corr=hf[f'elements/properties/dpsi_corr/{i}'][()],
                            error=hf[f'elements/properties/error/{i}'][()]
                        )
                    )
                elif hf['elements/index/type_'][i] == 2:  # Well
                    elements.append(
                        Well(
                            label=hf['elements/index/label'][i].decode(),
                            id_=hf['elements/index/id_'][i],
                            radius=hf[f'elements/properties/radius/{i}'][()],
                            center=hf[f'elements/properties/center/{i}'][()],
                            head=hf[f'elements/properties/head/{i}'][()],
                            frac0=fracs[hf[f'elements/properties/frac0/{i}'][()]],
                            q=hf[f'elements/properties/q/{i}'][()],
                            phi=hf[f'elements/properties/phi/{i}'][()],
                            error=hf[f'elements/properties/error/{i}'][()]
                        )
                    )
                elif hf['elements/index/type_'][i] == 3:  # Constant head line
                    elements.append(
                        ConstantHeadLine(
                            label=hf['elements/index/label'][i].decode(),
                            id_=hf['elements/index/id_'][i],
                            head=hf[f'elements/properties/head/{i}'][()],
                            endpoints0=hf[f'elements/properties/endpoints0/{i}'][()],
                            ncoef=hf[f'elements/properties/ncoef/{i}'][()],
                            nint=hf[f'elements/properties/nint/{i}'][()],
                            frac0=fracs[hf[f'elements/properties/frac0/{i}'][()]],
                            phi=hf[f'elements/properties/phi/{i}'][()],
                            thetas=hf[f'elements/properties/thetas/{i}'][()],
                            coef=hf[f'elements/properties/coef/{i}'][()],
                            error=hf[f'elements/properties/error/{i}'][()]
                        )
                    )
                elif hf['elements/index/type_'][i] == 4:  # Impermeable circle
                    elements.append(
                        ImpermeableCircle(
                            label=hf['elements/index/label'][i].decode(),
                            id_=hf['elements/index/id_'][i],
                            radius=hf[f'elements/properties/radius/{i}'][()],
                            center=hf[f'elements/properties/center/{i}'][()],
                            frac0=fracs[hf[f'elements/properties/frac0/{i}'][()]],
                            ncoef=hf[f'elements/properties/ncoef/{i}'][()],
                            nint=hf[f'elements/properties/nint/{i}'][()],
                            thetas=hf[f'elements/properties/thetas/{i}'][()],
                            coef=hf[f'elements/properties/coef/{i}'][()],
                            error=hf[f'elements/properties/error/{i}'][()]
                        )
                    )
                elif hf['elements/index/type_'][i] == 5:  # Impermeable line
                    elements.append(
                        ImpermeableLine(
                            label=hf['elements/index/label'][i].decode(),
                            id_=hf['elements/index/id_'][i],
                            endpoints0=hf[f'elements/properties/focis/{i}'][()],
                            frac0=fracs[hf[f'elements/properties/frac0/{i}'][()]],
                            ncoef=hf[f'elements/properties/ncoef/{i}'][()],
                            nint=hf[f'elements/properties/nint/{i}'][()],
                            thetas=hf[f'elements/properties/thetas/{i}'][()],
                            coef=hf[f'elements/properties/coef/{i}'][()],
                            dpsi_corr=hf[f'elements/properties/dpsi_corr/{i}'][()],
                            error=hf[f'elements/properties/error/{i}'][()]
                        )
                    )

            # Add the fractures and elements to the DFN
            for e in elements:
                e.frac0.add_element(e)
                if isinstance(e, Intersection):
                    e.frac1.add_element(e)
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
        elements_index_array = np.empty(self.number_of_elements(), dtype=element_index_dtype)

        if hpc:
            for i, e in enumerate(self.elements):
                elements_struc_array[i], elements_index_array[i] = e.consolidate_hpc()
        else:
            for i, e in enumerate(self.elements):
                elements_struc_array[i], elements_index_array[i] = e.consolidate()

        # Consolidate fractures
        fractures_struc_array = np.empty(self.number_of_fractures(), dtype=f_dtype)
        fractures_index_array = np.empty(self.number_of_fractures(), dtype=fracture_index_dtype)

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
                f.unconsolidate_hpc(self.fractures_struc_array_hpc[i], self.fractures_index_array[i])
            for i, e in enumerate(self.elements):
                e.unconsolidate_hpc(self.elements_struc_array_hpc[i], self.elements_index_array[i], self.fractures)
        else:
            for i, f in enumerate(self.fractures):
                f.unconsolidate(self.fractures_struc_array[i], self.fractures_index_array[i])
            for i, e in enumerate(self.elements):
                e.unconsolidate(self.elements_struc_array[i], self.elements_index_array[i], self.fractures)



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
        print(f'Added {len(self.elements)} elements to the DFN.')

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
        self.discharge_elements = [e for e in self.elements
                                   if isinstance(e, Intersection)
                                   or isinstance(e, ConstantHeadLine)
                                   or isinstance(e, Well)]

        self.discharge_elements_index = [e.id_ for e in self.discharge_elements]

    def get_dfn_discharge(self):
        # sum all discharges, except the intersections
        if self.discharge_elements is None:
            self.get_discharge_elements()
        q = 0
        for e in self.discharge_elements:
            if isinstance(e, Intersection):
                continue
            q += np.abs(e.q)
        return q/2

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
                print(f'Added {new_fracture[0]} fracture to the DFN.')
            else:
                self.fractures.extend(new_fracture)
                print(f'Added {len(new_fracture)} fractures to the DFN.')
        else:
            self.fractures.append(new_fracture)
            print(f'Added {new_fracture} fracture to the DFN.')
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

    def generate_connected_dfn(self, num_fracs, radius_factor, center_factor, ncoef_i, nint_i, ncoef_b, nint_b, frac_surface=None):
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
        frac_list = generate_connected_fractures(num_fracs, radius_factor, center_factor, ncoef_i, nint_i, ncoef_b,
                                                 nint_b, frac_surface)
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
                    i0 = Intersection(f'{fr.label}_{fr2.label}', endpoints0, endpoints1, fr, fr2, ncoef, nint)
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
                        i0 = Intersection(f'{fr.label}_{fr2.label}', endpoints0, endpoints1, fr, fr2, ncoef, nint)
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

    def set_constant_head_boundary(self, center, normal, radius, head, label='Constant Head Boundary', ncoef=5, nint=10):
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
        gf.set_head_boundary(self.fractures, ncoef, nint, head, center, radius, normal, label)

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
                        data.append(e.frac0.head_from_phi(ee.discharge_term(z0, e.frac0)))
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
                        data.append(e.frac1.head_from_phi(-ee.discharge_term(z1, e.frac1)))
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
        self.get_elements()
        self.consolidate_dfn(hpc=True)
        self.build_discharge_matrix()
        print('\n')
        print('---------------------------------------')
        print('Starting HPC solve...')
        print('---------------------------------------')
        print(f'Number of elements: {len(self.elements)} \nNumber of fractures: {len(self.fractures)} \nNumber of entries in discharge matrix: {self.discharge_matrix.getnnz()}')
        self.print_solver_constants()
        self.elements_struc_array = hpc_solve(self.fractures_struc_array_hpc, self.elements_struc_array_hpc,
                                                    self.discharge_matrix, self.discharge_int, self.constants)
        self.unconsolidate_dfn(hpc=True)

    ####################################################################################################################
    #                    Plotting functions                                                                            #
    ####################################################################################################################

    def initiate_plotter(self, window_size=(800, 800), grid=False, lighting='light kit', title=True, off_screen=False,
                         scale=1.0, axis=True, notebook=False):
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
        print('---------------------------------------')
        print('Starting plotter...')
        print('---------------------------------------')
        pl = pv.Plotter(window_size=window_size, lighting=lighting, off_screen=off_screen, notebook=notebook)
        if axis:
            _ = pl.add_axes(
                line_width=2*scale,
                cone_radius=0.3+0.1*(1-1/scale),
                shaft_length=0.7+0.3*(1-1/scale),
                tip_length=0.3+0.1*(1-1/scale),
                ambient=0.5,
                label_size=(0.2/scale, 0.08/scale),
                xlabel='X (E)',
                ylabel='Y (N)',
                zlabel='Z')
        if grid:
            _ = pl.show_grid()
        #_ = pl.add_bounding_box(line_width=5, color='black', outline=False, culling='back')
        if isinstance(title, str):
            _ = pl.add_text(title, font_size=10*scale, position='upper_left', color='k', shadow=True)
            return pl
        if title:
            _ = pl.add_text(f'DFN: {self.label}', font_size=10*scale, position='upper_left', color='k', shadow=True)
        return pl

    def get_flow_fractures(self, cond=2e-1):
        q_dfn = self.get_dfn_discharge()
        fracs = []
        for i, f in enumerate(self.fractures):
            if f.get_total_discharge() / q_dfn > cond:
                fracs.append(f)
        return fracs

    def plot_fractures(self, pl, num_side=50, filled=True, color='#FFFFFF', opacity=1.0, show_edges=True,
                       line_width=2.0, fracs=None):
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
            pl.add_mesh(pv.Polygon(f.center, f.radius, normal=f.normal, n_sides=num_side, fill=filled),
                        color=color, opacity=opacity, show_edges=show_edges, line_width=line_width)
            if print_prog:
                print(f'\rPlotting fractures: {i + 1} / {len(self.fractures)}', end='')

    def plot_fractures_flow_net(self, pl, lvs, n_points, line_width=2, margin=0.01, only_flow=False):
        """
        Plots the flow net for the fractures in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        lvs : int
            The number of levels to contour for the flow net.
        n_points : int
            The number of points to use for the flow net (n_points x n_points).
        line_width : float
            The line width of the flow net.
        margin : float
            The margin around the fracture to use for the flow net.
        only_flow : bool
            Whether to plot only the fractures with flow.
        """
        if self.fractures_struc_array_hpc is None:
            self.consolidate_dfn(hpc=True)
        if only_flow:
            fracs = self.get_flow_fractures()
            self.plot_fractures(pl, fracs=fracs)
        else:
            fracs = self.fractures

        # Calculate the flow net for each fracture
        omegas, x_arrays, y_arrays = hpc_get_flow_nets(self.fractures_struc_array_hpc, n_points, margin, self.elements_struc_array_hpc)


        # Get the levels for the flow net
        lvs_re, lvs_im = get_lvs(lvs, omegas)

        # Plot the flow net for each fracture
        for i, f in enumerate(fracs):
            print(f'\rPlotting flow net: {i + 1} / {len(fracs)}', end='')
            # plot the flow net using matplotlib
            contours_re = plt.contour(x_arrays[i], y_arrays[i], np.real(omegas[i]), levels=lvs_re)
            contours_im = plt.contour(x_arrays[i], y_arrays[i], np.imag(omegas[i]), levels=lvs_im)
            # Extract the contour line and plot them in 3D, real and imaginary parts
            for contour in contours_re.allsegs:
                for seg in contour:
                    if len(seg) == 0:
                        continue
                    plot_line_3d(seg, f, pl, 'red', line_width=line_width)
            for contour in contours_im.allsegs:
                for seg in contour:
                    if len(seg) == 0:
                        continue
                    plot_line_3d(seg, f, pl, 'blue', line_width=line_width)
        print('')

    def plot_fractures_head(self, pl, lvs=20, n_points=100, line_width=2, margin=1e-3, opacity=1.0, only_flow=False,
                            color_map='jet', limits=None, contour=True):
        """
        Plots the flow net for the fractures in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        lvs : int
            The number of levels to contour for the flow net.
        n_points : int
            The number of points to use for the flow net (n_points x n_points).
        line_width : float
            The line width of the flow net.
        margin : float
            The margin around the fracture to use for the flow net.
        opacity : float
            The opacity of the fractures in the flownet.
        only_flow : bool
            Whether to plot only the fractures with flow.
        color_map : str
            The color map to use for the flow net.
        limits : list | tuple
            Custom limits for the flow net, overwrites the calculated limits.
        """
        if self.fractures_struc_array_hpc is None:
            self.consolidate_dfn(hpc=True)
        if only_flow:
            fracs = self.get_flow_fractures()
        else:
            fracs = self.fractures

        # Calculate the flow net for each fracture
        max_min_head_list = []
        heads, x_arrays, y_arrays = hpc_get_heads(self.fractures_struc_array_hpc, n_points, margin, self.elements_struc_array_hpc)
        for i, f in enumerate(fracs):
            max_min_head = f.get_max_min_head()
            if max_min_head[0] is not None:
                max_min_head_list.append(max_min_head)

        # Get the levels for the flow net
        head_max, head_min = np.nanmax(heads), np.nanmin(heads)
        if head_max < np.max(max_min_head_list):
            head_max = np.max(max_min_head_list)
        if head_min > np.min(max_min_head_list):
            head_min = np.min(max_min_head_list)
        if limits is not None:
            head_min, head_max = limits
        # Create the levels for the equipotential contours
        lvs_re = np.linspace(head_min, head_max, lvs)

        cmap = plt.colormaps[color_map]
        colors = cmap(np.linspace(0, 1, lvs))

        # Plot the flow net for each fracture
        for i, f in enumerate(fracs):
            # Plot the fractures
            mean_head = np.nanmean(heads[i])
            pos_frac = np.where(np.abs(lvs_re - mean_head) == np.min(np.abs(lvs_re - mean_head)))[0][0]
            color_frac = colors[pos_frac]
            self.plot_fractures(pl, filled=True, color=color_frac, opacity=opacity, show_edges=True, line_width=2.0,
                                fracs=[f])

            if contour:
                # plot the contour lines using matplotlib
                contours_re = plt.contour(x_arrays[i], y_arrays[i], heads[i], levels=lvs_re)

                # Extract the contour line and plot them in 3D, real and imaginary parts
                for contour in contours_re.allsegs:
                    for seg in contour:
                        if len(seg) == 0:
                            continue
                        loc = int(len(seg)/4)
                        oms = f.calc_omega(np.array([seg[loc][0] + seg[loc][1]*1j, seg[loc*2][0] + seg[loc*2][1]*1j, seg[loc*3][0] + seg[loc*3][1]*1j]))
                        head = f.head_from_phi(np.real(np.nanmean(oms)))
                        pos = np.where(np.abs(lvs_re-head) == np.min(np.abs(lvs_re-head)))[0][0]
                        color = colors[pos]
                        plot_line_3d(seg, f, pl, color, line_width=line_width)
            print(f'\rPlotting hydraulic head: {i + 1} / {len(fracs)}', end='')

        # Add the color bar
        # Create a sample mesh
        mesh = pv.Sphere(radius=0.001, center=self.get_dfn_center())
        # Create a scalar array ranging from 10 to 20
        scalars = np.linspace(np.floor(head_min), np.ceil(head_max), mesh.n_points)
        # Add the scalar array to the mesh
        mesh.point_data['Hydraulic head'] = scalars
        # Add the mesh to the plotter
        _ = pl.add_mesh(mesh, opacity=0.0, show_scalar_bar=False, cmap=cmap)
        _ = pl.add_scalar_bar(
            'Hydraulic head',
            interactive=True,
            vertical=False,
            fmt='%10.1f',
        )
        plt.close()
        print('')



    def plot_elements(self, pl, elements=None):
        """
        Plots the elements in the DFN.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        """
        # Check if the elements have been stored in the DFN
        assert self.elements is not None and len(
            self.elements) > 0, 'The elements have not been stored in the DFN. Use the get_elements method.'
        # Plot the elements
        if elements is None:
            elements = self.elements
        for i, e in enumerate(elements):
            if isinstance(e, Intersection):
                line = gf.map_2d_to_3d(e.endpoints0, e.frac0)
                pl.add_mesh(pv.Line(line[0], line[1]), color='#000000', line_width=3)
            if isinstance(e, (ConstantHeadLine, ImpermeableLine)):
                line = gf.map_2d_to_3d(e.endpoints0, e.frac0)
                pl.add_mesh(pv.Line(line[0], line[1]), color='#000000', line_width=3)
            if isinstance(e, (Well, ImpermeableCircle)):
                point = gf.map_2d_to_3d(e.center, e.frac0)
                pl.add_mesh(pv.Polygon(point, e.radius, normal=e.frac0.normal, n_sides=50, fill=False),
                            color='#000000', line_width=3)
            print(f'\rPlotting elements: {i + 1} / {len(self.elements)}', end='')
        print('')

    def plot_sparse_matrix(self, save=False, name='sparse_matrix.png'):
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
        # Check if the discharge matrix has been built
        if self.discharge_matrix is None:
            self.build_discharge_matrix()

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        # set background color of plot surface
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        # set tick and title color and axis box
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.title.set_color('white')
        for spine in ['top', 'left', 'right', 'bottom']:
            ax.spines[spine].set_color('white')
        ax.spy( self.discharge_matrix, markersize=0.5, color='white')
        # Equal axis
        ax.set_aspect('equal')

        num_entries = self.discharge_matrix.nnz
        num_zeros = self.discharge_matrix.shape[0] * self.discharge_matrix.shape[1]
        # title
        ax.set_title(f'Sparse matrix of the DFN\nNumber of fractures: {self.number_of_fractures()}, Number of elements: {self.number_of_elements()}'
                     f'\nNumber of entries: {num_entries}, Number of zeros: '
                     f'{num_zeros - num_entries}\nFilled percentage: {num_entries / num_zeros * 100:.2f}%')
        if save:
            plt.savefig(name)
        plt.show()

    def plot_ncoef(self, save=False, name='ncoef.png'):
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
        ncoef = []
        for e in self.elements:
            ncoef.append(e.ncoef)
        fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
        nbins = np.ceil((max(ncoef) - min(ncoef))/5).astype(int)
        counts, edges, bars = ax.hist(ncoef, bins=nbins, color='grey', edgecolor='black')
        ax.set_title('Number of coefficients for the elements in the DFN')
        ax.set_xlabel('Number of coefficients')
        ax.set_ylabel('Number of elements')
        ax.set_yscale('log')
        ax.bar_label(bars)
        if save:
            plt.savefig(name)
        plt.show()

    ####################################################################################################################
    #                    Streamline tracking functions                                                                 #
    ####################################################################################################################
    def plot_streamline_tracking(self, pl, z0, frac, ds=1e-2, max_length=1000, line_width=2.0, elevation=0.0,
                                 remove_false=True, color='black'):
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

        streamlines = []
        streamlines_frac = []
        velocities = []
        elements = []
        for i, z in enumerate(z0): # type: int, complex
            for j, e in enumerate(elevation):
                streamline, streamline_frac, velocity, element = self.streamline_tracking(z, frac, e, ds, max_length)
                streamlines.append(streamline)
                streamlines_frac.append(streamline_frac)
                velocities.append(velocity)
                elements.append(element)
                print(f'\rTracing streamline: {i + 1} / {len(z0)}', end='')

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
            pl.add_mesh(pv.MultipleLines(streamline_3d), color=color, line_width=line_width)

        return streamlines, streamlines_frac, velocities, elements

    def streamline_tracking(self, z0, frac, elevation, ds, max_length):
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
            # get the current number of points
            length = sum([len(s) for s in streamline])
            # Start the tracking process
            psi = [z_start]
            w = [np.abs(frac.calc_w(z0))]
            discharge_elements = frac.get_discharge_elements()

            # get the next points
            z1 = self.runge_kutta(z0, frac, ds)
            if np.isnan(np.real(z1)) or np.isnan(np.imag(z1)):
                break
            z3, element = self.check_streamline_exit(z0, z1, discharge_elements, frac)
            while z3 is False:
                psi.append(z1)
                w.append(np.abs(frac.calc_w(z1)))
                z0 = z1
                z1 = self.runge_kutta(z0, frac, ds)
                if np.isnan(np.real(z1)) or np.isnan(np.imag(z1)):
                    z3 = z0
                    break
                z3, element = self.check_streamline_exit(z0, z1, discharge_elements, frac)
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
                z0, z_start, elevation = self.get_exit_intersection(z3d, element, frac, frac_old, elevation)
            else:
                cond = False

        return streamline, streamline_frac, velocity, element

    @staticmethod
    def check_streamline_exit(z0, z1, discharge_elements, frac):
        """
        Function that checks if the streamline has exited the DFN
        """
        for e in discharge_elements:
            if isinstance(e, Intersection):
                z2 = e.check_chi_crossing(z0, z1, frac)
            else:
                z2 = e.check_chi_crossing(z0, z1)
            if np.isnan(np.real(z2)):
                return z1, False
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
        chi0 = gf.map_z_line_to_chi(z, endpoints)
        chi1 = np.conj(chi0)
        z0 = gf.map_chi_to_z_line(chi0*(1+dchi), endpoints)
        z1 = gf.map_chi_to_z_line(chi1*(1+dchi), endpoints)
        w0 = frac.calc_w(z0)
        w1 = frac.calc_w(z1)
        z01 = z0 + np.conj(w0) / np.abs(w0) * dchi
        z11 = z1 + np.conj(w1) / np.abs(w1) * dchi
        chi01 = gf.map_z_line_to_chi(z01, endpoints)
        chi11 = gf.map_z_line_to_chi(z11, endpoints)

        # Magnitude
        abs_w0 = np.abs(w0)
        abs_w1 = np.abs(w1)

        # check angles between w0, w1 and z-z0, z-z1, using the dot product
        diff0 = np.arccos(np.real(np.vdot(np.conj(w0) , (z0 - z))) / (abs_w0 * np.abs(z0 - z)))
        diff1 = np.arccos(np.real(np.vdot(np.conj(w1) , (z1 - z))) / (abs_w1 * np.abs(z1 - z)))
        """
        if diff0 > np.pi / 2 and diff1 > np.pi / 2:
            return z1, z, elevation

        if diff0 > np.pi / 2:
            return z1, z, elevation

        if diff1 > np.pi / 2:
            return z0, z, elevation
        """
        divide = abs_w0 / (abs_w0 + abs_w1)
        pointz0 = gf.map_2d_to_3d(z0, frac)
        pointz1 = gf.map_2d_to_3d(z1, frac)
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
            return down, z0, elevation

        elevation = (elevation - divide) / (1 - divide)
        return up, z0, elevation

    @staticmethod
    def runge_kutta(z0, frac,  ds, tolerance=1e-6, max_it=10):
        """
        Runge-Kutta method for streamline tracing.

        Parameters
        ----------
        z0 : complex
            The initial point.
        ds : float
            The step size.
        tolerance : float
            The tolerance for the error.
        max_it : int
            The maximum number of iterations.
        frac : Fracture
            The fracture to trace the streamline on.

        Returns
        -------
        z1 : complex
            The point at the end of the streamline.
        """
        w0 = frac.calc_w(z0)
        if np.isnan(np.real(w0)):
            return np.nan + np.nan*1j
        z1 = z0 + np.conj(w0)/np.abs(w0) * ds
        dz = np.abs(z1 - z0)
        it = 0
        while dz > tolerance and it < max_it:
            w1 = frac.calc_w(z1)
            if np.isnan(np.real(w1)):
                break
            z2 = z0 + np.conj(w0 + w1)/np.abs(w0 + w1) * ds
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