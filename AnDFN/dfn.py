"""
Notes
-----
This module contains the DFN class.
"""
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from . import geometry_functions as gf
from .well import Well
from .const_head import ConstantHeadLine
from .intersection import Intersection


def gererate_connected_fractures(num_fracs, radius_factor, center_factor, ncoef, nint, frac_surface=None):
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
    ncoef : int
        The number of coefficients to use for the intersection elements.
    nint : int
        The number of integration points to use for the intersection elements.
    frac_surface : Fracture
        The fracture to use as the surface fracture.

    Returns
    -------
    frac_list : list
        The list of fractures.

    """
    print('Generating fractures...')
    fracs = gf.generate_fractures(num_fracs, radius_factor=radius_factor, center_factor=center_factor)

    print('Analyzing intersections...')
    frac_list = gf.get_connected_fractures(fracs, ncoef, nint, frac_surface)

    return frac_list

class DFN:
    def __init__(self, label, discharge_int=50):
        """
        Initializes the DFN class.
        Parameters
        ----------
        label : str or int
            The label of the DFN.
        discharge_int : int
            The number of points to use for the discharge integral.

        """
        self.label = label
        self.discharge_int = discharge_int
        self.fractures = []
        self.elements = None

        # Initialize the discharge matrix
        self.discharge_matrix = None
        self.discharge_elements = None

    def number_of_fractures(self):
        return len(self.fractures)

    def number_of_elements(self):
        return len(self.elements)

    def get_elements(self):
        """
        Gets the elements from the fractures and add store them in the DFN.
        """
        elements = []
        for f in self.fractures:
            if f.elements is None or len(f.elements) == 0:
                continue
            for i in range(len(f.elements)):
                if f.elements[i] not in elements:
                    elements.append(f.elements[i])
        self.elements = elements
        print(f'Added {len(self.elements)} elements to the DFN.')

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

    def add_fracture(self, new_fracture):
        """
        Adds a fracture to the DFN.
        Parameters
        ----------
        new_fracture : Fracture | list
            The fracture to add to the DFN.
        """
        if len(new_fracture) == 1:
            self.fractures.append(new_fracture[0])
        else:
            self.fractures.extend(new_fracture)
        # reset the discharge matrix and elements
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None
        print(f'Added {len(new_fracture)} fractures to the DFN.')

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

    def consolidate(self):
        for e in self.elements:
            # some function to consolidate the elements into numpy arrays for the numba solver
            pass
        return None

    def build_discharge_matrix(self):
        # some function to build the discharge matrix
        # TODO: test this function
        self.get_discharge_elements()
        size = len(self.discharge_elements) + self.number_of_fractures()
        matrix = np.zeros((size, size))

        # Add the discharge for each discharge element
        row = 0
        for e in self.discharge_elements:
            if isinstance(e, Intersection):
                z0 = e.z_array(self.discharge_int, e.fracs[0])
                z1 = e.z_array(self.discharge_int, e.fracs[1])
                for ee in e.fracs[0].get_discharge_elements():
                    if ee == e:
                        continue
                    # add the discharge term to the matrix for each element in the first fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z0, e.fracs[0])
                    else:
                        matrix[row, pos] = ee.discharge_term(z0)
                for ee in e.fracs[1].get_discharge_elements():
                    if ee == e:
                        continue
                    # add the discharge term to the matrix for each element in the second fracture
                    pos = self.discharge_elements.index(ee)
                    # TODO: has to be minus because it is subtracted from frac0
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = -ee.discharge_term(z1, e.fracs[1])
                    else:
                        matrix[row, pos] = -ee.discharge_term(z1)
                pos_f0 = self.fractures.index(e.fracs[0])
                matrix[row, len(self.discharge_elements) + pos_f0] = 1
                pos_f1 = self.fractures.index(e.fracs[1])
                matrix[row, len(self.discharge_elements) + pos_f1] = -1
            else:
                z = e.z_array(self.discharge_int)
                for ee in e.frac.get_discharge_elements():
                    if ee == e:
                        continue
                    # add the discharge term to the matrix for each element in the fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z, e.frac)
                    else:
                        matrix[row, pos] = ee.discharge_term(z)
                pos_f = self.fractures.index(e.frac)
                matrix[row, len(self.discharge_elements) + pos_f] = 1
            row += 1

        # Add the constants for each fracture
        for f in self.fractures:
            # fill the matrix for the fractures
            for e in f.elements:
                if e in self.discharge_elements:
                    # add the discharge term to the matrix for each element in the fracture
                    pos = self.discharge_elements.index(e)
                    if isinstance(e, Intersection):
                        if e.fracs[0] == f:
                            matrix[row, pos] = 1
                        else:
                            matrix[row, pos] = -1
                    else:
                        matrix[row, pos] = 1
            row += 1

        self.discharge_matrix = matrix

    def build_head_matrix(self):
        # some function to build the head matrix
        size = len(self.discharge_elements) + self.number_of_fractures()
        matrix = np.zeros(size)

        # Add the head for each discharge element
        row = 0
        for e in self.discharge_elements:
            if isinstance(e, Intersection):
                z0 = e.z_array(self.discharge_int, e.fracs[0])
                z1 = e.z_array(self.discharge_int, e.fracs[1])
                omega0 = e.fracs[0].calc_omega(z0, exclude=e)
                omega1 = e.fracs[1].calc_omega(z1, exclude=e)
                matrix[row] = np.real(np.mean(omega0)) / e.fracs[0].t - np.real(np.mean(omega1)) / e.fracs[1].t
            else:
                z = e.z_array(self.discharge_int)
                omega = e.frac.calc_omega(z, exclude=e)
                matrix[row] = e.phi*e.frac.t - np.real(np.mean(omega))
            row += 1
        return matrix

    def solve_discharge_matrix(self):
        # some function to solve the discharge matrix
        # TODO: test this function

        # Check if the discharge matrix has been built
        if self.discharge_matrix is None:
            self.build_discharge_matrix()

        # Set the discharges equal to zero
        for e in self.discharge_elements:
            e.q = 0.0

        # Set the constants equal to zero
        for f in self.fractures:
            f.constant = 0.0

        # Get the head matrix
        head_matrix = self.build_head_matrix()

        # Solve the discharge matrix
        discharges = np.linalg.solve(self.discharge_matrix, head_matrix)

        # Set the discharges for each element
        for i, e in enumerate(self.discharge_elements):
            e.q = discharges[i]

        # Set the constants for each fracture
        for i, f in enumerate(self.fractures):
            f.constant = discharges[len(self.discharge_elements) + i]

    def get_error(self):
        error_list = []
        for e in self.elements:
            error_list.append(e.error)
        return max(error_list)

    def solve(self):
        cnt = 0
        nit = 0
        error = 1
        while error > 1e-6 and nit < 50:
            cnt = 0
            nit += 1
            self.solve_discharge_matrix()
            for i, e in enumerate(self.elements):
                if isinstance(e, Well):
                    print(f'\rSolved elements: {i + 1} / {len(self.elements)}', end='')
                    cnt += 1
                    continue
                e.solve()
                print(f'\rSolved elements: {i+1} / {len(self.elements)}', end='')
            error = self.get_error()
            print(f', Iteration: {nit}, Max error: {error}')
        print('')

    def generate_connected_DFN(self, num_fracs, radius_factor, center_factor, ncoef, nint, frac_surface=None):
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
        ncoef : int
            The number of coefficients to use for the intersection elements.
        nint : int
            The number of integration points to use for the intersection elements.
        frac_surface : Fracture
            The fracture to use as the surface fracture.

        Returns
        -------

        """
        # Generate the connected fractures
        frac_list = gererate_connected_fractures(num_fracs, radius_factor, center_factor, ncoef, nint, frac_surface)
        # Add the fractures to the DFN
        self.add_fracture(frac_list)

    def get_fracture_intersections(self, ncoef, nint):
        """
        Finds the intersections between the fractures in the DFN and adds them to the DFN.
        Parameters
        ----------
        ncoef : int
            The number of coefficients to use for the intersection elements.
        nint : int
            The number of integration points to use for the intersection elements
        """
        for i in range(len(self.fractures)):
            fr = self.fractures[i]
            for k in range(i + 1, len(self.fractures)):
                fr2 = self.fractures[k]
                if fr == fr2:
                    continue
                endpoints0, endpoints1 = gf.fracture_intersection(fr, fr2)
                if endpoints0 is not None:
                    i0 = Intersection(f'{fr.label}_{fr2.label}', endpoints0, endpoints1, ncoef, nint, fr, fr2)
                    fr.add_element(i0)
                    fr2.add_element(i0)

    def initiate_plotter(self, window_size=(800, 800), lighting='light kit', title=True, credit=True):
        """
        Initiates the plotter for the DFN.
        Parameters
        ----------
        window_size : tuple
            The size of the plot window.
        lighting : str
            The type of lighting to use.
        title : bool or str
            Whether to add a title to the plot. Boolian or string.
        credit : bool
            Whether to add the credits to the plot.

        Returns
        -------
        pl : pyvista.Plotter
            The plotter object.
        """
        pl = pv.Plotter(window_size=window_size, lighting=lighting)
        _ = pl.add_axes(
            line_width=2,
            cone_radius=0.3,
            shaft_length=0.7,
            tip_length=0.3,
            ambient=0.5,
            label_size=(0.2, 0.08),
        )
        if isinstance(title, str):
            _ = pl.add_text(title, font_size=10, position='upper_left', color='k', shadow=True)
            return pl
        if title:
            _ = pl.add_text(f'DFN: {self.label}', font_size=10, position='upper_left', color='k', shadow=True)
        if credit:
            _ = pl.add_text('Made with AnDFN', font_size=4, position='lower_right', color='k', shadow=True)
        return pl

    def plot_fractures(self, pl,  num_side=50, filled=True, color='#FFFFFF', opacity=1.0, show_edges=True,
                       line_width=2.0):
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
        color : str
            The color of the fractures.
        opacity : float
            The opacity of the fractures.
        show_edges : bool
            Whether to show the edges of the fractures.
        line_width : float
            The line width of the lines.
        """
        for i, f in enumerate(self.fractures):
            # plot the fractures
            pl.add_mesh(pv.Polygon(f.center, f.radius, normal=f.normal, n_sides=num_side, fill=filled),
                            color=color, opacity=opacity, show_edges=show_edges, line_width=line_width)
            print(f'\rPlotting fractures: {i + 1} / {len(self.fractures)}', end='')
        print('')

    def plot_fractures_flow_net(self, pl, lvs, n_points, margin=0.1):
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
        margin : float
            The margin around the fracture to use for the flow net.
        """
        # Calculate the flow net for each fracture
        omega_fn_list = []
        x_array_list = []
        y_array_list = []
        for i, f in enumerate(self.fractures):
            omega_fn, x_array, y_array = f.calc_flow_net(n_points, margin)
            omega_fn_list.append(omega_fn)
            x_array_list.append(x_array)
            y_array_list.append(y_array)
            print(f'\rPlotting flow net: {i + 1} / {len(self.fractures)}', end='')

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

        # Plot the flow net for each fracture
        for i, f in enumerate(self.fractures):
            # plot the flow net using matplotlib
            contours_re = plt.contour(x_array_list[i], y_array_list[i], np.real(omega_fn_list[i]), levels=lvs_re)
            contours_im = plt.contour(x_array_list[i], y_array_list[i], np.imag(omega_fn_list[i]), levels=lvs_im)
            # Extract the contour line and plot them in 3D, real and imaginary parts
            for contour in contours_re.allsegs:
                for seg in contour:
                    if len(seg) == 0:
                        continue
                    x = seg[:, 0]
                    y = seg[:, 1]
                    contour_complex = x + 1j * y
                    line_3d = gf.map_2d_to_3d(contour_complex, f)
                    pl.add_mesh(pv.MultipleLines(line_3d), color='red', line_width=2)
            for contour in contours_im.allsegs:
                for seg in contour:
                    if len(seg) == 0:
                        continue
                    x = seg[:, 0]
                    y = seg[:, 1]
                    contour_complex = x + 1j * y
                    line_3d = gf.map_2d_to_3d(contour_complex, f)
                    pl.add_mesh(pv.MultipleLines(line_3d), color='blue', line_width=2)

    def plot_elements(self, pl):
        """
        Plots the elements in the DFN.
        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object.
        """
        # Check if the elements have been stored in the DFN
        assert self.elements is not None and len(self.elements) > 0, 'The elements have not been stored in the DFN. Use the get_elements method.'
        # Plot the elements
        for i, e in enumerate(self.elements):
            if isinstance(e, Intersection):
                line = gf.map_2d_to_3d(e.endpoints[0], e.fracs[0])
                pl.add_mesh(pv.Line(line[0], line[1]), color='#000000', line_width=3)
            if isinstance(e, ConstantHeadLine):
                line = gf.map_2d_to_3d(e.endpoints, e.frac)
                pl.add_mesh(pv.Line(line[0], line[1]), color='#000000', line_width=3)
            if isinstance(e, Well):
                point = gf.map_2d_to_3d(e.center, e.frac)
                pl.add_mesh(pv.Polygon(point, e.radius, normal=e.frac.normal, n_sides=50),
                            color='#000000', line_width=3)
            print(f'\rPlotting elements: {i + 1} / {len(self.elements)}', end='')
        print('')

