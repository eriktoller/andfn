"""
Notes
-----
This module contains the DFN class.
"""
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import scipy as sp

from . import geometry_functions as gf
from .well import Well
from .const_head import ConstantHeadLine
from .intersection import Intersection


def gererate_connected_fractures(num_fracs, radius_factor, center_factor, ncoef_i, nint_i, ncoef_b, nint_b, frac_surface=None):
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
    x = seg[:, 0]
    y = seg[:, 1]
    contour_complex = x + 1j * y
    line_3d = gf.map_2d_to_3d(contour_complex, f)
    pl.add_mesh(pv.MultipleLines(line_3d), color=color, line_width=line_width)


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
        self.lup = None
        self.discharge_error = 1

    def __str__(self):
        """
        Returns the string representation of the DFN.
        Returns
        -------
        str
            The string representation of the DFN.
        """
        return f'DFN: {self.label}'

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
                print(f'Added {new_fracture[0]} fractures to the DFN.')
            else:
                self.fractures.extend(new_fracture)
                print(f'Added {len(new_fracture)} fractures to the DFN.')
        else:
            self.fractures.append(new_fracture)
            print(f'Added {new_fracture} fractures to the DFN.')
        # reset the discharge matrix and elements
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None
        self.lup = None

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
        frac_list = gererate_connected_fractures(num_fracs, radius_factor, center_factor, ncoef_i, nint_i, ncoef_b,
                                                 nint_b, frac_surface)
        # Add the fractures to the DFN
        self.add_fracture(frac_list)

    def get_fracture_intersections(self, ncoef, nint, new_frac=None):
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
                    i0 = Intersection(f'{fr.label}_{fr2.label}', endpoints0, endpoints1, ncoef, nint, fr, fr2)
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
                        i0 = Intersection(f'{fr.label}_{fr2.label}', endpoints0, endpoints1, ncoef, nint, fr, fr2)
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

    ####################################################################################################################
    #                      Solve functions                                                                             #
    ####################################################################################################################

    def consolidate(self):
        for e in self.elements:
            # some function to consolidate the elements into numpy arrays for the numba solver
            e.consolidate()
            pass
        return None

    def build_discharge_matrix(self):
        """
        Builds the discharge matrix for the DFN and adds it to the DFN.
        """
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

    def lu_decomposition(self):
        """
        LU decomposition of the discharge matrix.
        """
        if self.discharge_matrix is None:
            self.build_discharge_matrix()
        self.lup = sp.linalg.lu_factor(self.discharge_matrix)

    def build_head_matrix(self):
        """
        Builds the head matrix for the DFN and stores it.
        """
        # some function to build the head matrix
        size = len(self.discharge_elements) + self.number_of_fractures()
        matrix = np.zeros(size)

        # Add the head for each discharge element
        row = 0
        for e in self.discharge_elements:
            if isinstance(e, Intersection):
                z0 = e.z_array(self.discharge_int, e.fracs[0])
                z1 = e.z_array(self.discharge_int, e.fracs[1])
                omega0 = e.fracs[0].calc_omega(z0, exclude=None)
                omega1 = e.fracs[1].calc_omega(z1, exclude=None)
                matrix[row] = np.mean(np.real(omega1)) / e.fracs[1].t - np.mean(np.real(omega0)) / e.fracs[0].t
            else:
                z = e.z_array(self.discharge_int)
                omega = e.frac.calc_omega(z, exclude=None)
                matrix[row] = e.phi - np.mean(np.real(omega))
            row += 1
        return matrix

    def solve_discharge_matrix(self, lu_decomp):
        """
        Solves the discharge matrix for the DFN and stores the discharges and constants in the elements and fractures.
        """

        # Set the discharges equal to zero
        for e in self.discharge_elements:
            e.q = 0.0

        # Set the constants equal to zero
        for f in self.fractures:
            f.constant = 0.0

        # Get the head matrix
        head_matrix = self.build_head_matrix()

        # Solve the discharge matrix
        if lu_decomp:
            discharges = sp.linalg.lu_solve(self.lup, head_matrix)
        else:
            discharges = np.linalg.solve(self.discharge_matrix, head_matrix)

        error_list = []
        # Set the discharges for each element
        for i, e in enumerate(self.discharge_elements):
            error_list.append(np.abs(discharges[i] - e.q))
            e.q = discharges[i]

        # Set the constants for each fracture
        for i, f in enumerate(self.fractures):
            error_list.append(np.abs(discharges[len(self.discharge_elements) + i] - f.constant))
            f.constant = discharges[len(self.discharge_elements) + i]

        self.discharge_error = max(error_list)

    def get_error(self):
        error_list = []
        for e in self.elements:
            error_list.append(e.error)
        return max(error_list)

    def solve(self, max_error=1e-5, max_iterations=50, boundary_check=False, tolerance=1e-2, n_boundary_check=100,
              max_iteration_boundary=5, lu_decomp=False):
        """
        Solves the DFN and saves the coefficients to the elements.
        """
        # Check if the discharge matrix has been built
        if self.discharge_matrix is None:
            self.build_discharge_matrix()
        if lu_decomp:
            self.lu_decomposition()


        cnt_error = 0
        cnt_bc = 0
        nit = 0
        nit_boundary = 0
        while cnt_error < 2 and nit < max_iterations:
            cnt = 0
            nit += 1
            self.solve_discharge_matrix(lu_decomp)
            for i, e in enumerate(self.elements):
                if isinstance(e, Well):
                    print(f'\rSolved elements: {i + 1} / {len(self.elements)}', end='')
                    e.error = 0.0
                    cnt += 1
                    continue
                # Skip solve if error is below max_error (after 3 iterations)
                if e.error < max_error and nit > 3 and cnt_error == 0:
                    cnt += 1
                    continue
                e.solve()
                print(f'\rSolved elements: {i + 1} / {len(self.elements)}', end='')
            error = self.get_error()

            # I max error is reached, set all errors to a higher value (only once)
            if error < max_error:
                cnt_error += 1
            if cnt_error == 1 and boundary_check and nit_boundary < max_iteration_boundary:
                cnt_bc = 0
                for ee in self.elements:
                    ee.error = max_error * 1.0001
                    if ee.check_boundary_condition(n=n_boundary_check) > tolerance:
                        cnt_bc += 1
                        ee.increase_coef(ee.ncoef)
                        ee.check_boundary_condition(n=n_boundary_check)
            if nit < 10:
                print(
                    f', Iteration: 0{nit}, Max error: {error:.4e}, Elements in solve loop: {len(self.elements) - cnt}', end='')
            else:
                print(f', Iteration: {nit}, Max error: {error:.4e}, Elements in solve loop: {len(self.elements) - cnt}', end='')
            if cnt_bc > 0:
                print(f', Elements BC error > tolerance: {cnt_bc}', end='')
                cnt_error = 0
                cnt_bc = 0
                nit_boundary += 1
            print('')
        if boundary_check and cnt_bc == 0:
            print('All element meet the boundary condition tolerance.')

    ####################################################################################################################
    #                    Plotting functions                                                                            #
    ####################################################################################################################

    def initiate_plotter(self, window_size=(800, 800), grid=False, lighting='light kit', title=True, off_screen=False,
                         scale=1, axis=True):
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
            Whether to add a title to the plot. Boolian or string.
        off_screen : bool
            Whether to plot off screen.
        scale : float
            The scale of the plot.
        axis : bool
            Whether to add the axis to the plot.

        Returns
        -------
        pl : pyvista.Plotter
            The plotter object.
        """
        pl = pv.Plotter(window_size=window_size, lighting=lighting, off_screen=off_screen)
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
        only_flow : bool
            Whether to plot only the fractures with flow.

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
        opacity : float
            The opacity of the fractures in the flownet.
        only_flow : bool
            Whether to plot only the fractures with flow.
        """
        if only_flow:
            fracs = self.get_flow_fractures()
            self.plot_fractures(pl, fracs=fracs)
        else:
            fracs = self.fractures

        # Calculate the flow net for each fracture
        omega_fn_list = []
        x_array_list = []
        y_array_list = []
        for i, f in enumerate(fracs):
            print(f'\rPlotting flow net: {i + 1} / {len(fracs)}', end='')
            omega_fn, x_array, y_array = f.calc_flow_net(n_points, margin)
            omega_fn_list.append(omega_fn)
            x_array_list.append(x_array)
            y_array_list.append(y_array)

        # Get the levels for the flow net
        lvs_re, lvs_im = get_lvs(lvs, omega_fn_list)

        # Plot the flow net for each fracture
        for i, f in enumerate(fracs):
            # plot the flow net using matplotlib
            contours_re = plt.contour(x_array_list[i], y_array_list[i], np.real(omega_fn_list[i]), levels=lvs_re)
            contours_im = plt.contour(x_array_list[i], y_array_list[i], np.imag(omega_fn_list[i]), levels=lvs_im)
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

    def plot_fractures_head(self, pl, lvs, n_points, line_width=2, margin=0.01, opacity=1.0, only_flow=False,
                            color_map='jet'):
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
        """
        if only_flow:
            fracs = self.get_flow_fractures()
        else:
            fracs = self.fractures

        # Calculate the flow net for each fracture
        head_fn_list = []
        maxmin_head_list = []
        x_array_list = []
        y_array_list = []
        for i, f in enumerate(fracs):
            omega_fn, x_array, y_array = f.calc_flow_net(n_points, margin)
            head_fn_list.append(f.head_from_phi(np.real(omega_fn)))
            maxmin_head = f.get_max_min_head()
            if maxmin_head[0] is not None:
                maxmin_head_list.append(maxmin_head)
            x_array_list.append(x_array)
            y_array_list.append(y_array)
            print(f'\rPlotting hydraulic head: {i + 1} / {len(fracs)}', end='')

        # Get the levels for the flow net
        head_max, head_min = np.nanmax(head_fn_list), np.nanmin(head_fn_list)
        if head_max < np.max(maxmin_head_list):
            head_max = np.max(maxmin_head_list)
        if head_min > np.min(maxmin_head_list):
            head_min = np.min(maxmin_head_list)
        # Create the levels for the equipotential contours
        lvs_re = np.linspace(head_min, head_max, lvs)

        cmap = plt.colormaps[color_map]
        colors = cmap(np.linspace(0, 1, lvs))

        # Plot the flow net for each fracture
        for i, f in enumerate(fracs):
            # plot the flow net using matplotlib
            contours_re = plt.contour(x_array_list[i], y_array_list[i], head_fn_list[i], levels=lvs_re)
            # Plot the fractures
            mean_head = np.nanmean(head_fn_list[i])
            pos_frac, = np.where(np.abs(lvs_re - mean_head) == np.min(np.abs(lvs_re - mean_head)))[0]
            color_frac = colors[pos_frac]
            self.plot_fractures(pl, filled=True, color=color_frac, opacity=opacity, show_edges=True, line_width=2.0,
                                fracs=[f])
            # Extract the contour line and plot them in 3D, real and imaginary parts
            for contour in contours_re.allsegs:
                for seg in contour:
                    if len(seg) == 0:
                        continue
                    head = f.head_from_phi(np.real(f.calc_omega(seg[0][0] + seg[0][1]*1j)))
                    pos, = np.where(np.abs(lvs_re-head) == np.min(np.abs(lvs_re-head)))[0]
                    color = colors[pos]
                    plot_line_3d(seg, f, pl, color, line_width=line_width)

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
        print('')



    def plot_elements(self, pl):
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

    ####################################################################################################################
    #                    Streamline tracking functions                                                                 #
    ####################################################################################################################
    def streamline_tracking(self, z0, elevation, frac):
        """

        Parameters
        ----------
        z0 : complex
            Starting point for streamline tracking
        elevation : float
            Elevation of the starting point
        frac: Fracture
            The fracture where to start the streamline tracking

        Returns
        -------
        streamlines: ndarray
            Array with streamline
        """
        psi = []
        # Start the tracking process
    
        # Move to the next fracture
        streamline = np.array(psi)
        return streamline

    @staticmethod
    def runge_kutta(z0, ds, tolerance, max_it, frac):
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
        z1 = z0 + np.conj(w0)/np.abs(w0) * ds
        dz = np.abs(z1 - z0)
        it = 0
        while dz > tolerance and it < max_it:
            w1 = frac.calc_w(z1)
            z2 = z0 + (np.conj(w0) + np.conj(w1))/(np.abs(w0) + np.abs(w1)) * ds
            dz = np.abs(z2 - z1)
            z1 = z2
            it += 1

        return z1
