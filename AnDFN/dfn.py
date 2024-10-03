"""
Notes
-----
This module contains the DFN class.
"""
import numpy as np
import pyvista as pv

from . import geometry_functions as gf
from .well import Well
from .const_head import ConstantHeadLine
from .intersection import Intersection


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
            if len(f.elements) == 1:
                elements.append(f.elements)
            else:
                elements.extend(f.elements)
        self.elements = elements

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
            self.fractures.append(new_fracture)
        else:
            self.fractures.extend(new_fracture)
        # reset the discharge matrix and elements
        self.discharge_matrix = None
        self.elements = None
        self.discharge_elements = None

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
                    # add the discharge term to the matrix for each element in the first fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z0, e.fracs[0])
                    else:
                        matrix[row, pos] = ee.discharge_term(z0)
                for ee in e.fracs[1].get_discharge_elements():
                    # add the discharge term to the matrix for each element in the second fracture
                    pos = self.discharge_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z1, e.fracs[1])
                    else:
                        matrix[row, pos] = ee.discharge_term(z1)
            else:
                z = e.z_array(self.discharge_int)
                for ee in e.frac.get_discharge_elements():
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
            for e in f.discharge_elements:
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
                matrix[row] = e.phi - np.real(np.mean(omega))
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

    def initiate_plotter(self, window_size=(800, 800), lighting='light kit', title=True, credit=True):
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
                       line_width=2):
        for i, f in enumerate(self.fractures):
            # plot the fractures
            pl.add_mesh(pv.Polygon(f.center, f.radius, normal=f.normal, n_sides=num_side, fill=filled),
                            color=color, opacity=opacity, show_edges=show_edges, line_width=line_width)
            print(f'\rPlotting fractures: {i + 1} / {len(self.fractures)}', end='')
        print('')

    def plot_fractures_flow_net(self, n_points, margin=0.1):
        plot_frac = pv.polydata()
        for f in self.fractures:
            # plot the fractures
            omega_fn, x_array, y_array = f.calc_flow_net(n_points, margin)

    def plot_elements(self, pl):
        assert self.elements is not None and len(self.elements) > 0, 'The elements have not been stored in the DFN. Use the get_elements method.'
        for i, e in enumerate(self.elements):
            if isinstance(e, Intersection):
                line = gf.map_2d_to_3d(e.endpoints[0], e.fracs[0])
                pl.add_mesh(pv.Line(line[0], line[1]), color='#000000', line_width=3)
            print(f'\rPlotting elements: {i + 1} / {len(self.elements)}', end='')
        print('')

