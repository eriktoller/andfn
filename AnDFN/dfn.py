"""
Notes
-----
This module contains the DFN class.
"""
import numpy as np
import pyvista as pv
from AnDFN.well import Well
from AnDFN.intersection import Intersection
from AnDFN.const_head import ConstantHeadLine


class DFN:
    def __init__(self, label, fractures=None, elements=None):
        self.label = label
        if fractures is None:
            fractures = []
        else:
            self.fractures = fractures
        if elements is None:
            elements = []
        else:
            self.elements = elements

    def number_of_fractures(self):
        return len(self.fractures)

    def number_of_elements(self):
        return len(self.elements)

    def make_disc_elements_list(self):
        return [e for e in self.elements
                if isinstance(e, Intersection)
                or isinstance(e, ConstantHeadLine)
                or isinstance(e, Well)]

    def add_fracture(self, new_fracture):
        self.fractures.append(new_fracture)
        if len(new_fracture.elements) > 0:
            self.elements.extend(new_fracture.elements)
        else:
            self.elements.append(new_fracture.elements)
        return None

    def delete_fracture(self, fracture):
        self.fractures.remove(fracture)
        for element in fracture.elements:
            self.elements.remove(element)
        return None

    def consolidate(self):
        for e in self.elements:
            # some function to consolidate the elements into numpy arrays for the numba solver
            pass
        return None

    def build_discharge_matrix(self, n):
        # some function to build the discharge matrix
        # TODO: test this function
        q_elements = self.make_disc_elements_list()
        size = len(q_elements) + self.number_of_fractures()
        matrix = np.zeros((size, size))

        # Add the discharge for each discharge element
        row = 0
        for e in q_elements:
            z = e.z_array(n)
            if isinstance(e, Intersection):
                for ee in e.fracs[0].get_discharge_elements():
                    # add the discharge term to the matrix for each element in the first fracture
                    pos = q_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z, e.fracs[0])
                    else:
                        matrix[row, pos] = ee.discharge_term(z)
                for ee in e.fracs[1].get_discharge_elements():
                    # add the discharge term to the matrix for each element in the second fracture
                    pos = q_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z, e.fracs[1])
                    else:
                        matrix[row, pos] = ee.discharge_term(z)
            else:
                for ee in e.frac.get_discharge_elements():
                    # add the discharge term to the matrix for each element in the fracture
                    pos = q_elements.index(ee)
                    if isinstance(ee, Intersection):
                        matrix[row, pos] = ee.discharge_term(z, e.frac)
                    else:
                        matrix[row, pos] = ee.discharge_term(z)
            pos_f = self.fractures.index(e.frac)
            matrix[row, len(q_elements) + pos_f] = 1
            row += 1

        # Add the constants for each fracture
        for f in self.fractures:
            # fill the matrix for the fractures
            for e in f.get_discharge_elements():
                # add the discharge term to the matrix for each element in the fracture
                pos = q_elements.index(e)
                if isinstance(e, Intersection):
                    if e.fracs[0] == f:
                        matrix[row, pos] = 1
                    else:
                        matrix[row, pos] = -1
                else:
                    matrix[row, pos] = 1
            row += 1

        return matrix

    def solve_discharge_matrix(self):
        # some function to solve the discharge matrix
        # TODO: implement this function you lazy bum :) (you can do it! or can you? but you can! not sure though)
        pass

    def plot_fractures(self, num_side=50, filled=True):
        plot_frac = pv.polydata()
        for f in self.fractures:
            # plot the fractures
            plot_frac += pv.Polygon(f.radius, f.center, f.normal, n_sides=num_side, fill=filled)
        return plot_frac
