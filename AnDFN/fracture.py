"""
Notes
-----
This module contains the fracture class.

make sure to allow for passing an array of (x,y) coordinates to the fracture class

add start stop for each element in the fracture class
"""
from AnDFN.intersection import Intersection
from AnDFN.const_head import ConstantHeadLine
from AnDFN.well import Well


class Fracture:
    def __init__(self, label, t, radius, center, normal, x_vector, y_vector, elements=None):
        self.label = label
        self.t = t
        self.radius = radius
        self.center = center
        self.normal = normal
        self.x_vector = x_vector
        self.y_vector = y_vector
        if elements is None:
            elements = []
        else:
            self.elements = elements

    def __str__(self):
        return f'Fracture {self.label}'

    def add_element(self, new_element):
        if new_element in self.elements:
            print('Element already in fracture.')
        else:
            self.elements.append(new_element)

    def get_discharge_elements(self):
        return [e for e in self.elements
                if isinstance(e, Intersection)
                or isinstance(e, ConstantHeadLine)
                or isinstance(e, Well)]

    def set_new_label(self, new_label):
        self.label = new_label

    def calc_omega(self, z, exclude=None):
        """
        Calculates the omega for the fracture excluding element "exclude".

        Parameters
        ----------
        z : complex
            A point in the complex z plane.
        exclude : any
            Label of element to exclude from the omega calculation.

        Returns
        -------
        omega : complex
            The complex potential for the fracture.
        """
        omega = 0
        if exclude is None:
            for e in self.elements:
                omega += e.calc_omega(z)
        else:
            for e in self.elements:
                if e.label != exclude:
                    omega += e.calc_omega(z)
        return omega

