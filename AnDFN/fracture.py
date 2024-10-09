"""
Notes
-----
This module contains the fracture class.

make sure to allow for passing an array of (x,y) coordinates to the fracture class

add start stop for each element in the fracture class
"""
import numpy as np

from AnDFN.intersection import Intersection
from AnDFN.const_head import ConstantHeadLine
from AnDFN.well import Well
import AnDFN.bounding


class Fracture:
    def __init__(self, label, t, radius, center, normal, elements=None):
        self.label = label
        self.t = t
        self.radius = radius
        self.center = center
        self.normal = normal / np.linalg.norm(normal)
        self.x_vector = np.cross(normal, normal + np.random.rand(3))
        if np.linalg.norm(self.x_vector) == 0:
            self.x_vector = np.cross(normal, normal + np.random.rand(3))
        self.x_vector = self.x_vector / np.linalg.norm(self.x_vector)
        self.y_vector = np.cross(normal, self.x_vector)
        self.y_vector = self.y_vector / np.linalg.norm(self.y_vector)
        if elements is None:
            self.elements = [AnDFN.bounding.BoundingCircle(label, radius, 10, 20, self)]
            #self.elements = []
        else:
            self.elements = elements
            self.elements.append(AnDFN.bounding.BoundingCircle(label, radius, 10, 20, self))
        self.constant = 0.0

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

    def get_total_discharge(self):
        elements = self.get_discharge_elements()
        return sum([np.abs(e.q) for e in elements])/len(elements)

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
        omega = self.constant

        for e in self.elements:
            if e != exclude:
                if isinstance(e, Intersection):
                    omega += e.calc_omega(z, self)
                else:
                    omega += e.calc_omega(z)
        return omega

    def phi_from_head(self, head):
        """
        Calculates the head from the phi for the fracture.

        Parameters
        ----------
        head : float
            The head for the .

        Returns
        -------
        phi : float
            The phi for the fracture.
        """
        return head * self.t

    def calc_flow_net(self, n_points, margin=0.1):
        """
        Calculates the flow net for the fracture.
        Parameters
        ----------
        n_points : int
            The number of points to use for the flow net.
        margin : float
            The margin around the fracture to use for the flow net.
        """
        # Create the arrays for the flow net
        radius_margin = self.radius * (1 + margin)
        omega_fn = np.zeros((n_points,n_points), dtype=complex)
        x_array = np.linspace(-radius_margin, radius_margin, n_points)
        y_array = np.linspace(-radius_margin, radius_margin, n_points)

        # Calculate the omega for each point in the flow net
        for i, x in enumerate(x_array):
            z = x + 1j * y_array
            omega_fn[:, i] = self.calc_omega(z)

        return omega_fn, x_array, y_array
