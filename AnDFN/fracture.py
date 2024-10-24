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
    def __init__(self, label, t, radius, center, normal, ncoef=10, nint=20, elements=None):
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
        self.elements = [AnDFN.bounding.BoundingCircle(label, radius, ncoef, nint, self)]
        if elements is not None:
            self.elements.append(elements)
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
        return sum([np.abs(e.q) for e in elements])

    def check_discharge(self):
        elements = self.get_discharge_elements()
        q = 0.0
        for e in elements:
            if isinstance(e, Intersection):
                if e.fracs[1] == self:
                    q -= e.q
                    continue
            q += e.q
        return np.abs(q)

    def get_max_min_head(self):
        elements = self.get_discharge_elements()
        head = []
        for e in elements:
            if isinstance(e, Well):
                head.append(e.head)
            elif isinstance(e, ConstantHeadLine):
                head.append(e.head)
        if len(head) == 0:
            return [None, None]
        return [max(head), min(head)]

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

    def calc_w(self, z, exclude=None):
        """
        Calculates the complex discharge vector for the fracture excluding element "exclude".

        Parameters
        ----------
        z : complex
            A point in the complex z plane.
        exclude : any
            Label of element to exclude from the omega calculation.

        Returns
        -------
        w : complex
            The complex discharge vector for the fracture.
        """
        w = 0

        for e in self.elements:
            if e != exclude:
                if isinstance(e, Intersection):
                    w += e.calc_w(z, self)
                else:
                    w += e.calc_w(z)
        return w

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

    def head_from_phi(self, phi):
        """
        Calculates the head from the phi for the fracture.

        Parameters
        ----------
        phi : float
            The phi for the fracture.

        Returns
        -------
        head : float
            The head for the fracture.
        """
        return phi / self.t

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
