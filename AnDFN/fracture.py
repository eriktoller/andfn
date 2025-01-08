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
from .element import fracture_dtype, fracture_index_dtype


class Fracture:
    def __init__(self, label, t, radius, center, normal, ncoef=5, nint=10, elements=None, **kwargs):
        self.label = label
        self.id_ = 0
        self.t = t
        self.radius = radius
        self.center = center
        self.normal = normal / np.linalg.norm(normal)
        self.x_vector = np.cross(normal, normal + np.array([1, 0, 0]))
        if np.linalg.norm(self.x_vector) == 0:
            self.x_vector = np.cross(normal, normal + np.array([1, 1, 1]))
        self.x_vector = self.x_vector / np.linalg.norm(self.x_vector)
        self.y_vector = np.cross(normal, self.x_vector)
        self.y_vector = self.y_vector / np.linalg.norm(self.y_vector)
        if elements is False:
            self.elements = []
        elif elements is not None:
            self.elements.append(elements)
        else:
            self.elements = [AnDFN.bounding.BoundingCircle(label, radius, self, ncoef, nint)]
        self.constant = 0.0

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f'Fracture {self.label}'

    def set_id(self, id_):
        self.id_ = id_

    def consolidate(self):
        fracture_struc_array = np.empty(1, dtype=fracture_dtype)

        fracture_struc_array['id_'][0] = self.id_
        fracture_struc_array['t'][0] = self.t
        fracture_struc_array['radius'][0] = self.radius
        fracture_struc_array['center'][0] = self.center
        fracture_struc_array['normal'][0] = self.normal
        fracture_struc_array['x_vector'][0] = self.x_vector
        fracture_struc_array['y_vector'][0] = self.y_vector
        fracture_struc_array['elements'][0] = np.array([e.id_ for e in self.elements])
        fracture_struc_array['constant'][0] = self.constant

        fracture_index_array = np.array([(
            self.label,
            self.id_
        )], dtype=fracture_index_dtype)

        return fracture_struc_array, fracture_index_array

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
                if e.frac1 == self:
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
        z : complex | np.ndarray
            A point in the complex z plane.
        exclude : any
            Label of element to exclude from the omega calculation.

        Returns
        -------
        omega : complex | np.ndarray
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
        w = 0.0 + 0.0j

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
