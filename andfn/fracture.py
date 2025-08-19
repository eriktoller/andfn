"""
Notes
-----
This module contains the fracture class.
"""

import numpy as np

from andfn.intersection import Intersection
from andfn.const_head import ConstantHeadLine
from andfn.well import Well
import andfn.bounding
from .element import fracture_dtype, fracture_dtype_hpc, fracture_index_dtype


class Fracture:
    def __init__(
        self,
        label,
        t,
        radius,
        center,
        normal,
        aperture=1e-6,
        ncoef=5,
        nint=10,
        elements=None,
        **kwargs,
    ):
        """
        Initializes the fracture class.

        Parameters
        ----------
        label : str
            The label of the fracture.
        t : float
            The transmissivity of the fracture.
        radius : float
            The radius of the fracture.
        center : np.ndarray
            The center of the fracture.
        normal : np.ndarray
            The normal vector of the fracture.
        ncoef : int
            The number of coefficients for the bounding circle that bounds the fracture.
        nint : int
            The number of integration points for the bounding circle that bounds the fracture.
        elements : list
            A list of elements that the fracture is associated with. If elements is None the bounding circle will be
            created.
        kwargs : dict
            Additional keyword arguments.
        """
        self.label = label
        self._id = 0
        self.t = t
        self.aperture = aperture
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
            self.elements = []
            andfn.bounding.BoundingCircle(label, radius, self, ncoef, nint)
        self.constant = 0.0

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Returns the string representation of the fracture.

        Returns
        -------
        str
            The string representation of the fracture.
        """
        return f"Fracture {self.label}"

    def set_id(self, _id):
        """
        Sets the id for the fracture.

        Parameters
        ----------
        _id : int
            The id for the fracture.
        """
        self._id = _id

    def consolidate(self):
        """
        Consolidates the fracture into a structured array.

        Returns
        -------
        fracture_struc_array : np.ndarray
            The structured array for the fracture.
        fracture_index_array : np.ndarray
            The structured array for the fracture index.
        """
        fracture_struc_array = np.empty(1, dtype=fracture_dtype)

        fracture_struc_array["_id"][0] = self._id
        fracture_struc_array["t"][0] = self.t
        fracture_struc_array["radius"][0] = self.radius
        fracture_struc_array["center"][0] = self.center
        fracture_struc_array["normal"][0] = self.normal
        fracture_struc_array["x_vector"][0] = self.x_vector
        fracture_struc_array["y_vector"][0] = self.y_vector
        fracture_struc_array["elements"][0] = np.array([e._id for e in self.elements])
        fracture_struc_array["constant"][0] = self.constant

        fracture_index_array = np.array(
            [(self.label, self._id)], dtype=fracture_index_dtype
        )

        return fracture_struc_array, fracture_index_array

    def consolidate_hpc(self):
        """
        Consolidates the fracture into a structured array for HPC.

        Returns
        -------
        fracture_struc_array : np.ndarray
            The structured array for the fracture.
        fracture_index_array : np.ndarray
            The structured array for the fracture index.
        """
        fracture_struc_array = np.empty(1, dtype=fracture_dtype_hpc)

        fracture_struc_array["_id"][0] = self._id
        fracture_struc_array["t"][0] = self.t
        fracture_struc_array["radius"][0] = self.radius
        fracture_struc_array["center"][0] = self.center
        fracture_struc_array["normal"][0] = self.normal
        fracture_struc_array["x_vector"][0] = self.x_vector
        fracture_struc_array["y_vector"][0] = self.y_vector
        elements = np.array([e._id for e in self.elements])
        fracture_struc_array["elements"][0][: elements.size] = elements
        fracture_struc_array["nelements"][0] = elements.size
        fracture_struc_array["constant"][0] = self.constant

        fracture_index_array = np.array(
            [(self.label, self._id)], dtype=fracture_index_dtype
        )

        return fracture_struc_array, fracture_index_array

    def unconsolidate(self, fracture_struc_array, fracture_index_array):
        """
        Unconsolidates the fracture from the structured array.

        Parameters
        ----------
        fracture_struc_array : np.ndarray
            The structured array for the fracture.
        fracture_index_array : np.ndarray
        """
        self._id = fracture_struc_array["_id"]
        self.t = fracture_struc_array["t"]
        self.radius = fracture_struc_array["radius"]
        self.center = fracture_struc_array["center"]
        self.normal = fracture_struc_array["normal"]
        self.x_vector = fracture_struc_array["x_vector"]
        self.y_vector = fracture_struc_array["y_vector"]
        self.elements = [
            e
            for e in self.elements
            if e._id
            in fracture_struc_array["elements"][: fracture_struc_array["nelements"]]
        ]
        self.constant = fracture_struc_array["constant"]

        self.label = fracture_index_array["label"]

    def unconsolidate_hpc(self, fracture_struc_array, fracture_index_array):
        """
        Unconsolidates the fracture from the structured array for HPC.

        Parameters
        ----------
        fracture_struc_array : np.ndarray
            The structured array for the fracture.
        fracture_index_array : np.ndarray
            The structured array for the fracture index.
        """
        self._id = fracture_struc_array["_id"]
        self.t = fracture_struc_array["t"]
        self.radius = fracture_struc_array["radius"]
        self.center = fracture_struc_array["center"]
        self.normal = fracture_struc_array["normal"]
        self.x_vector = fracture_struc_array["x_vector"]
        self.y_vector = fracture_struc_array["y_vector"]
        self.elements = [
            e for e in self.elements if e._id in fracture_struc_array["elements"]
        ]
        self.constant = fracture_struc_array["constant"]

        self.label = fracture_index_array["label"]

    def add_element(self, new_element):
        """
        Adds a new element to the fracture.

        Parameters
        ----------
        new_element : Element
            The element to add to the fracture.
        """
        if new_element in self.elements:
            print("Element already in fracture.")
        else:
            self.elements.append(new_element)

    def get_discharge_elements(self):
        """
        Returns the elements in the fracture that have a discharge.

        Returns
        -------
        list
            A list of elements in the fracture that have a discharge.
        """
        return [
            e
            for e in self.elements
            if isinstance(e, Intersection)
            or isinstance(e, ConstantHeadLine)
            or isinstance(e, Well)
        ]

    def get_discharge_entries(self):
        """
        Returns the elements in the fracture that have a discharge.

        Returns
        -------
        int
            The number of discharge entries required in the discharge matrix.
        """
        el = self.get_discharge_elements()
        len_el = len(el)
        cnt = (len_el - 1) * len_el + len_el
        for e in el:
            if isinstance(e, Intersection):
                cnt += 1
        return cnt

    def get_total_discharge(self):
        """
        Returns the total discharge from absolute values in the fracture.

        Returns
        -------
        float
            The total discharge in the fracture.
        """
        elements = self.get_discharge_elements()
        return sum([np.abs(e.q) for e in elements])

    def check_discharge(self):
        """
        Checks so the discharge in the fracture adds up to zero.

        Returns
        -------
        float
            The total discharge in the fracture.
        """
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
        """
        Returns the maximum and minimum head from the constant head elements for the fracture.

        Returns
        -------
        head : list
            A list containing the maximum and minimum head for the fracture.
        """
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
        """
        Sets a new label for the fracture.

        Parameters
        ----------
        new_label : str
            The new label for the fracture.
        """
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

    def calc_head(self, z):
        """
        Calculates the head for the fracture at a point z in the complex plane.

        Parameters
        ----------
        z : complex
            A point in the complex z plane.

        Returns
        -------
        head : float
            The head for the fracture at the point z.
        """
        omega = self.calc_omega(z)
        return self.head_from_phi(omega.real)

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
        omega_fn = np.zeros((n_points, n_points), dtype=complex)
        x_array = np.linspace(-radius_margin, radius_margin, n_points)
        y_array = np.linspace(-radius_margin, radius_margin, n_points)

        # Calculate the omega for each point in the flow net
        for i, x in enumerate(x_array):
            z = x + 1j * y_array
            omega_fn[:, i] = self.calc_omega(z)

        return omega_fn, x_array, y_array
