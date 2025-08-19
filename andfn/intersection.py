"""
Notes
-----
This module contains the intersection class.
"""

from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np
from .element import Element


class Intersection(Element):
    def __init__(
        self, label, endpoints0, endpoints1, frac0, frac1, ncoef=5, nint=10, **kwargs
    ):
        """
        Constructor for the intersection element.

        Parameters
        ----------
        label : str
            The label of the intersection
        endpoints0 : np.ndarray
            The endpoints of the first fracture that the intersection is associated with
        endpoints1 : np.ndarray
            The endpoints of the second fracture that the intersection is associated with
        frac0 : Fracture
            The first fracture that the intersection is associated with
        frac1 : Fracture
            The second fracture that the intersection is associated with
        ncoef : int
            The number of coefficients in the asymptotic expansion
        nint : int
            The number of integration points in solver
        kwargs : dict
            Additional keyword arguments
        """
        super().__init__(label=label, _id=-1, _type=0, frac0=frac0)
        # Assign frac1 to the intersection (frac0 is already assigned in Element)
        self.frac1 = frac1
        self.frac1.add_element(self)

        self.label = label
        self.endpoints0 = endpoints0
        self.endpoints1 = endpoints1
        self.ncoef = ncoef
        self.nint = nint
        self.q = 0.0

        # Create the pre-calculation variables
        self.coef = np.zeros(ncoef, dtype=complex)

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)



    def length(self):
        """
        Calculate the length of the intersection

        Returns
        -------
        length : float
            The length of the intersection
        """
        return np.abs(self.endpoints0[1] - self.endpoints0[0])

    def discharge_term(self, z, frac_is):
        """
        Calculate the discharge term for the intersection.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the discharge term at for the intersection
        frac_is : Fracture
            The fracture that contains the points

        Returns
        -------
        float
            The discharge term
        """
        if frac_is == self.frac0:
            chi = gf.map_z_line_to_chi(z, self.endpoints0)
            sign = 1
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints1)
            sign = -1
        return np.sum(np.real(mf.well_chi(chi, sign))) / len(z)

    def z_array(self, n, frac_is):
        """
        Create an array of z points along the intersection.

        Parameters
        ----------
        n : int
            The number of points
        frac_is : Fracture
            The fracture that the points are associated with

        Returns
        -------
        z : np.ndarray
            The array of z points
        """
        if frac_is == self.frac0:
            return np.linspace(self.endpoints0[0], self.endpoints0[1], n + 2)[1 : n + 1]
        return np.linspace(self.endpoints1[0], self.endpoints1[1], n + 2)[1 : n + 1]

    def omega_along_element(self, n, frac_is):
        """
        Calculate the complex potential along the intersection.

        Parameters
        ----------
        n : int
            The number of points to calculate the omega at
        frac_is : Fracture
            The fracture that the calculation is being done for

        Returns
        -------
        omega : np.ndarray
            The complex discharge potential along the intersection
        """
        z = self.z_array(n, frac_is)
        omega = frac_is.calc_omega(z)
        return omega

    def calc_omega(self, z, frac_is):
        """
        Calculate the complex potential for the intersection.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the complex potential at
        frac_is : Fracture
            The fracture that the points are associated with

        Returns
        -------
        omega : np.ndarray
            The complex discharge
        """
        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.frac0:
            chi = gf.map_z_line_to_chi(z, self.endpoints0)
            omega = mf.asym_expansion(chi, self.coef) + mf.well_chi(chi, self.q)
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints1)
            omega = mf.asym_expansion(chi, -self.coef) + mf.well_chi(chi, -self.q)
        return omega

    def calc_w(self, z, frac_is):
        """
        Calculate the complex discharge vector for the intersection.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the complex discharge vector at
        frac_is : Fracture
            The fracture that the points are associated with

        Returns
        -------
        w : np.ndarray
            The complex discharge vector
        """
        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.frac0:
            chi = gf.map_z_line_to_chi(z, self.endpoints0)
            w = -mf.asym_expansion_d1(chi, self.coef) - self.q / (2 * np.pi * chi)
            w *= (
                2
                * chi**2
                / (chi**2 - 1)
                * 2
                / (self.endpoints0[1] - self.endpoints0[0])
            )
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints1)
            w = -mf.asym_expansion_d1(chi, -self.coef) + self.q / (2 * np.pi * chi)
            w *= (
                2
                * chi**2
                / (chi**2 - 1)
                * 2
                / (self.endpoints1[1] - self.endpoints1[0])
            )
        return w

    def check_boundary_condition(self, n=10):
        """
        Check if the intersection satisfies the boundary conditions.

        Parameters
        ----------
        n : int
            The number of points to calculate the boundary condition at

        Returns
        -------
        float
            The error in the boundary condition
        """
        chi = np.exp(1j * np.linspace(0, np.pi, n))
        # Calculate the head in fracture 0
        z0 = gf.map_chi_to_z_line(chi, self.endpoints0)
        omega0 = self.frac0.calc_omega(z0, exclude=None)
        head0 = np.real(omega0) / self.frac0.t
        # Calculate the head in fracture 1
        z1 = gf.map_chi_to_z_line(chi, self.endpoints1)
        omega1 = self.frac1.calc_omega(z1, exclude=None)
        head1 = np.real(omega1) / self.frac1.t
        dhead = np.abs(head0 - head1)

        # Calculate the difference in head in the intersection
        # return np.mean(np.abs(head0 - head1)) / np.abs(np.mean(head0))

        return (np.max(dhead) - np.min(dhead)) / np.abs(np.mean(head0))

    def check_chi_crossing(self, z0, z1, frac, atol=1e-12):
        """
        Check if the line between two points crosses the intersection.

        Parameters
        ----------
        z0 : complex
            The first point
        z1 : complex
            The second point
        frac : Fracture
            The fracture that the points are associated with
        atol : float
            The absolute tolerance for the check

        Returns
        -------
        complex | bool
            The intersection point if it exists, otherwise False
        """
        # Check if the function is in the first or second fracture
        if frac == self.frac0:
            endpoints = self.endpoints0
        else:
            endpoints = self.endpoints1

        z = gf.line_line_intersection(z0, z1, endpoints[0], endpoints[1])

        if z is None:
            return False

        if np.abs(np.abs(z - z0) + np.abs(z1 - z) - np.abs(z0 - z1)) > atol:
            return False

        if (
            np.abs(
                np.abs(z - endpoints[0])
                + np.abs(z - endpoints[1])
                - np.abs(endpoints[0] - endpoints[1])
            )
            > atol
        ):
            return False

        return z
