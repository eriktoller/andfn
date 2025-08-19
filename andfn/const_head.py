"""
Notes
-----
This module contains the constant head classes.
"""

from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np
from .element import Element


class ConstantHeadLine(Element):
    def __init__(self, label, endpoints0, head, frac0, ncoef=5, nint=10, **kwargs):
        """
        Constructor for the constant head line element.

        Parameters
        ----------
        label : str
            The label of the constant head line
        endpoints0 : np.ndarray[complex]
            The endpoints of the constant head line
        head : float
            The head of the constant head line
        frac0 : Fracture
            The fracture that the constant head line is associated with
        ncoef : int
            The number of coefficients in the asymptotic expansion
        nint : int
            The number of integration points in the asymptotic expansion
        kwargs : dict
            Additional keyword arguments
        """
        super().__init__(label, _id=-1, _type=3, frac0=frac0)
        self.label = label
        self.endpoints0 = endpoints0
        self.ncoef = ncoef
        self.nint = nint
        self.q = 0.0
        self.head = head

        # Create the pre-calculation variables
        self.coef = np.zeros(ncoef, dtype=complex)
        self.phi = frac0.phi_from_head(head)

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    def discharge_term(self, z):
        """
        Calculate the discharge term for the constant head line.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the discharge term at

        Returns
        -------
        float
            The discharge term
        """
        chi = gf.map_z_line_to_chi(z, self.endpoints0)
        return np.sum(np.real(mf.well_chi(chi, 1))) / len(z)

    def update_head(self, head):
        """
        Update the head of the constant head line.

        Parameters
        ----------
        head : float
            The new head of the constant head line--
        """
        self.head = head
        self.phi = self.frac0.phi_from_head(head)

    def length(self):
        """
        Calculate the length of the constant head line.

        Returns
        -------
        float
            The length of the constant head line
        """
        return np.abs(self.endpoints0[0] - self.endpoints0[1])

    def z_array(self, n):
        """
        Create an array of z points along the constant head line.

        Parameters
        ----------
        n : int
            The number of points to create

        Returns
        -------
        np.ndarray
            The array of z points
        """
        return np.linspace(self.endpoints0[0], self.endpoints0[1], n + 2)[1 : n + 1]

    def omega_along_element(self, n, frac_is):
        """
        Calculate the omega along the constant head line.

        Parameters
        ----------
        n : int
            The number of points to calculate the omega at
        frac_is : Fracture
            The fracture that the calculation is being done for

        Returns
        -------
        np.ndarray
            The complex discharge potential along the constant head line
        """
        z = self.z_array(n)
        omega = frac_is.calc_omega(z)
        return omega

    def z_array_tracking(self, n, offset=1e-3):
        """
        Create an array of z points along the constant head line with an offset.

        Parameters
        ----------
        n : int
            The number of points to create
        offset : float
            The offset to use

        Returns
        -------
        np.ndarray
            The array of z points
        """
        chi = np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False)) * (1 + offset)
        return gf.map_chi_to_z_line(chi, self.endpoints0)

    def calc_omega(self, z):
        """
        Calculate the complex discharge potential for the constant head line.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the complex discharge potential at

        Returns
        -------
        np.ndarray
            The complex discharge potential
        """
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints0)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef) + mf.well_chi(chi, self.q)
        return omega

    def calc_w(self, z):
        """
        Calculate the complex discharge vector for the constant head line.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the complex discharge vector at

        Returns
        -------
        np.ndarray
            The complex discharge vector
        """
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints0)
        # Calculate w
        w = -mf.asym_expansion_d1(chi, self.coef) - self.q / (2 * np.pi * chi)
        w *= 2 * chi**2 / (chi**2 - 1) * 2 / (self.endpoints0[1] - self.endpoints0[0])
        return w

    def check_boundary_condition(self, n=10):
        """
        Check if the constant head line satisfies the boundary conditions.

        Parameters
        ----------
        n : int
            The number of points to check the boundary condition at

        Returns
        -------
        float
            The error in the boundary condition
        """
        chi = np.exp(1j * np.linspace(0, np.pi, n))
        # Calculate the head in fracture 0
        z0 = gf.map_chi_to_z_line(chi, self.endpoints0)
        omega0 = self.frac0.calc_omega(z0, exclude=None)

        return np.mean(np.abs(self.phi - np.real(omega0))) / np.abs(self.phi)

    def check_chi_crossing(self, z0, z1, atol=1e-12):
        """
        Check the line between two points crosses the constant head line.

        Parameters
        ----------
        z0 : complex
            The first point
        z1 : complex
            The second point
        atol : float
            The absolute tolerance for the check

        Returns
        -------
        complex | bool
            The intersection point if it exists, otherwise False
        """
        z = gf.line_line_intersection(z0, z1, self.endpoints0[0], self.endpoints0[1])

        # TODO: implement this if np.abs(np.imag(chi1)) > atol and np.abs(np.imag(chi2)) > atol:

        if z is None:
            return False

        if np.abs(np.abs(z - z0) + np.abs(z1 - z) - np.abs(z0 - z1)) > atol:
            return False

        if (
            np.abs(
                np.abs(z - self.endpoints0[0])
                + np.abs(z - self.endpoints0[1])
                - np.abs(self.endpoints0[0] - self.endpoints0[1])
            )
            > atol
        ):
            return False

        return z
