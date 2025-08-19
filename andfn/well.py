"""
Notes
-----
This module contains the well classes.
"""

from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np
from .element import Element


class Well(Element):
    def __init__(self, label, radius, center, head, frac0, **kwargs):
        """
        Initializes the well class.

        Parameters
        ----------
        label : str or int
            The label of the well.
        radius : float
            The radius of the well.
        center : complex
            The complex location of the well.
        head : float
            The hydraulic head of the well.
        frac : Fracture
            The label of the fracture the well is associated with.
        q : float
            The flow rate of the well.
        """
        super().__init__(label=label, _id=-1, _type=2, frac0=frac0)
        self.radius = radius
        self.center = center
        self.head = head

        self.phi = frac0.phi_from_head(head)
        self.q = 0.0

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def discharge_term(self, z):
        """
        Returns the discharge term for the well.

        Parameters
        ----------
        z : complex | ndarray
            A point, or an array of points, in the complex z plane.

        Returns
        -------
        discharge : float
            The average discharge term for the well.
        """
        chi = gf.map_z_circle_to_chi(z, self.radius, self.center)
        return np.mean(np.real(mf.well_chi(chi, 1)))

    def z_array(self, n):
        """
        Returns an array of n points on the well.

        Parameters
        ----------
        n : int
            The number of points to return.

        Returns
        -------
        z : ndarray
            An array of n points on the well.
        """
        return (
            self.radius * np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False))
            + self.center
        )

    def z_array_tracking(self, n, offset=1e-2):
        """
        Returns an array of n points on the well for tracking.

        Parameters
        ----------
        n : int
            The number of points to return.
        offset : float
            The offset for the tracking.

        Returns
        -------
        z : ndarray
            An array of n points on the well.
        """
        return (
            self.radius
            * np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False))
            * (1 + offset)
            + self.center
        )

    def calc_omega(self, z):
        """
        Calculates the omega for the well. If z is inside the well, the omega is set to nan + nan*1j.

        Parameters
        ----------
        z : complex | ndarray
            A point in the complex z plane.


        Returns
        -------
        omega : complex | ndarray
            The complex potential for the well.
        """
        chi = gf.map_z_circle_to_chi(z, self.radius, self.center)
        if isinstance(chi, np.complex128):
            omega = mf.well_chi(chi, self.q)
            if np.abs(chi) < 1.0 - 1e-10:
                omega = self.head * self.frac0.t + 0 * 1j
        else:
            omega = mf.well_chi(chi, self.q)
            omega[np.abs(chi) < 1.0 - 1e-10] = self.head * self.frac0.t + 0 * 1j
        return omega

    def calc_w(self, z):
        """
        Calculates the omega for the well. If z is inside the well, the omega is set to nan + nan*1j.

        Parameters
        ----------
        z : complex | ndarray
            A point in the complex z plane.


        Returns
        -------
        w : complex | ndarray
            The complex discharge vector for the well.
        """
        chi = gf.map_z_circle_to_chi(z, self.radius, self.center)
        if isinstance(chi, np.complex128):
            if np.abs(chi) < 1.0 - 1e-10:
                return np.nan + 1j * np.nan
            w = -self.q / (2 * np.pi * chi)
        else:
            w = -self.q / (2 * np.pi * chi)
            w[np.abs(chi) < 1.0 - 1e-10] = np.nan + 1j * np.nan
        w /= self.radius
        return w

    @staticmethod
    def check_boundary_condition(n=10):
        """
        Checks the boundary condition of the well. This is allways zero as the well is solved in teh discharge matrix.
        """
        return 0.0

    def check_chi_crossing(self, z0, z1, atol=1e-10):
        """
        Checks if the line between two points, z0 and z1, crosses the well.

        Parameters
        ----------
        z0 : complex
            The first point.
        z1 : complex
            The second point.
        atol : float
            The absolute tolerance for the check

        Returns
        -------
        z : complex | bool
            The point where the line crosses the well or False if it does not cross.
        """
        chi0 = gf.map_z_circle_to_chi(z0, self.radius, self.center)
        chi1 = gf.map_z_circle_to_chi(z1, self.radius, self.center)

        chi2, chi3 = gf.line_circle_intersection(chi0, chi1, 1.0)

        if chi2 is None:
            return False

        diff0 = np.abs(np.abs(chi2 - chi0) + np.abs(chi2 - chi1) - np.abs(chi1 - chi0))
        diff1 = np.abs(np.abs(chi3 - chi0) + np.abs(chi3 - chi1) - np.abs(chi1 - chi0))
        if diff0 > atol and diff1 > atol:
            return False

        chi2 *= 1 + atol
        chi3 *= 1 + atol

        z2 = gf.map_chi_to_z_circle(chi2, self.radius, self.center)
        z3 = gf.map_chi_to_z_circle(chi3, self.radius, self.center)
        if np.abs(z2 - z0) < np.abs(z3 - z0):
            return z2
        return z3
