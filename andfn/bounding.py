"""
Notes
-----
This module contains the bounding classes.
"""

import andfn.math_functions as mf
import andfn.geometry_functions as gf
import numpy as np
from .element import Element


class BoundingCircle(Element):
    def __init__(self, label, radius, frac0, ncoef=5, nint=10, **kwargs):
        """
        Initializes the bounding circle class.

        Parameters
        ----------
        label : str or int
            The label of the bounding circle.
        r : float
            The radius of the bounding circle.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac : Fracture
            The fracture object that the bounding circle is associated with.
        """
        super().__init__(label, _id=-1, _type=1, frac0=frac0)
        self.label = label
        self.radius = radius
        self.center = 0.0 + 0.0j  # Default center at origin
        self.ncoef = ncoef
        self.nint = nint

        # Create the pre-calculation variables
        self.coef = np.zeros(ncoef, dtype=complex)

        # Correction to the stream function
        self.sign_array = None
        self.discharge_element = None
        self.element_pos = None

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_chi(self, z):
        """
        Get the chi for the bounding circle.

        Parameters
        ----------
        z : complex | np.ndarray
            A point in the complex z plane.

        Returns
        -------
        chi : complex | np.ndarray
            The complex chi for the bounding circle.
        """
        chi = gf.map_z_circle_to_chi(z, self.radius)
        if isinstance(chi, np.ndarray) and len(chi) > 1:
            chi[np.abs(chi) > 1.0 + 1e-8] = np.nan + 1j * np.nan
        else:
            if np.abs(chi) > 1.0 + 1e-8:
                chi = np.nan + 1j * np.nan
        return chi

    def calc_omega(self, z):
        """
        Calculates the omega for the bounding circle.

        Parameters
        ----------
        z : complex | np.ndarray
            A point in the complex z plane.

        Returns
        -------
        omega : complex
            The complex potential for the bounding circle.
        """
        chi = self.get_chi(z)
        omega = mf.taylor_series(chi, self.coef)
        return omega

    def calc_w(self, z):
        """
        Calculates the complex discharge vector for the bounding circle.

        Parameters
        ----------
        z : complex
            A point in the complex z plane.

        Returns
        -------
        w : complex
            The complex discharge vector for the bounding circle.
        """
        chi = self.get_chi(z)
        w = -mf.taylor_series_d1(chi, self.coef)
        w /= self.radius
        return w

    def check_boundary_condition(self, n=10):
        """
        Check if the bounding circle satisfies the boundary conditions.

        Parameters
        ----------
        n : int
            The number of points to check the boundary condition at.

        Returns
        -------
        float
            The error in the boundary condition.
        """

        # Calculate the stream function on the boundary of the fracture
        theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        z0 = gf.map_chi_to_z_circle(np.exp(1j * theta), self.radius)
        omega0 = self.frac0.calc_omega(z0, exclude=None)
        psi = np.imag(omega0)
        dpsi = np.diff(psi)
        mean_dpsi = np.abs(np.max(dpsi) - np.min(dpsi))
        return mean_dpsi / np.max(dpsi)
