"""
Notes
-----
This module contains the impermeable object classes.
"""

import numpy as np
from .element import Element
from . import math_functions as mf
from . import geometry_functions as gf


class _ImpermeableEllipse:
    def __init__(self, label, focis, nu, ncoef, nint, frac):
        """
        Initializes the impermeable ellipse class.

        Parameters
        ----------
        label : str or int
            The label of the impermeable ellipse.
        focis : list
            The focis of the impermeable ellipse.
        nu : float
            The angle of the major axis of the impermeable ellipse.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac : Fracture
            The fracture object that the impermeable ellipse is associated with.
        """
        self.label = label
        self.focis = focis
        self.nu = nu
        self.ncoef = ncoef
        self.nint = nint
        self.frac = frac

        # Create the pre-calculation variables
        self.coef = np.zeros(ncoef, dtype=complex)
        self.error = 1

    def __str__(self):
        return f"Impermeable ellipse: {self.label}"


class ImpermeableCircle(Element):
    def __init__(self, label, radius, center, frac0, ncoef=5, nint=10, **kwargs):
        """
        Initializes the impermeable circle class.

        Parameters
        ----------
        label : str or int
            The label of the impermeable circle.
        radius : float
            The radius of the impermeable circle.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac0 : Fracture
            The fracture object that the impermeable circle is associated with.
        """
        super().__init__(label, _id=-1, _type=4, frac0=frac0)
        self.label = label
        self.radius = radius
        self.center = center
        self.ncoef = ncoef
        self.nint = nint

        # Create the pre-calculation variables
        self.coef = np.zeros(ncoef, dtype=complex)
        self.error = 1

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    def __str__(self):
        return f"Impermeable circle: {self.label}"

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
        chi = np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False)) * (
            1 + offset / self.radius
        )
        return gf.map_chi_to_z_circle(chi, self.radius, self.center)

    def calc_omega(self, z):
        """
        Calculate the complex potential for the impermeable circle.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the complex potential at

        Returns
        -------
        omega : np.ndarray
            The complex potential
        """
        # Map the z point to the chi plane
        chi = gf.map_z_circle_to_chi(z, self.radius, self.center)
        # Calculate omega
        if isinstance(chi, np.complex128):
            if np.abs(chi) < 1.0 - 1e-10:
                return np.nan + 1j * np.nan
            omega = mf.asym_expansion(chi, self.coef)
        else:
            omega = mf.asym_expansion(chi, self.coef)
            omega[np.abs(chi) < 1.0 - 1e-10] = np.nan + np.nan * 1j
        return omega

    def calc_w(self, z):
        """
        Calculate the complex discharge vector for the impermeable circle.

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
        chi = gf.map_z_circle_to_chi(z, self.radius, self.center)
        # Calculate w
        if isinstance(chi, np.complex128):
            if np.abs(chi) < 1.0 - 1e-10:
                return np.nan + 1j * np.nan
            w = -mf.asym_expansion_d1(chi, self.coef)
        else:
            w = -mf.asym_expansion_d1(chi, self.coef)
            w[np.abs(chi) < 1.0 - 1e-10] = np.nan + 1j * np.nan
        w /= self.radius
        return w


class ImpermeableLine(Element):
    def __init__(self, label, endpoints0, frac0, ncoef=5, nint=10, **kwargs):
        super().__init__(label, _id=-1, _type=5, frac0=frac0)
        self.label = label
        self.endpoints0 = endpoints0
        self.ncoef = ncoef
        self.nint = nint

        # Create the pre-calculation variables
        self.thetas = np.linspace(
            start=np.pi / (2 * nint),
            stop=np.pi + np.pi / (2 * nint),
            num=nint,
            endpoint=False,
        )
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)
        self.error = 1

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    def __str__(self):
        return f"Impermeable line: {self.label}"

    def calc_omega(self, z):
        """
        Calculate the complex potential for the impermeable circle.

        Parameters
        ----------
        z : np.ndarray
            The points to calculate the complex potential at

        Returns
        -------
        omega : np.ndarray
            The complex potential
        """
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints0)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef)
        return omega

    def calc_w(self, z):
        """
        Calculate the complex discharge vector for the impermeable circle.

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
        w = -mf.asym_expansion_d1(chi, self.coef)
        w *= 2 * chi**2 / (chi**2 - 1) * 2 / (self.endpoints0[1] - self.endpoints0[0])
        return w
