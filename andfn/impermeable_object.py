"""
Notes
-----
This module contains the impermeable object classes.
"""
import numpy as np
from .element import Element
from . import math_functions as mf
from . import geometry_functions as gf


class ImpermeableEllipse:
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
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)
        self.error = 1

    def __str__(self):
        return f'Impermeable ellipse: {self.label}'


class ImpermeableCircle(Element):
    def __init__(self, label, radius, center, ncoef, nint, frac0):
        """
        Initializes the impermeable circle class.
        Parameters
        ----------
        label : str or int
            The label of the impermeable circle.
        r : float
            The radius of the impermeable circle.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac : Fracture
            The fracture object that the impermeable circle is associated with.
        """
        super().__init__(label, id_=0, type_=4)
        self.label = label
        self.radius = radius
        self.center = center
        self.ncoef = ncoef
        self.nint = nint
        self.frac0 = frac0

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)
        self.error = 1

        # Assign to the fracture
        self.frac0.add_element(self)

    def __str__(self):
        return f'Impermeable circle: {self.label}'

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
        w = -mf.asym_expansion_d1(chi, self.coef)
        w /= self.radius
        return w

    def solve(self):
        """
        Solve the coefficients of the impermeable circle.
        """
        s = mf.cauchy_integral_imag(self.nint, self.ncoef, self.thetas,
                                    lambda z: self.frac0.calc_omega(z, exclude=self),
                                    lambda chi: gf.map_chi_to_z_circle(chi, self.radius, self.center))
        s[0] = 0.0 + 0.0j

        self.error = np.max(np.abs(s + self.coef))
        self.coef = -s