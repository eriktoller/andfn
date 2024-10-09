"""
Notes
-----
This module contains the well classes.
"""
from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np


class Well:
    def __init__(self, label, radius, center, head, frac):
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
        q : float
            The flow rate of the well.
        frac : Fracture
            The label of the fracture the well is associated with.
        """
        self.label = label
        self.radius = radius
        self.center = center
        self.q = 0.0
        self.head = head
        self.frac = frac

        self.phi = frac.phi_from_head(head)
        self.error = 0

    def __str__(self):
        return f'Well: {self.label}'

    def discharge_term(self, z):
        """
        Returns the discharge term for the well.
        """
        chi = gf.map_z_circle_to_chi(z, self.radius, self.center)
        return np.sum(np.real(mf.well_chi(chi, 1)) / len(z))

    def z_array(self, n):
        """
        Returns an array of n points on the well.
        """
        return self.radius * np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False)) + self.center

    def calc_omega(self, z):
        """
        Calculates the omega for the well.
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
            if np.abs(chi) < 1.0-1e-10:
                chi = np.nan
        else:
            chi[np.abs(chi) < 1.0-1e-10] = np.nan
        omega = mf.well_chi(chi, self.q)
        return omega

    @staticmethod
    def check_boundary_condition(n=10):
        return 0
