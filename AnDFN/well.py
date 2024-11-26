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
        head : float
            The hydraulic head of the well.
        frac : Fracture
            The label of the fracture the well is associated with.
        q : float
            The flow rate of the well.
        """
        self.label = label
        self.radius = radius
        self.center = center
        self.head = head
        self.frac = frac

        self.phi = frac.phi_from_head(head)
        self.q = 0.0
        self.error = 0.0

    def __str__(self):
        """
        Returns the string representation of the well.
        Returns
        -------
        str
            The string representation of the well.
        """
        return f'Well: {self.label}'

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
        return self.radius * np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False)) + self.center

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
                omega = self.head*self.frac.t + 0*1j
        else:
            omega = mf.well_chi(chi, self.q)
            omega[np.abs(chi) < 1.0 - 1e-10] = self.head*self.frac.t + 0*1j
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
            w = -self.q / chi
        else:
            w = -self.q / chi
            w[np.abs(chi) < 1.0 - 1e-10] = np.nan + 1j * np.nan
        return w

    @staticmethod
    def check_boundary_condition(n=10):
        """
        Checks the boundary condition of the well. This is allways zero as the well is solved in teh discharge matrix.
        """
        return 0
