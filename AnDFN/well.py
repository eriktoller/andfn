"""
Notes
-----
This module contains the well classes.
"""
import AnDFN.math_functions as mf
import AnDFN.geometry_functions as gf
import numpy as np


class Well:
    def __init__(self, label, r, zw, q, frac_label):
        """
        Initializes the well class.
        Parameters
        ----------
        label : str or int
            The label of the well.
        r : float
            The radius of the well.
        zw : complex
            The complex location of the well.
        q : float
            The flow rate of the well.
        frac_label : str or int
            The label of the fracture the well is associated with.
        """
        self.label = label
        self.r = r
        self.zw = zw
        self.q = q
        self.frac_label = frac_label

    def calc_omega(self, z):
        """
        Calculates the omega for the well.
        Parameters
        ----------
        z : complex
            A point in the complex z plane.

        Returns
        -------
        omega : complex
            The complex potential for the well.
        """
        chi = gf.map_z_circle_to_chi(z, self.r, self.zw)
        if (chi * np.conj(chi)) < 1.0:
            omega = np.nan + 1j * np.nan
        else:
            omega = mf.well_chi(chi, self.q)
        return omega
