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
        self.label = label
        self.r = r
        self.zw = zw
        self.q = q
        self.frac_label = frac_label

    def calc_omega(self, z):
        chi = gf.map_z_circle_to_chi(z, self.r, self.zw)
        if (chi * np.conj(chi)) < 1.0:
            omega = np.nan + 1j * np.nan
        else:
            omega = mf.well_chi(chi, self.q)
        return omega