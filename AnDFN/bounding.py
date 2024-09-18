"""
Notes
-----
This module contains the bounding classes.
"""
import AnDFN.math_functions as mf
import AnDFN.geometry_functions as gf
import numpy as np


class BoundingCircle:
    def __init__(self, label, r, ncoef, coef, nint, frac_label):
        self.label = label
        self.r = r
        self.ncoef = ncoef
        self.coef = coef
        self.nint = nint
        self.frac_label = frac_label

    def calc_omega(self, z):
        chi = gf.map_z_circle_to_chi(z, self.r)
        if chi * np.conj(chi) > 1.0:
            omega = np.nan + 1j * np.nan
        else:
            omega = mf.taylor_series(chi, self.coef)
        return omega
