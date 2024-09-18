"""
Notes
-----
This module contains the intersection class.
"""
import AnDFN.math_functions as mf
import AnDFN.geometry_functions as gf
import numpy as np


class Intersection:
    def __init__(self, label, endpoints, ncoef, coef, nint, q, frac_label):
        self.label = label
        self.endpoints = endpoints
        self.ncoef = ncoef
        self.coef = coef
        self.nint = nint
        self.q = q
        self.frac_label = frac_label

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=np.pi, num=nint)

    def __str__(self):
        return f'Intersection: {self.label}'

    def calc_omega(self, z):
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef, offset=0) + mf.well_chi(chi, self.q)
        return omega
