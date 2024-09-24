"""
Notes
-----
This module contains the constant head classes.
"""
import AnDFN.math_functions as mf
import AnDFN.geometry_functions as gf
import numpy as np


class ConstantHeadLine:
    def __init__(self, label, endpoints, ncoef, nint, q, frac):
        self.label = label
        self.endpoints = endpoints
        self.ncoef = ncoef
        self.nint = nint
        self.q = q
        self.frac = frac

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=np.pi, num=nint)
        self.coef = np.zeros(ncoef, dtype=complex)

    def __str__(self):
        return f'Constant head line: {self.label}'

    def calc_omega(self, z):
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef, offset=0) + mf.well_chi(chi, self.q)
        return omega

    def solve(self):
        s = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                lambda z: self.frac.calc_omega(z, exclude=self.label),
                                lambda chi: gf.map_chi_to_z_line(chi, self.endpoints))

        self.coef = -s
        return None