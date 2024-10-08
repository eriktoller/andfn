"""
Notes
-----
This module contains the constant head classes.
"""
from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np


class ConstantHeadLine:
    def __init__(self, label, endpoints, ncoef, nint, head, frac):
        self.label = label
        self.endpoints = endpoints
        self.ncoef = ncoef
        self.nint = nint
        self.q = 0
        self.head = head
        self.frac = frac

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=np.pi / (2 * nint), stop=np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.phi = frac.phi_from_head(head)
        self.error = 1

    def __str__(self):
        return f'Constant head line: {self.label}'

    def discharge_term(self, z):
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        return np.sum(np.real(mf.well_chi(chi, 1))) / len(z)

    def z_array(self, n):
        return np.linspace(self.endpoints[0], self.endpoints[1], n)

    def calc_omega(self, z):
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef, offset=0) + mf.well_chi(chi, self.q)
        return omega

    def solve(self):
        s = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                    lambda z: self.frac.calc_omega(z, exclude=self),
                                    lambda chi: gf.map_chi_to_z_line(chi, self.endpoints))

        s = np.real(s)
        s[0] = 0  # Set the first coefficient to zero (constant embedded in discharge matrix)
        self.error = np.max(np.abs(s + self.coef))
        self.coef = -s
