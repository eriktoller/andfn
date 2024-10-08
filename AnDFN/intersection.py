"""
Notes
-----
This module contains the intersection class.
"""
from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np


class Intersection:
    def __init__(self, label, endpoints0, endpoints1, ncoef, nint, frac0, frac1):
        self.label = label
        self.endpoints = np.array([endpoints0, endpoints1])
        self.ncoef = ncoef
        self.nint = nint
        self.q = 0
        self.fracs = [frac0, frac1]

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=np.pi/(2*nint), stop=np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.error = 1

    def __str__(self):
        return f'Intersection: {self.label}'

    def discharge_term(self, z, frac_is):
        if frac_is == self.fracs[0]:
            chi = gf.map_z_line_to_chi(z, self.endpoints[0])
            sign = 1
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints[1])
            sign = -1
        return np.sum(np.real(mf.well_chi(chi, sign))) / len(z)

    def z_array(self, n, frac_is):
        if frac_is == self.fracs[0]:
            return np.linspace(self.endpoints[0][0], self.endpoints[0][1], n)
        return np.linspace(self.endpoints[1][0], self.endpoints[1][1], n)

    def calc_omega(self, z, frac_is):

        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.fracs[0]:
            chi = gf.map_z_line_to_chi(z, self.endpoints[0])
            omega = mf.asym_expansion(chi, self.coef, offset=0) + mf.well_chi(chi, self.q)
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints[1])
            omega = mf.asym_expansion(chi, -self.coef, offset=0) + mf.well_chi(chi, -self.q)
        return omega

    def solve(self):

        s0 = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                     lambda z: self.fracs[0].calc_omega(z, exclude=self),
                                     lambda chi: gf.map_chi_to_z_line(chi, self.endpoints[0]))
        s1 = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                     lambda z: self.fracs[1].calc_omega(z, exclude=self),
                                     lambda chi: gf.map_chi_to_z_line(chi, self.endpoints[1]))

        s = np.real((self.fracs[0].t * s1 - self.fracs[1].t * s0) / (self.fracs[0].t + self.fracs[1].t))
        s[0] = 0  # Set the first coefficient to zero (constant embedded in discharge matrix)
        self.error = np.max(np.abs(s - self.coef))
        self.coef = s
