"""
Notes
-----
This module contains the intersection class.
"""
import AnDFN.math_functions as mf
import AnDFN.geometry_functions as gf
import numpy as np


class Intersection:
    def __init__(self, label, endpoints, ncoef, nint, q, frac0, frac1):
        self.label = label
        self.endpoints = endpoints
        self.ncoef = ncoef
        self.nint = nint
        self.q = q
        self.fracs = [frac0, frac1]

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=np.pi, num=nint)
        self.coef = np.zeros(ncoef, dtype=complex)


    def __str__(self):
        return f'Intersection: {self.label}'

    def calc_omega(self, z, frac_is):
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.fracs[0]:
            omega = mf.asym_expansion(chi, self.coef, offset=0) + mf.well_chi(chi, self.q)
        else:
            omega = mf.asym_expansion(chi, -self.coef, offset=0) + mf.well_chi(chi, -self.q)
        return omega

    def solve(self):

        s0 = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                lambda z: self.fracs[0].calc_omega(z, exclude=self.label),
                                lambda chi: gf.map_chi_to_z_line(chi, self.endpoints))
        s1 = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                lambda z: self.fracs[0].calc_omega(z, exclude=self.label),
                                lambda chi: gf.map_chi_to_z_line(chi, self.endpoints))

        s = np.real((self.fracs[0].t * s1 - self.fracs[1].t * s0) / (self.fracs[0].t + self.fracs[1].t))
        self.coef = s
        return None
