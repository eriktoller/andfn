"""
Notes
-----
This module contains the intersection class.
"""
from . import math_functions as mf
from . import geometry_functions as gf
import numpy as np
from .element import Element

class Intersection(Element):
    def __init__(self, label, endpoints0, endpoints1, frac0, frac1, ncoef=5, nint=10, **kwargs):
        super().__init__(label=label, id_=0, type_=0)
        self.label = label
        self.endpoints0 = endpoints0
        self.endpoints1 = endpoints1
        self.ncoef = ncoef
        self.nint = nint
        self.q = 0.0
        self.frac0 = frac0
        self.frac1 = frac1

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=np.pi/(2*nint), stop=np.pi + np.pi/(2*nint), num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def length(self):
        return np.abs(self.endpoints0[0] - self.endpoints0[1])

    def discharge_term(self, z, frac_is):
        if frac_is == self.frac0:
            chi = gf.map_z_line_to_chi(z, self.endpoints0)
            sign = 1
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints1)
            sign = -1
        return np.sum(np.real(mf.well_chi(chi, sign))) / len(z)

    def z_array(self, n, frac_is):
        if frac_is == self.frac0:
            return np.linspace(self.endpoints0[0], self.endpoints0[1], n)
        return np.linspace(self.endpoints1[0], self.endpoints1[1], n)

    def omega_along_element(self, n, frac_is):
        z = self.z_array(n, frac_is)
        omega = frac_is.calc_omega(z)
        return omega

    def calc_omega(self, z, frac_is):

        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.frac0:
            chi = gf.map_z_line_to_chi(z, self.endpoints0)
            omega = mf.asym_expansion(chi, self.coef) + mf.well_chi(chi, self.q)
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints1)
            omega = mf.asym_expansion(chi, -self.coef) + mf.well_chi(chi, -self.q)
        return omega

    def calc_w(self, z, frac_is):
        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.frac0:
            chi = gf.map_z_line_to_chi(z, self.endpoints0)
            w = -mf.asym_expansion_d1(chi, self.coef)  - self.q / (2 * np.pi * chi)
            w *= 2 * chi ** 2 / (chi ** 2 - 1) * 2 / (self.endpoints0[1] - self.endpoints0[0])
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints1)
            w = -mf.asym_expansion_d1(chi, -self.coef) + self.q / (2 * np.pi * chi)
            w *= 2 * chi ** 2 / (chi ** 2 - 1) * 2 / (self.endpoints1[1] - self.endpoints1[0])
        return w

    def solve(self):

        s0 = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                     lambda z: self.frac0.calc_omega(z, exclude=self),
                                     lambda chi: gf.map_chi_to_z_line(chi, self.endpoints0))
        s1 = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                     lambda z: self.frac1.calc_omega(z, exclude=self),
                                     lambda chi: gf.map_chi_to_z_line(chi, self.endpoints1))

        s = np.real((self.frac0.t * s1 - self.frac1.t * s0) / (self.frac0.t + self.frac1.t))
        s[0] = 0  # Set the first coefficient to zero (constant embedded in discharge matrix)

        self.error = np.max(np.abs(s - self.coef))
        self.coef = s


    def check_boundary_condition(self, n=10):
        """
        Check if the intersection satisfies the boundary conditions.
        """
        chi = np.exp(1j * np.linspace(0, np.pi, n))
        # Calculate the head in fracture 0
        z0 = gf.map_chi_to_z_line(chi, self.endpoints0)
        omega0 = self.frac0.calc_omega(z0, exclude=None)
        head0 = np.real(omega0)/self.frac0.t
        # Calculate the head in fracture 1
        z1 = gf.map_chi_to_z_line(chi, self.endpoints1)
        omega1 = self.frac1.calc_omega(z1, exclude=None)
        head1 = np.real(omega1)/self.frac1.t
        dhead = np.abs(head0 - head1)

        # Calculate the difference in head in the intersection
        #return np.mean(np.abs(head0 - head1)) / np.abs(np.mean(head0))

        return (np.max(dhead) - np.min(dhead)) / np.abs(np.mean(head0))

    def check_chi_crossing(self, z0, z1, frac, atol=1e-10):
        if frac == self.frac0:
            endpoints = self.endpoints0
        else:
            endpoints = self.endpoints1

        z = gf.line_line_intersection(z0, z1, endpoints[0], endpoints[1])

        if z is None:
            return False

        if np.abs(z - z0) + np.abs(z - z1) - np.abs(z0 - z1) > 1e-16:
            return False

        if np.abs(z - endpoints[0]) + np.abs(z - endpoints[1]) > np.abs(endpoints[0] - endpoints[1]):
            return False

        return z
