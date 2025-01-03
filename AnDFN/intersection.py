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
        self.thetas = np.linspace(start=np.pi/(2*nint), stop=np.pi + np.pi/(2*nint), num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.error = 1

    def __str__(self):
        return f'Intersection: {self.label}'

    def length(self):
        return np.abs(self.endpoints[0][0] - self.endpoints[0][1])

    def increase_coef(self, n):
        """
        Increase the number of coefficients in the asymptotic expansion.
        """
        self.ncoef += n
        self.coef = np.append(self.coef, np.zeros(n, dtype=complex))
        self.nint += n*2
        self.thetas = np.linspace(start=np.pi/(2*self.nint), stop=np.pi + np.pi/(2*self.nint), num=self.nint, endpoint=False)

    def discharge_term(self, z, frac_is):
        if frac_is == self.fracs[0]:
            chi = gf.map_z_line_to_chi(z, self.endpoints[0])
            sign = 1
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints[1])
            sign = -1
        return np.sum(np.real(mf.well_chi(chi, sign))) / len(z)

    def z_array(self, n, frac_is):
        theta = np.linspace(0, np.pi, n, endpoint=False)
        if frac_is == self.fracs[0]:
            #return gf.map_chi_to_z_line(np.exp(1j * theta), self.endpoints[0])
            return np.linspace(self.endpoints[0][0], self.endpoints[0][1], n)
        #return gf.map_chi_to_z_line(np.exp(1j * theta), self.endpoints[1])
        return np.linspace(self.endpoints[1][0], self.endpoints[1][1], n)

    def calc_omega(self, z, frac_is):

        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.fracs[0]:
            chi = gf.map_z_line_to_chi(z, self.endpoints[0])
            omega = mf.asym_expansion(chi, self.coef) + mf.well_chi(chi, self.q)
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints[1])
            omega = mf.asym_expansion(chi, -self.coef) + mf.well_chi(chi, -self.q)
        return omega

    def calc_w(self, z, frac_is):
        # Se if function is in the first or second fracture that the intersection is associated with
        if frac_is == self.fracs[0]:
            chi = gf.map_z_line_to_chi(z, self.endpoints[0])
            w = -mf.asym_expansion_d1(chi, self.coef)  - self.q / (2 * np.pi * chi)
            w *= 2 * chi ** 2 / (chi ** 2 - 1) * 2 / (self.endpoints[0][1] - self.endpoints[0][0])
        else:
            chi = gf.map_z_line_to_chi(z, self.endpoints[1])
            w = -mf.asym_expansion_d1(chi, -self.coef) + self.q / (2 * np.pi * chi)
            w *= 2 * chi ** 2 / (chi ** 2 - 1) * 2 / (self.endpoints[1][1] - self.endpoints[1][0])
        return w

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


    def check_boundary_condition(self, n=10):
        """
        Check if the intersection satisfies the boundary conditions.
        """
        chi = np.exp(1j * np.linspace(0, np.pi, n))
        # Calculate the head in fracture 0
        z0 = gf.map_chi_to_z_line(chi, self.endpoints[0])
        omega0 = self.fracs[0].calc_omega(z0, exclude=None)
        head0 = np.real(omega0)/self.fracs[0].t
        # Calculate the head in fracture 1
        z1 = gf.map_chi_to_z_line(chi, self.endpoints[1])
        omega1 = self.fracs[1].calc_omega(z1, exclude=None)
        head1 = np.real(omega1)/self.fracs[1].t
        dhead = np.abs(head0 - head1)

        # Calculate the difference in head in the intersection
        #return np.mean(np.abs(head0 - head1)) / np.abs(np.mean(head0))

        return (np.max(dhead) - np.min(dhead)) / np.abs(np.mean(head0))

    def check_chi_crossing(self, z0, z1, frac, atol=1e-10):
        if frac == self.fracs[0]:
            endpoints0 = self.endpoints[0]
        else:
            endpoints0 = self.endpoints[1]

        z = gf.line_line_intersection(z0, z1, endpoints0[0], endpoints0[1])

        if z is None:
            return False

        if (np.abs(z - z0) + np.abs(z - z1) > np.abs(z0 - z1)):
            return False

        if (np.abs(z - endpoints0[0]) + np.abs(z - endpoints0[1]) > np.abs(endpoints0[0] - endpoints0[1])):
            return False

        return z
