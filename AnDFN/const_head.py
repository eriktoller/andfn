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
        self.thetas = np.linspace(start=np.pi/(2*nint), stop=np.pi + np.pi/(2*nint), num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.phi = frac.phi_from_head(head)
        self.error = 1

    def __str__(self):
        return f'Constant head line: {self.label}'

    def increase_coef(self, n):
        """
        Increase the number of coefficients in the asymptotic expansion.
        """
        self.ncoef += n
        self.coef = np.append(self.coef, np.zeros(n, dtype=complex))
        self.nint += n*2
        self.thetas = np.linspace(start=np.pi/(2*self.nint), stop=np.pi + np.pi/(2*self.nint), num=self.nint, endpoint=False)

    def discharge_term(self, z):
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        return np.sum(np.real(mf.well_chi(chi, 1))) / len(z)

    def z_array(self, n):
        return np.linspace(self.endpoints[0], self.endpoints[1], n)

    def z_array_tracking(self, n, offset=1e-3):
        chi = np.exp(1j * np.linspace(0, 2*np.pi, n, endpoint=False))*(1+offset)
        return gf.map_chi_to_z_line(chi, self.endpoints)


    def calc_omega(self, z):
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef) + mf.well_chi(chi, self.q)
        return omega

    def calc_w(self, z):
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Calculate w
        w = -mf.asym_expansion_d1(chi, self.coef) - self.q / (2 * np.pi * chi)
        w *= 2 * chi ** 2 / (chi ** 2 - 1) * 2 / (self.endpoints[1] - self.endpoints[0])
        return w

    def solve(self):
        s = mf.cauchy_integral_real(self.nint, self.ncoef, self.thetas,
                                    lambda z: self.frac.calc_omega(z, exclude=self),
                                    lambda chi: gf.map_chi_to_z_line(chi, self.endpoints))

        s = np.real(s)
        s[0] = 0  # Set the first coefficient to zero (constant embedded in discharge matrix)
        self.error = np.max(np.abs(s + self.coef))
        self.coef = -s

    def check_boundary_condition(self, n=10):
        chi = np.exp(1j * np.linspace(0, np.pi, n))
        # Calculate the head in fracture 0
        z0 = gf.map_chi_to_z_line(chi, self.endpoints)
        omega0 = self.frac.calc_omega(z0, exclude=None)

        return np.mean(np.abs(self.phi - np.real(omega0))) / np.abs(self.phi)

    def check_chi_crossing(self, z0, z1, atol=1e-10):
        z = gf.line_line_intersection(z0, z1, self.endpoints[0], self.endpoints[1])

        if z is None:
            return False

        if (np.abs(z - z0) + np.abs(z - z1) > np.abs(z0 - z1)):
            return False

        if (np.abs(z - self.endpoints[0]) + np.abs(z - self.endpoints[1]) > np.abs(self.endpoints[0] - self.endpoints[1])):
            return False

        return z
