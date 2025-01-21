"""
Notes
-----
This module contains the bounding classes.
"""


import AnDFN.math_functions as mf
import AnDFN.geometry_functions as gf
import numpy as np
from AnDFN.well import Well
from AnDFN.intersection import Intersection
from AnDFN.const_head import ConstantHeadLine
from .element import Element


class BoundingCircle(Element):
    def __init__(self, label, radius, frac0, ncoef=5, nint=10, **kwargs):
        """
        Initializes the bounding circle class.
        Parameters
        ----------
        label : str or int
            The label of the bounding circle.
        r : float
            The radius of the bounding circle.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac : Fracture
            The fracture object that the bounding circle is associated with.
        """
        super().__init__(label, id_=0, type_=1)
        self.label = label
        self.radius = radius
        self.ncoef = ncoef
        self.nint = nint
        self.frac0 = frac0

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_chi(self, z):
        chi = gf.map_z_circle_to_chi(z, self.radius)
        if isinstance(chi, np.ndarray) and len(chi) > 1:
            chi[np.abs(chi) > 1.0 + 1e-10] = np.nan + 1j * np.nan
        else:
            if np.abs(chi) > 1.0 + 1e-10:
                chi = np.nan + 1j * np.nan
        return chi

    def calc_omega(self, z):
        """
        Calculates the omega for the bounding circle.
        Parameters
        ----------
        z : complex | np.ndarray
            A point in the complex z plane.

        Returns
        -------
        omega : complex
            The complex potential for the bounding circle.
        """
        chi = self.get_chi(z)
        omega = mf.taylor_series(chi, self.coef)
        return omega

    def calc_w(self, z):
        """
        Calculates the complex discharge vector for the bounding circle.
        Parameters
        ----------
        z : complex
            A point in the complex z plane.

        Returns
        -------
        w : complex
            The complex discharge vector for the bounding circle.
        """
        chi = self.get_chi(z)
        w = -mf.taylor_series_d1(chi, self.coef)
        w /= self.radius
        return w

    def find_branch_cuts(self):
        """
        Find the branch cuts for the fracture.
        """
        # Find the branch cuts
        z_pos = gf.map_chi_to_z_circle(np.exp(1j * self.thetas), self.radius)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)

        for ii in range(self.nint - 1):
            for e in self.frac0.elements:
                if isinstance(e, Well):
                    # Find the branch cut for the well
                    chi0 = gf.map_z_circle_to_chi(z_pos[ii], e.radius, e.center)
                    chi1 = gf.map_z_circle_to_chi(z_pos[ii + 1], e.radius, e.center)
                    if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                        self.dpsi_corr[ii] -= e.q
                elif isinstance(e, ConstantHeadLine):
                    # Find the branch cut for the constant head line
                    chi0 = gf.map_z_line_to_chi(z_pos[ii], e.endpoints0)
                    chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e.endpoints0)
                    if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                        self.dpsi_corr[ii] -= e.q
                elif isinstance(e, Intersection):
                    # Find the branch cut for the intersection
                    if e.frac0 == self.frac0:
                        chi0 = gf.map_z_line_to_chi(z_pos[ii], e.endpoints0)
                        chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e.endpoints0)
                        ln0 = np.imag(np.log(chi0))
                        ln1 = np.imag(np.log(chi1))
                        if np.sign(ln0) != np.sign(ln1) and np.abs(ln0) + np.abs(ln1) > np.pi:
                            self.dpsi_corr[ii] -= e.q
                    else:
                        chi0 = gf.map_z_line_to_chi(z_pos[ii], e.endpoints1)
                        chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e.endpoints1)
                        ln0 = np.imag(np.log(chi0))
                        ln1 = np.imag(np.log(chi1))
                        if np.sign(ln0) != np.sign(ln1) and np.abs(ln0) + np.abs(ln1) > np.pi:
                            self.dpsi_corr[ii] += e.q

    def solve(self):
        self.find_branch_cuts()
        s = mf.cauchy_integral_domega(self.nint, self.ncoef, self.thetas, self.dpsi_corr,
                                      lambda z: self.frac0.calc_omega(z, exclude=self),
                                      lambda chi: gf.map_chi_to_z_circle(chi, self.radius))

        self.error = np.max(np.abs(s + self.coef))
        self.coef = -s

    def check_boundary_condition(self, n=10):

        # Calculate the stream function on the boundary of the fracture
        theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        z0 = gf.map_chi_to_z_circle(np.exp(1j * theta), self.radius)
        omega0 = self.frac0.calc_omega(z0, exclude=None)
        psi = np.imag(omega0)
        dpsi = np.diff(psi)
        q = np.sum(np.abs(self.dpsi_corr))
        mean_dpsi = np.abs(np.max(dpsi) - np.min(dpsi))
        if mean_dpsi > q/2:
            mean_dpsi = np.abs(np.abs(np.max(dpsi) - np.min(dpsi)) - q)
        if q < 1e-10:
            return np.abs(np.max(psi) - np.min(psi))
        return mean_dpsi / q