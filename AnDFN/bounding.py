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


class BoundingCircle:
    def __init__(self, label, r, ncoef, nint, frac):
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
        self.label = label
        self.r = r
        self.ncoef = ncoef
        self.nint = nint
        self.frac = frac

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)
        self.error = 1

    def calc_omega(self, z):
        """
        Calculates the omega for the bounding circle.
        Parameters
        ----------
        z : complex
            A point in the complex z plane.

        Returns
        -------
        omega : complex
            The complex potential for the bounding circle.
        """
        chi = gf.map_z_circle_to_chi(z, self.r)
        if isinstance(chi, np.ndarray) and len(chi) > 1:
            chi[np.abs(chi) > 1.0+1e-10] = np.nan + 1j * np.nan
        else:
            if np.abs(chi) > 1.0+1e-10:
                chi = np.nan + 1j * np.nan
        omega = mf.taylor_series(chi, self.coef)
        return omega

    def find_branch_cuts(self):
        """
        Find the branch cuts for the fracture.
        """
        # Find the branch cuts
        z_pos = gf.map_chi_to_z_circle(np.exp(1j * self.thetas), self.r)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)

        for ii in range(self.nint - 1):
            for e in self.frac.elements:
                if isinstance(e, Well):
                    # Find the branch cut for the well
                    chi0 = gf.map_z_circle_to_chi(z_pos[ii], e.radius, e.center)
                    chi1 = gf.map_z_circle_to_chi(z_pos[ii + 1], e.radius, e.center)
                    if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                        self.dpsi_corr[ii] -= e.q
                elif isinstance(e, ConstantHeadLine):
                    # Find the branch cut for the constant head line
                    chi0 = gf.map_z_line_to_chi(z_pos[ii], e.endpoints)
                    chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e.endpoints)
                    if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                        self.dpsi_corr[ii] -= e.q
                elif isinstance(e, Intersection):
                    # Find the branch cut for the intersection
                    if e.fracs[0] == self.frac:
                        chi0 = gf.map_z_line_to_chi(z_pos[ii], e.endpoints[0])
                        chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e.endpoints[0])
                        if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                            self.dpsi_corr[ii] -= e.q
                    else:
                        chi0 = gf.map_z_line_to_chi(z_pos[ii], e.endpoints[1])
                        chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e.endpoints[1])
                        if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                            self.dpsi_corr[ii] += e.q

    def solve(self):
        self.find_branch_cuts()
        s = mf.cauchy_integral_domega(self.nint, self.ncoef, self.thetas, self.dpsi_corr,
                                      lambda z: self.frac.calc_omega(z, exclude=self),
                                      lambda chi: gf.map_chi_to_z_circle(chi, self.r))

        self.error = np.max(np.abs(s + self.coef))
        self.coef = -s
