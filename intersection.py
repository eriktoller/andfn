"""
Notes
-----
This module contains the intersection class.
"""
import math_functions as mf
import geometry_functions as gf
import numpy as np


class Intersection:
    def __init__(self, id, endpoints, ncoef, coef, nint, q, frac_id):
        self.id = id
        self.endpoints = endpoints
        self.ncoef = ncoef
        self.coef = coef
        self.nint = nint
        self.q = q
        self.frac_id = frac_id

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=np.pi, num=nint)

    def __str__(self):
        return f'Intersection: {self.id}'

    def omega(self, z):
        # Map the z point to the chi plane
        chi = gf.map_z_line_to_chi(z, self.endpoints)
        # Calculate omega
        omega = mf.asym_expansion(chi, self.coef, offset=0)
        return omega
