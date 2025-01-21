"""
Notes
-----
This module contains the HPC Intersection functions.
"""

import numpy as np
import numba as nb
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf
from andfn.hpc import hpc_fracture, NO_PYTHON, FASTMATH


@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def z_array(self_, n, frac_is):
    if frac_is == self_['frac0']:
        return np.linspace(self_['endpoints0'][0], self_['endpoints0'][1], n)
    return np.linspace(self_['endpoints1'][0], self_['endpoints1'][1], n)


@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def solve(self_, fracture_struc_array, element_struc_array):
    frac0 = fracture_struc_array[self_['frac0']]
    frac1 = fracture_struc_array[self_['frac1']]
    s0 = mf.cauchy_integral_real(self_['nint'], self_['ncoef'], self_['thetas'][:self_['nint']],
                                 frac0, self_['id_'], element_struc_array,
                                 self_['endpoints0'])
    s1 = mf.cauchy_integral_real(self_['nint'], self_['ncoef'], self_['thetas'][:self_['nint']],
                                 frac1, self_['id_'], element_struc_array,
                                 self_['endpoints1'])

    s = np.real((frac0['t'] * s1 - frac1['t'] * s0) / (frac0['t'] + frac1['t']))
    s[0] = 0.0  # Set the first coefficient to zero (constant embedded in discharge matrix)

    error = np.max(np.abs(s - self_['coef'][:self_['ncoef']]))

    return s, error


@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def calc_omega(self_, z, frac_is_id):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    self_ : np.ndarray element_dtype
        The intersection element
    z : np.ndarray
        An array of points in the complex z-plane
    frac_is_id : int
        The fracture that the point is in

    Return
    ------
    omega : np.ndarray
        The resulting value for the omega function
    """
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == self_['frac0']:
        chi = gf.map_z_line_to_chi(z, self_['endpoints0'])
        omega = mf.asym_expansion(chi, self_['coef'][:self_['ncoef']]) + mf.well_chi(chi, self_['q'])
    else:
        chi = gf.map_z_line_to_chi(z, self_['endpoints1'])
        omega = mf.asym_expansion(chi, -self_['coef'][:self_['ncoef']]) + mf.well_chi(chi, -self_['q'])
    return omega

@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def z_array(self_, n, frac_is):
    if frac_is == self_['frac0']:
        return np.linspace(self_['endpoints0'][0], self_['endpoints0'][1], n)
    return np.linspace(self_['endpoints1'][0], self_['endpoints1'][1], n)
