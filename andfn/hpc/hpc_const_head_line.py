"""
Notes
-----
This module contains the HPC Constant head functions.
"""

import numpy as np
import numba as nb

from AnDFN.hpc import hpc_fracture, NO_PYTHON, FASTMATH
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf

@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def solve(self_, fracture_struc_array, element_struc_array):
    """
    Solves the constant head line element.
    Parameters
    ----------
    self_ : np.ndarray element_dtype
        The constant head line element
    fracture_struc_array : np.ndarray
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements

    Returns
    -------
    s : np.ndarray
        The resulting coefficients for the constant head line
    error : float
        The error in the calculation
    """
    frac0 = fracture_struc_array[self_['frac0']]
    s0 = mf.cauchy_integral_real(self_['nint'], self_['ncoef'], self_['thetas'][:self_['nint']],
                                frac0, self_['id_'], element_struc_array,
                                 self_['endpoints0'])

    s = -np.real(s0)
    s[0] = 0.0  # Set the first coefficient to zero (constant embedded in discharge matrix)
    error = np.max(np.abs(s - self_['coef'][:self_['ncoef']]))

    return s, error


@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def calc_omega(self_, z):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    self_ : np.ndarray element_dtype
        The intersection element
    z : np.complex128
        An array of points in the complex z-plane

    Return
    ------
    omega : np.ndarray
        The resulting value for the omega function
    """
    chi = gf.map_z_line_to_chi(z, self_['endpoints0'])
    omega = mf.well_chi(chi, self_['q'])
    omega += mf.asym_expansion(chi, self_['coef'][:self_['ncoef']])


    return omega

@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def z_array(self_, n):
    """
    Returns an array of n points on the well.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The constant head line element
    n : int
        The number of points to return.

    Returns
    -------
    z : ndarray
        An array of n points on the well.
    """
    return np.linspace(self_['endpoints0'][0], self_['endpoints0'][1], n)