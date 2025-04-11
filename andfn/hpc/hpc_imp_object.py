"""
Notes
-----
This module contains the HPC functions for impermeable objects.
"""

import numpy as np
import numba as nb
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_geometry_functions as gf

########################################################################################################################
## Impermeable Circle
########################################################################################################################

@nb.njit()
def solve_circle(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Solves the constant head line element.

    Parameters
    ----------
    self_ : np.ndarray element_dtype
        The constant head line element.
    fracture_struc_array : np.ndarray
        The array of fractures.
    element_struc_array : np.ndarray[element_dtype]
        The array of elements.
    work_array : np.ndarray[work_dtype]
        The work array.

    Returns
    -------
    None
        Edits the self_ array and works_array in place.
    """
    frac0 = fracture_struc_array[self_['frac0']]
    work_array['old_coef'][:self_['ncoef']] = self_['coef'][:self_['ncoef']]
    mf.cauchy_integral_imag_circle(self_['nint'], self_['ncoef'], self_['thetas'][:self_['nint']],
                                frac0, self_['id_'], element_struc_array,
                                self_['radius'], self_['center'], work_array, work_array['coef'][:self_['ncoef']])

    for i in range(self_['ncoef']):
        work_array['coef'][i] = np.conj(work_array['coef'][i])
    work_array['coef'][0] = 0.0 + 0.0j # Set the first coefficient to zero
    self_['error_old'] = self_['error']
    self_['error'] = mf.calc_error(work_array['coef'][:self_['ncoef']], work_array['old_coef'][:self_['ncoef']])


@nb.njit(inline='always')
def calc_omega_circle(self_, z):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The intersection element
    z : complex
        An array of points in the complex z-plane

    Return
    ------
    omega : complex
        The resulting value for the omega function
    """
    chi = gf.map_z_circle_to_chi(z, self_['radius'], self_['center'])
    if np.abs(chi) < 1.0 - 1e-10:
        return np.nan + 1j * np.nan
    omega = mf.asym_expansion(chi, self_['coef'][:self_['ncoef']])
    return omega