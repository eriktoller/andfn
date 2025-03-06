"""
Notes
-----
This module contains some general mathematical functions.
"""

import numpy as np
import numba as nb
import math

from docutils.nodes import inline

from . import NO_PYTHON
from . import hpc_fracture
from . import hpc_geometry_functions as gf

@nb.jit(nopython=NO_PYTHON, inline='always')
def asym_expansion(chi, coef):
    """
    Function that calculates the asymptotic expansion starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : np.ndarray[np.complex128]
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = chi*0
    for n in range(len(coef)):
        res += coef[n] * chi ** -n

    return res


def asym_expansion_d1(chi, coef):
    """
    Function that calculates the asymptotic expansion starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : np.ndarray
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0 + 0.0j
    for n, c in enumerate(coef):
        res -= c * n * np.pow(chi,(-n - 1))

    return res

@nb.jit(nopython=NO_PYTHON, inline='always')
def taylor_series(chi, coef):
    """
    Function that calculates the Taylor series starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : np.ndarray
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = chi*0
    for n, c in enumerate(coef):
        res += c * chi ** n

    return res


def taylor_series_d1(chi, coef):
    """
    Function that calculates the Taylor series starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : np.ndarray
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0 + 0.0j
    for n, c in enumerate(coef[1:], start=1):
        res += c * n * chi ** (n-1)

    return res

@nb.jit(nopython=NO_PYTHON, inline='always')
def well_chi(chi, q):
    """
    Function that return the complex potential for a well as a function of chi.

    Parameters
    ----------
    chi : complex
        A point in the complex chi plane
    q : np.float64
        The discharge eof the well.

    Returns
    -------
    omega : complex
        The complex discharge potential
    """
    return q / (2 * np.pi) * np.log(chi)

@nb.jit(nopython=NO_PYTHON)
def cauchy_integral_real(n, m, thetas, frac0, element_id_, element_struc_array, endpoints0, work_array, coef):
    """
    FUnction that calculates the Cauchy integral with the discharge potential for a given array of thetas.

    Parameters
    ----------
    n : int
        Number of integration points
    m : int
        Number of coefficients
    thetas : np.ndarray
        Array with thetas along the unit circle
    frac0 : np.ndarray
        The fracture
    element_id_ : int
        The element id
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    endpoints0 : np.ndarray[np.complex128]
        The endpoints of the constant head line

    Return
    ------
    coef : np.ndarray[np.complex128]
        Array of coefficients
    """
    # set integral to zero
    work_array['integral'][:] = 0.0

    for ii in range(n):
        chi = np.exp(1j * thetas[ii])
        z = gf.map_chi_to_z_line(chi, endpoints0)
        omega = hpc_fracture.calc_omega(frac0, z, element_struc_array, element_id_)
        work_array['phi'][ii] = np.real(omega)
    for jj in range(m):
        for ii in range(n):
            work_array['integral'][jj] += work_array['phi'][ii] * np.exp(-1j * jj * thetas[ii])


    for ii in range(m):
        coef[ii] = 2 * work_array['integral'][ii] / n
    coef[0] = coef[0] / 2


def cauchy_integral_imag(n, m, thetas, omega_func, z_func):
    """
    FUnction that calculates the Cauchy integral with the stream function for a given array of thetas.

    Parameters
    ----------
    n : int
        Number of integration points
    m : int
        Number of coefficients
    thetas : np.ndarray
        Array with thetas along the unit circle
    omega_func : function
        The function for the complex potential
    z_func : function
        The function for the mapping of chi to z

    Return
    ------
    coef : np.ndarray
        Array of coefficients
    """
    integral = np.zeros((n, m), dtype=complex)
    coef = np.zeros(m, dtype=complex)

    for ii in range(n):
        chi = np.exp(1j * thetas[ii])
        z = z_func(chi)
        psi = np.imag(omega_func(z))
        for jj in range(m):
            integral[ii, jj] = psi * np.exp(-1j * jj * thetas[ii])

    for ii in range(m):
        coef[ii] = 2 * sum(integral[:, ii]) / n
    coef[0] = coef[0] / 2

    return coef


@nb.jit(nopython=NO_PYTHON)
def cauchy_integral_domega(n, m, thetas, dpsi_corr, frac0, element_id_, element_struc_array, radius, work_array, coef):
    """
    FUnction that calculates the Cauchy integral with the stream function for a given array of thetas.

    Parameters
    ----------
    n : np.int64
        Number of integration points
    m : np.int64
        Number of coefficients
    thetas : np.ndarray[np.float64]
        Array with thetas along the unit circle
    dpsi_corr : np.ndarray[np.complex128]
        Correction for the stream function
    frac0 : np.ndarray[fracture_dtype]
        The fracture
    element_id_ : np.int64
        The element id
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    radius : np.float64
        The radius of the bounding circle

    Return
    ------
    coef : np.ndarray[np.complex128]
        Array of coefficients
    """
    for ii in range(n):
        chi = np.exp(1j * thetas[ii])
        z = gf.map_chi_to_z_circle(chi, radius)
        omega = hpc_fracture.calc_omega(frac0, z, element_struc_array, element_id_)
        work_array['psi'][ii] = np.imag(omega)
    delta_psi = work_array['psi'][1:n] - work_array['psi'][:n-1]
    work_array['dpsi'][1:n] = delta_psi - dpsi_corr
    # set integral to zero
    work_array['integral'][:] = 0.0

    psi0 = work_array['psi'][0]
    for ii in range(n):
        psi1 = psi0 + work_array['dpsi'][ii]
        work_array['psi'][ii] = psi1
        psi0 = psi1
    for jj in range(m):
        for ii in range(n):
            work_array['integral'][jj] += work_array['psi'][ii] * np.exp(-1j * jj * thetas[ii])

    for ii in range(m):
        coef[ii] = 2j * work_array['integral'][ii] / n
    coef[0] = work_array['coef'][0] / 2

@nb.jit(nopython=NO_PYTHON, inline='always')
def calc_error(coef, coef_ref):
    """
    Function that calculates the error between two sets of coefficients.

    Parameters
    ----------
    coef : np.ndarray
        The coefficients to compare
    coef_ref : np.ndarray
        The reference coefficients

    Return
    ------
    error : float
        The error
    """
    error = 0.0
    max_coef = np.max(np.abs(coef))
    for i in range(len(coef)):
        error += np.abs(coef[i] - coef_ref[i])
    if max_coef == 0:
        max_coef = 1
    return (error/len(coef))

@nb.jit(nopython=NO_PYTHON, inline='always')
def calc_thetas(n, type_):
    """
    Function that calculates the thetas for the unit circle.

    Parameters
    ----------
    n : int
        The number of thetas to calculate
    type_ : int
        The element type. 0 for bounding circle, 1 for constant head line, 3 for intersection.

    Return
    ------
    thetas : np.ndarray
        The thetas
    """
    thetas = np.zeros(n, dtype=np.float64)
    # if type_ is 0 or 3
    del_theta = np.pi / n
    start = del_theta / 2
    if type_ == 1:
        del_theta = 2 * np.pi / n
        start = 0.0
    for i in range(n):
        thetas[i] = start + i * del_theta
    return thetas

########################################################################################################################
# Functions NUMBA
########################################################################################################################

@nb.jit(nopython=NO_PYTHON)
def cut_trail(f_str):
    cut = 0
    for c in f_str[::-1]:
        if c == "0":
            cut += 1
        else:
            break
    if cut == 0:
        for c in f_str[::-1]:
            if c == "9":
                cut += 1
            else:
                cut -= 1
                break
    if cut > 0:
        f_str = f_str[:-cut]
    if f_str == "":
        f_str = "0"
    return f_str


@nb.jit(nopython=NO_PYTHON)
def float2str(value):
    if math.isnan(value):
        return "nan"
    elif value == 0.0:
        return "0.0"
    elif value < 0.0:
        return "-" + float2str(-value)
    elif math.isinf(value):
        return "inf"
    else:
        max_digits = 4
        min_digits = -4
        e10 = math.floor(math.log10(value)) if value != 0.0 else 0
        if min_digits < e10 < max_digits:
            i_part = math.floor(value)
            f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits - e10])
            return i_str + "." + f_str
        else:
            m10 = value / 10.0 ** e10
            exp_str_len = 4
            i_part = math.floor(m10)
            f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits])
            e_str = str(e10)
            if e10 >= 0:
                e_str = "+" + e_str
            return i_str + "." + f_str + "e" + e_str