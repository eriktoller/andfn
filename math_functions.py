"""
Notes
-----
This module contains some general mathematical functions.
"""

import numpy as np
import geometry_functions as gf


def asym_expansion(chi, coef, offset=0):
    """
    Function that calculates the asymptotic expansion starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : array_like
        An array of coefficients
    offset : int
        Offset to exponent

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0
    for n, c in enumerate(coef):
        res += c * chi ** (-n + offset)

    return res


def taylor_series(chi, coef, offset=0):
    """
    Function that calculates the Taylor series starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : array_like
        An array of coefficients
    offset : int
        Offset to exponent

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0
    for n, c in enumerate(coef):
        res += c * chi ** (n + offset)

    return res

def well_chi(chi, q):
    """
    Function that return the complex potential for a well as a function of chi.

    Parameters
    ----------
    chi : complex
        A point in the complex chi plane
    q : float
        The discharge eof the well.

    Returns
    -------
    omega : complex
        The complex discharge potential
    """
    return q/(2*np.pi)*np.log(chi)


def cauchy_integral(n, m, thetas, omega_func, z_func):
    """
    FUnction that calculates the Cauchy integral for a given array of thetas.

    Parameters
    ----------
    n : int
        Number of integration points
    m : int
        Number of coefficients
    thetas : array_like
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
        omega = omega_func(z)
        for jj in range(m):
            integral[ii, jj] = omega * np.exp(-1j * jj * thetas[ii])

    for ii in range(m):
        coef[ii] = 2 * sum(integral[:, ii]) / n
    coef[0] = coef[0] / 2

    return coef