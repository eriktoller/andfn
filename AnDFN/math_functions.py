"""
Notes
-----
This module contains some general mathematical functions.
"""

import numpy as np


def asym_expansion(chi, coef):
    """
    Function that calculates the asymptotic expansion starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : array_like
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0
    for n, c in enumerate(coef):
        res += c * chi ** (-n)

    return res


def asym_expansion_d1(chi, coef):
    """
    Function that calculates the asymptotic expansion starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : array_like
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0
    for n, c in enumerate(coef):
        res -= c * n * chi ** (-n - 1)

    return res


def taylor_series(chi, coef):
    """
    Function that calculates the Taylor series starting from 0 for a given point chi and an array of
    coefficients.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    coef : array_like
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """
    res = 0.0 + 0.0j
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
    coef : array_like
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


def well_chi(chi, q):
    """
    Function that return the complex potential for a well as a function of chi.

    Parameters
    ----------
    chi : complex | ndarray
        A point in the complex chi plane
    q : float
        The discharge eof the well.

    Returns
    -------
    omega : complex  | ndarray
        The complex discharge potential
    """
    return q / (2 * np.pi) * np.log(chi)


def cauchy_integral_real(n, m, thetas, omega_func, z_func):
    """
    FUnction that calculates the Cauchy integral with the discharge potential for a given array of thetas.

    Parameters
    ----------
    n : int
        Number of integration points
    m : int
        Number of coefficients
    thetas : ndarray
        Array with thetas along the unit circle
    omega_func : function
        The function for the complex potential
    z_func : function
        The function for the mapping of chi to z

    Return
    ------
    coef : ndarray
        Array of coefficients
    """
    integral = np.zeros((n, m), dtype=complex)
    coef = np.zeros(m, dtype=complex)

    for ii in range(n):
        chi = np.exp(1j * thetas[ii])
        z = z_func(chi)
        phi = np.real(omega_func(z))
        for jj in range(m):
            integral[ii, jj] = phi * np.exp(-1j * jj * thetas[ii])

    for ii in range(m):
        coef[ii] = 2 * sum(integral[:, ii]) / n
    coef[0] = coef[0] / 2

    return coef


def cauchy_integral_imag(n, m, thetas, omega_func, z_func):
    """
    FUnction that calculates the Cauchy integral with the streamfunction for a given array of thetas.

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
    coef : ndarray
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


def cauchy_integral_domega(n, m, thetas, dpsi_corr, omega_func, z_func):
    """
    FUnction that calculates the Cauchy integral with the streamfunction for a given array of thetas.

    Parameters
    ----------
    n : int
        Number of integration points
    m : int
        Number of coefficients
    thetas : ndarray
        Array with thetas along the unit circle
    omega_func : function
        The function for the complex potential
    z_func : function
        The function for the mapping of chi to z

    Return
    ------
    coef : ndarray
        Array of coefficients
    """
    # TODO: enable doing this with an array (remove unnecessary overhead from calling cal_omega for each point) do
    #  this first when the program is tested and works
    integral = np.zeros((n, m), dtype=complex)
    coef = np.zeros(m, dtype=complex)

    psi = np.zeros(n)
    z = np.zeros(n, dtype=complex)
    for ii in range(n):
        chi = np.exp(1j * thetas[ii])
        z[ii] = z_func(chi)
    psi = np.imag(omega_func(z))
    dpsi = np.diff(psi)
    dpsi = np.hstack([0, np.add(dpsi, -dpsi_corr)])

    psi0 = psi[0]
    for ii in range(n):
        psi1 = psi0 + dpsi[ii]
        for jj in range(m):
            integral[ii, jj] = psi1 * np.exp(-1j * jj * thetas[ii])
        psi0 = psi1

    for ii in range(m):
        coef[ii] = 2j * sum(integral[:, ii]) / n
    coef[0] = coef[0] / 2

    return coef
