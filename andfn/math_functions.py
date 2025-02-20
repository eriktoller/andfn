"""
Notes
-----
This module contains some general mathematical functions.
"""

import numpy as np
import numba as nb

def asym_expansion(chi, coef):
    """
    Function that calculates the asymptotic expansion starting from 0 for a given point chi and an array of
    coefficients.

    .. math::
        f(\chi) = \sum_{n=0}^{\infty} c_n \chi^{-n}

    Parameters
    ----------
    chi : complex | np.ndarray
        A point in the complex chi-plane
    coef : np.ndarray
        An array of coefficients

    Return
    ------
    res : complex | np.ndarray
        The resulting value for the asymptotic expansion
    """
    #res = 0.0
    #for n, c in enumerate(coef):
    #    res += c * chi ** (-n)
    res = np.sum(coef * chi[:, np.newaxis] ** (-np.arange(len(coef))), axis=1)

    return res


def asym_expansion_d1(chi, coef):
    """
    Function that calculates the first derivative of the asymptotic expansion starting from 0 for a given point chi and
    an array of coefficients.

    .. math::
        f(\chi) = -\sum_{n=0}^{\infty} n c_n \chi^{-n-1}

    Parameters
    ----------
    chi : complex | np.ndarray
        A point in the complex chi-plane
    coef : np.ndarray
        An array of coefficients

    Return
    ------
    res : complex | np.ndarray
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

    .. math::
        f(\chi) = \sum_{n=0}^{\infty} c_n \chi^{n}

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
    #res = 0.0 + 0.0j
    #for n, c in enumerate(coef):
    #    res += c * chi ** n
    res = np.sum(coef * chi[:, np.newaxis] ** np.arange(len(coef)), axis=1)

    return res


def taylor_series_d1(chi, coef):
    """
    Function that calculates the first derivative of the Taylor series starting from 0 for a given point chi and an array of
    coefficients.

    .. math::
        f(\chi) = \sum_{n=1}^{\infty} n c_n \chi^{n-1}

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

    .. math::
        \omega = \frac{q}{2\pi} \log(\chi)

    Parameters
    ----------
    chi : np.ndarray
        A point in the complex chi plane
    q : float
        The discharge eof the well.

    Returns
    -------
    omega : np.ndarray
        The complex discharge potential
    """
    return q / (2 * np.pi) * np.log(chi)


def cauchy_integral_real(n, m, thetas, omega_func, z_func):
    """
    Function that calculates the Cauchy integral with the discharge potential for a given array of thetas.

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
    #integral = np.zeros((n, m), dtype=complex)

    #chi = np.exp(1j * thetas)
    #z = z_func(chi)
    #phi = np.real(omega_func(z))
    #integral[:, :m] = phi[:, np.newaxis] * np.exp(-1j * np.arange(m) * thetas[:, np.newaxis])
    integral = np.exp(-1j * np.arange(m) * thetas[:, np.newaxis]) * np.real(omega_func(z_func(np.exp(1j * thetas))))[:,
                                                                    np.newaxis]

    coef = 2 * np.sum(integral, axis=0) / n
    coef[0] = coef[0] / 2

    return coef


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

    chi = np.exp(1j * thetas)
    z = z_func(chi)
    psi = np.imag(omega_func(z))
    integral[:, :m] = psi[:, np.newaxis] * np.exp(-1j * np.arange(m) * thetas[:, np.newaxis])

    coef = 2j * np.sum(integral, axis=0) / n
    coef[0] = coef[0] / 2

    return coef


def cauchy_integral_domega(n, m, thetas, dpsi_corr, omega_func, z_func):
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
    dpsi_corr : np.ndarray
        Correction for the stream function
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

    chi = np.exp(1j * thetas)
    z = z_func(chi)
    psi = np.imag(omega_func(z))
    dpsi = np.diff(psi)
    dpsi = np.hstack([0, np.add(dpsi, -dpsi_corr)])

    psi1 = np.cumsum(dpsi) + psi[0]
    integral[:, :m] = psi1[:, np.newaxis] * np.exp(-1j * np.arange(m) * thetas[:, np.newaxis])

    coef = 2j * np.sum(integral, axis=0) / n
    coef[0] = coef[0] / 2

    return coef
