"""
Notes
-----
This module contains some general mathematical functions, e.g. series expansions and Cauchy integrals.

The mathematical functions are used by the element classes in the andfn module.
"""

import numpy as np

import andfn.hpc.hpc_math_functions as hpc_mf


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
    coef : np.ndarray[np.complex128]
        An array of coefficients

    Return
    ------
    res : complex | np.ndarray
        The resulting value for the asymptotic expansion
    """
    res = []
    if np.isscalar(chi):
        return hpc_mf.asym_expansion(chi, coef)
    else:
        for c in chi:
            res.append(hpc_mf.asym_expansion(c, coef))

    return np.array(res)


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

    return hpc_mf.asym_expansion_d1(chi, coef)


def taylor_series(chi, coef):
    """
    Function that calculates the Taylor series starting from 0 for a given point chi and an array of
    coefficients.

    .. math::
        f(\chi) = \sum_{n=0}^{\infty} c_n \chi^{n}

    Parameters
    ----------
    chi : complex | np.ndarray
        A point in the complex chi-plane
    coef : np.ndarray
        An array of coefficients

    Return
    ------
    res : complex
        The resulting value for the asymptotic expansion
    """

    res = []
    if np.isscalar(chi):
        return hpc_mf.taylor_series(chi, coef)
    else:
        for c in chi:
            res.append(hpc_mf.taylor_series(c, coef))

    return np.array(res)


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

    return hpc_mf.taylor_series_d1(chi, coef)


def well_chi(chi, q):
    r"""
    Function that return the complex potential for a well as a function of chi.

    .. math::
        \omega = \frac{q}{2\pi} \log(\chi)

    Parameters
    ----------
    chi : np.ndarray | complex
        A point in the complex chi plane
    q : np.float64
        The discharge eof the well.

    Returns
    -------
    omega : np.ndarray
        The complex discharge potential
    """

    return hpc_mf.well_chi(chi, q)
