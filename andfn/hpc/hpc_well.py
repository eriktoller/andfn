"""
Notes
-----
This module contains the HPC well functions.

"""

import numpy as np
import numba as nb
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf


@nb.njit(inline="always")
def calc_omega(self_, z):
    """
    Calculates the omega for the well. If z is inside the well, the omega is set to nan + nan*1j.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The well element.
    z : complex
        A point in the complex z plane.


    Returns
    -------
    omega : complex
        The complex potential for the well.
    """
    chi = gf.map_z_circle_to_chi(z, self_["radius"], self_["center"])
    omega = mf.well_chi(chi, self_["q"])
    if np.abs(chi) < 1 - 1e-12:
        omega = np.nan + np.nan * 1j
    return omega


def calc_w(self_, z):
    """
    Calculates the omega for the well. If z is inside the well, the omega is set to nan + nan*1j.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The well element.
    z : complex
        A point in the complex z plane.


    Returns
    -------
    omega : complex
        The complex potential for the well.
    """
    chi = gf.map_z_circle_to_chi(z, self_["radius"], self_["center"])
    w = -self_["q"] / (2 * np.pi * chi) / self_["radius"]
    return w


@nb.njit()
def z_array(self_, n):
    """
    Returns an array of n points on the well.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The well element
    n : int
        The number of points to return.

    Returns
    -------
    z : np.ndarray[complex]
        An array of n points on the well.
    """
    theta = np.linspace(0.0, 2.0 * np.pi - 2 * np.pi / n, n)
    z = self_["radius"] * np.exp(1j * theta) + self_["center"]
    return z
