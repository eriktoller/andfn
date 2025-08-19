"""
Notes
-----
This module contains the HPC Constant head functions.
"""

import numpy as np
import numba as nb

from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf


@nb.njit()
def solve(self_, fracture_struc_array, element_struc_array, work_array):
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
    Edits the self_ array and works_array in place.
    """
    frac0 = fracture_struc_array[self_["frac0"]]
    work_array["old_coef"][: self_["ncoef"]] = self_["coef"][: self_["ncoef"]]
    mf.cauchy_integral_real(
        self_["nint"],
        self_["ncoef"],
        self_["thetas"][: self_["nint"]],
        frac0,
        self_["_id"],
        element_struc_array,
        self_["endpoints0"],
        work_array,
        work_array["coef"][: self_["ncoef"]],
    )

    for i in range(self_["ncoef"]):
        work_array["coef"][i] = -np.real(work_array["coef"][i])
    work_array["coef"][0] = (
        0.0  # Set the first coefficient to zero (constant embedded in discharge matrix)
    )
    # self_['error'] = np.max(np.abs(work_array['coef'][:self_['ncoef']] - work_array['old_coef'][:self_['ncoef']]))
    self_["error_old"] = self_["error"]
    self_["error"] = mf.calc_error(
        work_array["coef"][: self_["ncoef"]], work_array["old_coef"][: self_["ncoef"]]
    )


@nb.njit(inline="always")
def calc_omega(self_, z):
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
    chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
    omega = mf.well_chi(chi, self_["q"])
    omega += mf.asym_expansion(chi, self_["coef"][: self_["ncoef"]])
    return omega


def calc_w(self_, z):
    """
    Calculate the complex discharge vector for the constant head line.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The constant head line element
    z : np.ndarray
        The points to calculate the complex discharge vector at

    Returns
    -------
    np.ndarray
        The complex discharge vector
    """
    # Map the z point to the chi plane
    chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
    # Calculate w
    w = -mf.asym_expansion_d1(chi, self_["coef"][: self_["ncoef"]]) - self_["q"] / (
        2 * np.pi * chi
    )
    w *= (
        2
        * chi**2
        / (chi**2 - 1)
        * 2
        / (self_["endpoints0"][1] - self_["endpoints0"][0])
    )
    return w


@nb.njit()
def z_array(self_, n):
    """
    Returns an array of n points along the constant head line.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The constant head line element
    n : int
        The number of points to return.

    Returns
    -------
    z : np.ndarray[complex]
        An array of n points on the well.
    """
    return np.linspace(self_["endpoints0"][0], self_["endpoints0"][1], n + 2)[1 : n + 1]
