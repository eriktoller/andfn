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
def discharge_term(self_, z):
    """
    Calculate the discharge term for the constant head line.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The constant head line element
    z : np.ndarray
        The points to calculate the discharge term at

    Returns
    -------
    float
        The discharge term
    """
    phi = 0.0
    center = (self_["endpoints0"][0] + self_["endpoints0"][1]) / 2.0
    half_vec = (self_["endpoints0"][1] - self_["endpoints0"][0]) / 2.0
    mirror_center = (1.0 + 1.0 - np.abs(center)) * center / np.abs(center)
    mirror_endpoints = np.array([mirror_center - half_vec, mirror_center + half_vec])
    for z0 in z:
        chi = gf.map_z_line_to_chi(z0, self_["endpoints0"])
        phi += np.real(mf.well_chi(chi, 1.0))
        chi_mirror = gf.map_z_line_to_chi(z0, mirror_endpoints)
        phi += np.real(mf.well_chi(chi_mirror, 1.0))
    return phi / len(z)


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
    self_["error_old2"] = self_["error_old"]
    self_["error_old"] = self_["error"]
    self_["error"] = mf.calc_error(
        work_array["coef"][: self_["ncoef"]], work_array["old_coef"][: self_["ncoef"]]
    )
    self_["error_coef"] = mf.calc_coef_error(
        work_array["coef"][: self_["ncoef"]], work_array["old_coef"][: self_["ncoef"]]
    )


@nb.njit(inline="always")
def calc_omega(self_, z, mirror=False):
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
    center = (self_["endpoints0"][0] + self_["endpoints0"][1]) / 2.0
    half_vec = (self_["endpoints0"][1] - self_["endpoints0"][0]) / 2.0
    mirror_center = (1.0 + 1.0 - np.abs(center)) * center / np.abs(center)
    mirror_endpoints = np.array([mirror_center - half_vec, mirror_center + half_vec])
    chi_mirror = gf.map_z_line_to_chi(z, mirror_endpoints)
    if mirror:
        return mf.well_chi(chi_mirror, self_["q"])
    chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
    omega = mf.well_chi(chi, self_["q"])
    omega += mf.asym_expansion(chi, self_["coef"][: self_["ncoef"]])
    # Add the mirror the endpoints by mirroring the center point over the unit circle  and then compute the new endpoints
    center = (self_["endpoints0"][0] + self_["endpoints0"][1]) / 2.0
    half_vec = (self_["endpoints0"][1] - self_["endpoints0"][0]) / 2.0
    mirror_center = (1.0 + 1.0 - np.abs(center)) * center / np.abs(center)
    mirror_endpoints = np.array([mirror_center - half_vec, mirror_center + half_vec])
    chi_mirror = gf.map_z_line_to_chi(z, mirror_endpoints)
    omega += mf.well_chi(chi_mirror, self_["q"])
    # omega += mf.asym_expansion(chi_mirror, self_["coef"][: self_["ncoef"]])
    # plot the endpoints and the z point in the chi plane for debugging
    # import matplotlib.pyplot as plt
    # plt.plot(self_["endpoints0"].real, self_["endpoints0"].imag, color="red", label="chi")
    # plt.plot(mirror_endpoints.real, mirror_endpoints.imag, color="blue", label="chi_mirror")
    # plt.gca().add_patch(plt.Circle((0, 0), 1, color="black", fill=False, label="chi point"))
    # plt.legend()
    # plt.axis("equal")
    # plt.show()
    return omega


@nb.njit()
def calc_omega_array(self_, omega, z):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The intersection element
    omega : np.ndarray[np.complex128]
        An array to store the resulting value for the omega function
    z : np.ndarray[np.complex128]
        An array of points in the complex z-plane

    Return
    ------
    None
            Edits the omega array in place with the resulting value for the omega function
    """
    chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
    mf.well_chi_array(omega, chi, self_["q"])
    mf.asym_expansion_array(omega, chi, self_["coef"][: self_["ncoef"]])


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
