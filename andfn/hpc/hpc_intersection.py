"""
Notes
-----
This module contains the HPC Intersection functions.
"""

import numpy as np
import numba as nb
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf


@nb.njit()
def z_array(self_, n, frac_is):
    if frac_is == self_["frac0"]:
        return np.linspace(self_["endpoints0"][0], self_["endpoints0"][1], n + 2)[
            1 : n + 1
        ]
    return np.linspace(self_["endpoints1"][0], self_["endpoints1"][1], n + 2)[1 : n + 1]


@nb.njit()
def discharge_term(self_, z, frac_is):
    """
    Calculate the discharge term for the intersection.

    Parameters
    ----------


    Returns
    -------
    float
        The discharge term
    """
    phi = 0.0
    sign = 1.0
    if frac_is == self_["frac0"]:
        endpoints = self_["endpoints0"]
    else:
        sign = -1.0
        endpoints = self_["endpoints1"]
    for z0 in z:
        chi = gf.map_z_line_to_chi(z0, endpoints)
        phi += np.real(mf.well_chi(chi, sign))
    return phi / len(z)


@nb.njit()
def solve(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Solves the intersection element.

    Parameters
    ----------
    self_ : np.ndarray element_dtype
        The intersection element.
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
    frac1 = fracture_struc_array[self_["frac1"]]
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
        work_array["coef0"][: self_["ncoef"]],
    )
    mf.cauchy_integral_real(
        self_["nint"],
        self_["ncoef"],
        self_["thetas"][: self_["nint"]],
        frac1,
        self_["_id"],
        element_struc_array,
        self_["endpoints1"],
        work_array,
        work_array["coef1"][: self_["ncoef"]],
    )

    for i in range(self_["ncoef"]):
        work_array["coef"][i] = np.real(
            (frac0["t"] * work_array["coef1"][i] - frac1["t"] * work_array["coef0"][i])
            / (frac0["t"] + frac1["t"])
        )
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
def calc_omega(self_, z, frac_is_id):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The intersection element
    z : complex
        An array of points in the complex z-plane
    frac_is_id : np.int64
        The fracture that the point is in

    Return
    ------
    omega : complex
        The resulting value for the omega function
    """
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == self_["frac0"]:
        chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
        omega = mf.asym_expansion(chi, self_["coef"][: self_["ncoef"]]) + mf.well_chi(
            chi, self_["q"]
        )
    else:
        chi = gf.map_z_line_to_chi(z, self_["endpoints1"])
        omega = mf.asym_expansion(chi, -self_["coef"][: self_["ncoef"]]) + mf.well_chi(
            chi, -self_["q"]
        )
    return omega


@nb.njit()
def calc_omega_array(
    omega, z, frac_is_id, frac0, ep0a, ep0b, ep1a, ep1b, coef, ncoef, q
):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    omega : np.ndarray[np.complex128]
        An array to store the resulting omega values
    z : np.ndarray[np.complex128]
        An array of points in the complex z-plane
    frac_is_id : np.int64
        The fracture that the point is in
    frac0 : np.int64
        The id of the first fracture that the intersection is associated with
    ep0a : complex128
        The first endpoint of the first fracture that the intersection is associated with
    ep0b : complex128
        The second endpoint of the first fracture that the intersection is associated with
    ep1a : complex128
        The first endpoint of the second fracture that the intersection is associated with
    ep1b : complex128
        The second endpoint of the second fracture that the intersection is associated with
    coef : np.ndarray[complex128]
        The coefficients of the asymptotic expansion for the intersection
    ncoef : np.int64
        The number of coefficients in the asymptotic expansion for the intersection
    q : float
        The discharge term for the intersection

    Return
    ------
    None
    """
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == frac0:
        chi = gf.map_z_line_to_chi_array(z, ep0a, ep0b)
        mf.asym_expansion_array(omega, chi, coef, ncoef)
        mf.well_chi_array(omega, chi, q)
    else:
        chi = gf.map_z_line_to_chi_array(z, ep1a, ep1b)
        mf.asym_expansion_array(omega, chi, -coef, ncoef)
        mf.well_chi_array(omega, chi, -q)


@nb.njit()
def calc_omega_sum(z, frac_is_id, frac0, ep0a, ep0b, ep1a, ep1b, coef, ncoef, q):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    z : np.ndarray[np.complex128]
        An array of points in the complex z-plane
    frac_is_id : np.int64
        The fracture that the point is in
    frac0 : np.int64
        The id of the first fracture that the intersection is associated with
    ep0a : complex128
        The first endpoint of the first fracture that the intersection is associated with
    ep0b : complex128
        The second endpoint of the first fracture that the intersection is associated with
    ep1a : complex128
        The first endpoint of the second fracture that the intersection is associated with
    ep1b : complex128
        The second endpoint of the second fracture that the intersection is associated with
    coef : np.ndarray[complex128]
        The coefficients of the asymptotic expansion for the intersection
    ncoef : np.int64
        The number of coefficients in the asymptotic expansion for the intersection
    q : float
        The discharge term for the intersection

    Return
    ------
    None
    """
    omega = 0.0 + 0.0j
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == frac0:
        chi = gf.map_z_line_to_chi_array(z, ep0a, ep0b)
        omega += mf.asym_expansion_sum(chi, coef, ncoef)
        omega += mf.well_chi_sum(chi, q)
    else:
        chi = gf.map_z_line_to_chi_array(z, ep1a, ep1b)
        omega += mf.asym_expansion_sum(chi, -coef, ncoef)
        omega += mf.well_chi_sum(chi, -q)
    return omega


def calc_w(self_, z, frac_is_id):
    """
    Calculate the complex discharge vector for the intersection.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The intersection element
    z : complex
        An array of points in the complex z-plane
    frac_is_id : np.int64
        The fracture that the point is in

    Returns
    -------
    w : np.ndarray
        The complex discharge vector
    """
    # Se if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == self_["frac0"]:
        chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
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
    else:
        chi = gf.map_z_line_to_chi(z, self_["endpoints1"])
        w = -mf.asym_expansion_d1(chi, -self_["coef"][: self_["ncoef"]]) + self_[
            "q"
        ] / (2 * np.pi * chi)
        w *= (
            2
            * chi**2
            / (chi**2 - 1)
            * 2
            / (self_["endpoints1"][1] - self_["endpoints1"][0])
        )
    return w
