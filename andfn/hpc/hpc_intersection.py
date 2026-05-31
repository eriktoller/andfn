"""
Notes
-----
This module contains the HPC Intersection functions.
"""

import numpy as np
import numba as nb
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf

R_COND = 90 / 100


@nb.njit()
def z_array(self_, n, frac_is):
    if frac_is == self_["frac0"]:
        return np.linspace(self_["endpoints0"][0], self_["endpoints0"][1], n + 2)[
            1 : n + 1
        ]
    return np.linspace(self_["endpoints1"][0], self_["endpoints1"][1], n + 2)[1 : n + 1]


@nb.njit()
def discharge_term(self_, z, frac_is, radius, mirror=False):
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
    if mirror:
        cond0 = np.abs(endpoints[0] + endpoints[1]) / 2.0 > radius * R_COND
        if cond0:
            m_endpoints = gf.mirror_endpoints(endpoints, radius)
            for z0 in z:
                chi_mirror = gf.map_z_line_to_chi(z0, m_endpoints)
                phi += np.real(mf.well_chi(chi_mirror, sign))
        return phi / len(z)
    for z0 in z:
        chi = gf.map_z_line_to_chi(z0, endpoints)
        phi += np.real(mf.well_chi(chi, sign))
    cond0 = np.abs(endpoints[0] + endpoints[1]) / 2.0 > radius * R_COND
    if cond0:
        m_endpoints = gf.mirror_endpoints(endpoints, radius)
        for z0 in z:
            chi_mirror = gf.map_z_line_to_chi(z0, m_endpoints)
            phi += np.real(mf.well_chi(chi_mirror, sign))
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


@nb.njit()
def solve_error(
    self_, fracture_struc_array, element_struc_array, error_struc_array, work_array
):
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
    mf.cauchy_integral_intersection_error(
        self_["nint"],
        self_["ncoef"],
        self_["thetas"][: self_["nint"]],
        frac0,
        frac1,
        self_["_id"],
        element_struc_array,
        error_struc_array,
        self_["endpoints0"],
        self_["endpoints1"],
        work_array,
        work_array["coef"][: self_["ncoef"]],
    )

    for i in range(self_["ncoef"]):
        work_array["coef"][i] = -np.real(work_array["coef"][i])
    work_array["coef"][0] = (
        0.0  # Set the first coefficient to zero (constant embedded in discharge matrix)
    )
    self_["error"] = mf.calc_error(
        work_array["coef"][: self_["ncoef"]], work_array["old_coef"][: self_["ncoef"]]
    )
    self_["error_coef"] = mf.calc_coef_error(
        work_array["coef"][: self_["ncoef"]], work_array["old_coef"][: self_["ncoef"]]
    )


@nb.njit(inline="always")
def calc_omega(self_, z, frac_is_id, radius, mirror=False):
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
    radius : float
        The radius of the fracture (used for the mirror term)
    mirror : bool, optional
        Whether to include the mirror term in the calculation (default is False)

    Return
    ------
    omega : complex
        The resulting value for the omega function
    """
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == self_["frac0"]:
        endpoints = self_["endpoints0"]
        sign = 1.0
    else:
        endpoints = self_["endpoints1"]
        sign = -1.0
    if mirror:
        cond0 = np.abs(endpoints[0] + endpoints[1]) / 2.0 > radius * R_COND
        if cond0:
            m_endpoints = gf.mirror_endpoints(endpoints, radius)
            chi_mirror = gf.map_z_line_to_chi(z, m_endpoints)
            return sign * mf.well_chi(chi_mirror, self_["q"])
        return 0.0 + 0.0j
    chi = gf.map_z_line_to_chi(z, endpoints)
    omega = sign * mf.asym_expansion(chi, self_["coef"][: self_["ncoef"]])
    omega += sign * mf.well_chi(chi, self_["q"])
    cond0 = np.abs(endpoints[0] + endpoints[1]) / 2.0 > radius * R_COND
    if cond0:
        m_endpoints = gf.mirror_endpoints(endpoints, radius)
        chi_mirror = gf.map_z_line_to_chi(z, m_endpoints)
        omega += sign * mf.well_chi(chi_mirror, self_["q"])
        # omega += sign * mf.asym_expansion(chi_mirror, self_["coef"][: self_["ncoef"]])
        # plot the endpoints and the z point in the chi plane for debugging
        """
        import matplotlib.pyplot as plt
        plt.plot(endpoints.real, endpoints.imag, color="red", label="chi")
        plt.plot(m_endpoints.real, m_endpoints.imag, color="blue", label="chi_mirror")
        plt.gca().add_patch(plt.Circle((0, 0), radius, color="black", fill=False, label="chi point"))
        plt.legend()
        plt.axis("equal")
        plt.show()
        """
    return omega


@nb.njit(inline="always")
def calc_omega_error(self_, z, frac_is_id):
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
        endpoints = self_["endpoints0"]
        sign = 1.0
    else:
        endpoints = self_["endpoints1"]
        sign = -1.0
    chi = gf.map_z_line_to_chi(z, endpoints)
    omega = sign * mf.asym_expansion(chi, self_["coef"][: self_["ncoef"]])
    return omega


@nb.njit()
def calc_omega_array(self_, omega, z, frac_is_id):
    """
    Function that calculates the omega function for a given point z and fracture.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The intersection element
    omega : np.ndarray[np.complex128]
        An array to store the resulting omega values
    z : np.ndarray[np.complex128]
        An array of points in the complex z-plane
    frac_is_id : np.int64
        The fracture that the point is in

    Return
    ------
    None
    """
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == self_["frac0"]:
        chi = gf.map_z_line_to_chi(z, self_["endpoints0"])
        mf.asym_expansion_array(omega, chi, self_["coef"][: self_["ncoef"]])
        mf.well_chi_array(omega, chi, self_["q"])
    else:
        chi = gf.map_z_line_to_chi(z, self_["endpoints1"])
        mf.asym_expansion_array(omega, chi, -self_["coef"][: self_["ncoef"]])
        mf.well_chi_array(omega, chi, -self_["q"])


@nb.njit()
def calc_w(self_, z, frac_is_id, radius):
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
    radius : float
        The radius of the fracture (used for the mirror term)

    Returns
    -------
    w : np.ndarray
        The complex discharge vector
    """
    # See if function is in the first or second fracture that the intersection is associated with
    if frac_is_id == self_["frac0"]:
        endpoints = self_["endpoints0"]
        sign = 1.0
    else:
        endpoints = self_["endpoints1"]
        sign = -1.0
    chi = gf.map_z_line_to_chi(z, endpoints)
    w = -sign * mf.asym_expansion_d1(chi, self_["coef"][: self_["ncoef"]])
    w -= sign * self_["q"] / (2 * np.pi * chi)
    w *= 2 * chi**2 / (chi**2 - 1) * 2 / (endpoints[1] - endpoints[0])
    # Mirror term: -d/dz[ well_chi(chi_mirror, q) ]
    cond0 = np.abs(endpoints[0] + endpoints[1]) / 2.0 > radius * R_COND
    if cond0:
        m_endpoints = gf.mirror_endpoints(endpoints, radius)
        chi_mirror = gf.map_z_line_to_chi(z, m_endpoints)
        w -= (
            sign
            * self_["q"]
            / (2 * np.pi * chi_mirror)
            * 2
            * chi_mirror**2
            / (chi_mirror**2 - 1)
            * 2
            / (endpoints[1] - endpoints[0])
        )

    return w
