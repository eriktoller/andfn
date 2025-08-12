"""
Notes
-----
This module contains some general mathematical functions.
"""

import numpy as np
import numba as nb
import math

from . import hpc_fracture
from . import hpc_geometry_functions as gf
from andfn.hpc import CACHE


@nb.njit(inline="always")
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
    res = 0.0 + 0.0j
    length = len(coef)
    for n in range(length - 1):
        res += coef[length - n - 1]
        res /= chi
    res += coef[0]
    return res


@nb.njit(inline="always")
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
    length = len(coef)
    for n in range(length):
        res -= coef[length - n - 1] * (length - n - 1)
        res /= chi

    return res


@nb.njit(inline="always")
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
    res = 0.0 + 0.0j
    length = len(coef)
    for n in range(length - 1):
        res += coef[length - n - 1]
        res *= chi
    res += coef[0]

    return res


@nb.njit(inline="always")
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
    length = len(coef)
    for n in range(length - 2):
        res += coef[length - n - 1] * (length - n - 1)
        res *= chi

    res += coef[1]

    return res


@nb.njit(inline="always")
def well_chi(chi, q):
    """
    Function that return the complex potential for a well as a function of chi.

    .. math::
        \omega = \frac{q}{2 \pi} \log(\chi)

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


@nb.njit()
def cauchy_integral_real(
    n, m, thetas, frac0, element_id_, element_struc_array, endpoints0, work_array, coef
):
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
    work_array : np.ndarray[work_array_dtype]
        The work array
    coef : np.ndarray[np.complex128]
        The coefficients that will be filled

    Return
    ------
    coef : np.ndarray[np.complex128]
        Array of coefficients
    """
    # set integral to zero
    work_array["integral"][:] = 0.0

    for ii in range(n):
        # chi = np.exp(1j * thetas[ii])
        chi = work_array["exp_array_p"][ii]
        z = gf.map_chi_to_z_line(chi, endpoints0)
        omega = hpc_fracture.calc_omega(frac0, z, element_struc_array, element_id_)
        work_array["phi"][ii] = np.real(omega)
    for jj in range(m):
        res_tmp = 0.0 + 0.0j
        for ii in range(n):
            exp_val = 1.0 + 0.0j
            for _ in range(jj):
                exp_val *= work_array["exp_array_m"][ii]
            res_tmp += work_array["phi"][ii] * exp_val
        work_array["integral"][jj] = res_tmp

    for ii in range(m):
        coef[ii] = 2 * work_array["integral"][ii] / n
    coef[0] = coef[0] / 2


@nb.njit()
def find_branch_cuts(
    self_, z_pos, fracture_struc_array, element_struc_array, work_array
):
    """
    Find the branch cuts for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z_pos : np.ndarray[np.complex128]
        The positions in the complex plane
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    work_array : np.ndarray[dtype_work]
        The work array

    Returns
    -------
    dpsi_corr : np.ndarray[np.float64]
        The correction to the potential due to the branch cuts
    """
    # Find the branch cuts
    dpsi_corr = np.zeros(self_["nint"] - 1, dtype=float)

    nel = fracture_struc_array[self_["frac0"]]["nelements"]
    elements_list = fracture_struc_array[self_["frac0"]]["elements"][:nel]
    elements = element_struc_array[elements_list]
    work_array["len_discharge_element"] = 0

    cnt = 0
    for ii in range(self_["nint"] - 1):
        for e in elements:
            if e["_type"] == 0:  # Intersection
                if e["frac0"] == self_["frac0"]:
                    chi0 = gf.map_z_line_to_chi(z_pos[ii], e["endpoints0"])
                    chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e["endpoints0"])
                    ln0 = np.imag(np.log(chi0))
                    ln1 = np.imag(np.log(chi1))
                    if (
                        np.sign(ln0) != np.sign(ln1)
                        and np.abs(ln0) + np.abs(ln1) > np.pi
                    ):
                        dpsi_corr[ii] -= e["q"]
                        work_array["element_pos"][cnt] = ii
                        work_array["discharge_element"][cnt] = e["_id"]
                        get_sign(self_, work_array, cnt, chi0, chi1, -1)
                        work_array["len_discharge_element"] += 1
                        cnt += 1
                else:
                    chi0 = gf.map_z_line_to_chi(z_pos[ii], e["endpoints1"])
                    chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e["endpoints1"])
                    ln0 = np.imag(np.log(chi0))
                    ln1 = np.imag(np.log(chi1))
                    if (
                        np.sign(ln0) != np.sign(ln1)
                        and np.abs(ln0) + np.abs(ln1) > np.pi
                    ):
                        dpsi_corr[ii] += e["q"]
                        work_array["element_pos"][cnt] = ii
                        work_array["discharge_element"][cnt] = e["_id"]
                        get_sign(self_, work_array, cnt, chi0, chi1, 1)
                        work_array["len_discharge_element"] += 1
                        cnt += 1
            elif e["_type"] == 2:  # Well
                chi0 = gf.map_z_circle_to_chi(z_pos[ii], e["radius"], e["center"])
                chi1 = gf.map_z_circle_to_chi(z_pos[ii + 1], e["radius"], e["center"])
                if (
                    np.sign(np.imag(chi0)) != np.sign(np.imag(chi1))
                    and np.real(chi0) < 0
                ):
                    dpsi_corr[ii] -= e["q"]
                    work_array["element_pos"][cnt] = ii
                    work_array["discharge_element"][cnt] = e["_id"]
                    get_sign(self_, work_array, cnt, chi0, chi1, -1)
                    work_array["len_discharge_element"] += 1
                    cnt += 1
            elif e["_type"] == 3:  # Constant head line
                chi0 = gf.map_z_line_to_chi(z_pos[ii], e["endpoints0"])
                chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e["endpoints0"])
                if (
                    np.sign(np.imag(chi0)) != np.sign(np.imag(chi1))
                    and np.real(chi0) < 0
                ):
                    dpsi_corr[ii] -= e["q"]
                    work_array["element_pos"][cnt] = ii
                    work_array["discharge_element"][cnt] = e["_id"]
                    get_sign(self_, work_array, cnt, chi0, chi1, -1)
                    work_array["len_discharge_element"] += 1
                    cnt += 1


@nb.njit(inline="always")
def get_sign(self_, work_array, cnt, chi0, chi1, sign_default):
    if self_["_type"] != 1:  # If not bounding circle
        if np.imag(chi0) < np.imag(chi1):
            work_array["sign_array"][cnt] = -1 * sign_default
        else:
            work_array["sign_array"][cnt] = sign_default
    else:  # If bounding circle
        work_array["sign_array"][cnt] = sign_default


@nb.njit()
def get_dpsi_corr(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Get the correction to the stream function due to the branch cuts.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    work_array : np.ndarray[dtype_work]
        The work array

    Returns
    -------
    None
        Edits the self_ array in place.
    """
    if work_array["len_discharge_element"] == 0:
        if self_["_type"] in [1, 4]:  # If bounding circle or impermeable circle
            z_pos = gf.map_chi_to_z_circle(
                work_array["exp_array_p"][: self_["nint"]],
                self_["radius"],
                self_["center"],
            )
        elif self_["_type"] == 5:  # If impermeable line
            z_pos = gf.map_chi_to_z_line(
                work_array["exp_array_p"][: self_["nint"]], self_["endpoints0"]
            )
        find_branch_cuts(
            self_, z_pos, fracture_struc_array, element_struc_array, work_array
        )
    # set dpsi_corr to zero
    self_["dpsi_corr"][: self_["nint"] - 1] = 0.0
    for i in range(work_array["len_discharge_element"]):
        e = element_struc_array[work_array["discharge_element"][i]]
        self_["dpsi_corr"][work_array["element_pos"][i]] += (
            e["q"] * work_array["sign_array"][i]
        )


@nb.njit()
def cauchy_integral_domega_line(
    n,
    m,
    dpsi_corr,
    frac0,
    element_id_,
    element_struc_array,
    endpoints0,
    work_array,
    coef,
):
    """
    FUnction that calculates the Cauchy integral with the stream function for a given array of thetas.

    Parameters
    ----------
    n : int
        Number of integration points
    m : int
        Number of coefficients
    dpsi_corr : np.ndarray[np.complex128]
        Correction for the stream function
    frac0 : np.ndarray
        The fracture
    element_id_ : int
        The element id
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    endpoints0 : np.ndarray[np.complex128]
        The endpoints of the line
    work_array : np.ndarray[work_array_dtype]
        The work array
    coef : np.ndarray[np.complex128]
        The coefficients that will be filled

    Return
    ------
    coef : np.ndarray
        Array of coefficients
    """
    for ii in range(n):
        chi = work_array["exp_array_p"][ii]
        z = gf.map_chi_to_z_line(chi, endpoints0)
        omega = hpc_fracture.calc_omega(frac0, z, element_struc_array, element_id_)
        work_array["psi"][ii] = np.imag(omega)
    delta_psi = work_array["psi"][1:n] - work_array["psi"][: n - 1]
    work_array["dpsi"][1:n] = delta_psi - dpsi_corr
    # set integral to zero
    work_array["integral"][:] = 0.0

    psi0 = work_array["psi"][0]
    for ii in range(n):
        psi1 = psi0 + work_array["dpsi"][ii]
        work_array["psi"][ii] = psi1
        psi0 = psi1

    for jj in range(m):
        res_tmp = 0.0 + 0.0j
        for ii in range(n):
            exp_val = 1.0 + 0.0j
            for _ in range(jj):
                exp_val *= work_array["exp_array_m"][ii]
            res_tmp += work_array["psi"][ii] * exp_val
        work_array["integral"][jj] = res_tmp

    # for ii in range(n):
    #    # chi = np.exp(1j * thetas[ii])
    #    chi = work_array["exp_array_p"][ii]
    #    z = gf.map_chi_to_z_line(chi, endpoints0)
    #    omega = hpc_fracture.calc_omega(frac0, z, element_struc_array, element_id_)
    #    work_array["phi"][ii] = np.imag(omega)

    # for jj in range(m):
    #    res_tmp = 0.0 + 0.0j
    #    for ii in range(n):
    #        exp_val = 1.0 + 0.0j
    #        for _ in range(jj):
    #            exp_val *= work_array["exp_array_m"][ii]
    #        res_tmp += work_array["phi"][ii] * exp_val
    #    work_array["integral"][jj] = res_tmp

    for ii in range(m):
        coef[ii] = 2j * work_array["integral"][ii] / n
    coef[0] = coef[0] / 2


@nb.njit()
def cauchy_integral_domega(
    n,
    m,
    dpsi_corr,
    frac0,
    element_id_,
    element_struc_array,
    radius,
    center,
    work_array,
    coef,
):
    """
    FUnction that calculates the Cauchy integral with the stream function for a given array of thetas.

    Parameters
    ----------
    n : np.int64
        Number of integration points
    m : np.int64
        Number of coefficients
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
    center : np.complex128
        The center of the bounding circle
    work_array : np.ndarray[work_array_dtype]
        The work array
    coef : np.ndarray[np.complex128]
        The coefficients that will be filled

    Return
    ------
    coef : np.ndarray[np.complex128]
        Array of coefficients
    """
    for ii in range(n):
        chi = work_array["exp_array_p"][ii]
        z = gf.map_chi_to_z_circle(chi, radius, center)
        omega = hpc_fracture.calc_omega(frac0, z, element_struc_array, element_id_)
        work_array["psi"][ii] = np.imag(omega)
    delta_psi = work_array["psi"][1:n] - work_array["psi"][: n - 1]
    work_array["dpsi"][1:n] = delta_psi - dpsi_corr
    # set integral to zero
    work_array["integral"][:] = 0.0

    psi0 = work_array["psi"][0]
    for ii in range(n):
        psi1 = psi0 + work_array["dpsi"][ii]
        work_array["psi"][ii] = psi1
        psi0 = psi1

    for jj in range(m):
        res_tmp = 0.0 + 0.0j
        for ii in range(n):
            exp_val = 1.0 + 0.0j
            for _ in range(jj):
                exp_val *= work_array["exp_array_m"][ii]
            res_tmp += work_array["psi"][ii] * exp_val
        work_array["integral"][jj] = res_tmp

    for ii in range(m):
        coef[ii] = 2j * work_array["integral"][ii] / n
    coef[0] = work_array["coef"][0] / 2


@nb.njit(inline="always")
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
    if max_coef == 0:
        max_coef = 1
    for i in range(len(coef)):
        error += np.abs((coef[i] - coef_ref[i]))
    return error / len(coef)


@nb.njit()
def calc_thetas(n, _type, thetas):
    """
    Function that calculates the thetas for the unit circle.

    Parameters
    ----------
    n : int
        The number of thetas to calculate
    _type : int
        The element type. 0 for bounding circle, 1 for constant head line, 3 for intersection.
    thetas : np.ndarray
        The array to fill with the thetas

    Return
    ------
    None
        Fills the thetas array with the values of thetas
    """
    # if _type is 0 or 3
    del_theta = np.pi / n
    start = del_theta / 2
    if _type == 1:
        del_theta = 2 * np.pi / n
        start = 0.0
    for i in range(n):
        thetas[i] = start + i * del_theta


@nb.njit(cache=CACHE)
def fill_exp_array(n, thetas, exp_array, sign):
    """
    Function that fills the exp_array with the values of exp(-1j * thetas).
    Parameters
    ----------
    n : int
        The number of thetas to calculate
    thetas : np.ndarray
        The thetas
    exp_array : np.ndarray
        The array to fill with the values of exp(sign * 1j * thetas)
    sign : int
        The sign of the exponent. 1 for positive, -1 for negative.

    Returns
    -------
    None
        Fills the exp_array with the values of exp(-1j * thetas)
    """
    for ii in range(n):
        exp_array[ii] = np.exp(sign * 1j * thetas[ii])


########################################################################################################################
# Functions NUMBA
########################################################################################################################


@nb.njit()
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


@nb.njit()
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
            f_part = math.floor((1 + value % 1) * 10.0**max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1 : max_digits - e10])
            return i_str + "." + f_str
        else:
            m10 = value / 10.0**e10
            i_part = math.floor(m10)
            f_part = math.floor((1 + m10 % 1) * 10.0**max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits])
            e_str = str(e10)
            if e10 >= 0:
                e_str = "+" + e_str
            return i_str + "." + f_str + "e" + e_str
