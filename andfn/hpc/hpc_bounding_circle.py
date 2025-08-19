import numpy as np
import numba as nb
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf
from andfn.hpc import hpc_fracture


@nb.njit(inline="always")
def get_chi(self_, z):
    """
    Maps the complex z plane to the complex chi plane for the bounding circle.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z : complex

    Returns
    -------
    chi : complex
        The mapped point in the chi plane.
    """
    chi = gf.map_z_circle_to_chi(z, self_["radius"])
    return chi


@nb.njit(inline="always")
def z_array(self_, n):
    """
    Returns an array of points in the complex z plane.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    n : int
        The number of points to return.

    Returns
    -------
    z : np.ndarray[complex]
        The array of points in the complex z plane.
    """
    theta = np.zeros(n, dtype=np.float64)
    mf.calc_thetas(n, self_["_type"], theta)
    chi = np.exp(1j * theta)
    z = gf.map_chi_to_z_circle(chi, self_["radius"])
    return z


@nb.njit(inline="always")
def calc_omega(self_, z):
    """
    Calculates the omega for the bounding circle.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z : complex
        A point in the complex z plane.

    Returns
    -------
    omega : complex
        The complex potential for the bounding circle.
    """
    chi = get_chi(self_, z)
    omega = mf.taylor_series(chi, self_["coef"][: self_["ncoef"]])
    return omega


# @nb.njit(, inline='always')
def calc_w(self_, z):
    """
    Calculates the omega for the bounding circle.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z : complex
        A point in the complex z plane.

    Returns
    -------
    omega : complex
        The complex potential for the bounding circle.
    """
    chi = get_chi(self_, z)
    w = mf.taylor_series_d1(chi, self_["coef"][: self_["ncoef"]])
    return w


@nb.njit()
def solve(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Solves the bounding circle element.

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
    Edits the self_ array and works_array in place.
    """

    mf.get_dpsi_corr(self_, fracture_struc_array, element_struc_array, work_array)
    frac0 = fracture_struc_array[self_["frac0"]]
    work_array["old_coef"][: self_["ncoef"]] = self_["coef"][: self_["ncoef"]]
    mf.cauchy_integral_domega(
        self_["nint"],
        self_["ncoef"],
        self_["dpsi_corr"][: self_["nint"] - 1],
        frac0,
        self_["_id"],
        element_struc_array,
        self_["radius"],
        self_["center"],
        work_array,
        work_array["coef"][: self_["ncoef"]],
    )
    work_array["coef"][: self_["ncoef"]] = -work_array["coef"][: self_["ncoef"]]
    # self_['error'] = np.max(np.abs(work_array['coef'][:self_['ncoef']] - work_array['old_coef'][:self_['ncoef']]))
    self_["error_old"] = self_["error"]
    self_["error"] = mf.calc_error(
        work_array["coef"][: self_["ncoef"]], work_array["old_coef"][: self_["ncoef"]]
    )


@nb.njit()
def check_boundary_condition(self_, fracture_struc_array, element_struc_array, n=10):
    """
    Check if the bounding circle satisfies the boundary conditions.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    n : int
        The number of points to check the boundary condition

    Returns
    -------
    error : np.float64
        The error in the boundary condition
    """

    # Calculate the stream function on the boundary of the fracture
    chi = np.exp(1j * self_["thetas"])
    z = gf.map_chi_to_z_circle(chi, self_["radius"])
    frac0 = fracture_struc_array[self_["frac0"]]
    omega0 = hpc_fracture.calc_omega(frac0, z, element_struc_array)
    psi = np.imag(omega0)
    dpsi = psi[1:] - psi[:-1]
    q = np.sum(np.abs(self_["dpsi_corr"][: self_["nint"] - 1]))
    mean_dpsi = np.abs(np.max(dpsi) - np.min(dpsi))
    # if mean_dpsi > q/2:
    #    mean_dpsi = np.abs(np.abs(np.max(dpsi) - np.min(dpsi)) - q)
    if q < 1e-10:
        return np.abs(np.max(psi) - np.min(psi))
    return mean_dpsi / q
