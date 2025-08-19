"""
Notes
-----
This module contains some geometrical functions.
"""

import numpy as np
import numba as nb


@nb.njit(inline="always")
def map_z_line_to_chi(z, endpoints):
    """
    Function that maps the exterior of a line in the complex z-plane onto the exterior of the unit circle in the
    complex chi-plane.

    Parameters
    ----------
    z : complex
        A complex point in the complex z-plane
    endpoints : np.ndarray
        Endpoints of the line in the complex z-plane

    Returns
    -------
    chi : complex
        The corresponding point in the complex chi-plane
    """
    # Map via the Z-plane
    big_z = (2 * z - endpoints[0] - endpoints[1]) / (endpoints[1] - endpoints[0])
    return big_z + np.sqrt(big_z - 1) * np.sqrt(big_z + 1)


@nb.njit(inline="always")
def map_chi_to_z_line(chi, endpoints):
    """
    Function that maps the exterior of the unit circle in the complex chi-plane onto the exterior of a line in the
    complex z-plane.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    endpoints : np.ndarray[np.complex128]
        Endpoints of the line in the complex z-plane

    Returns
    -------
    z : complex
        The corresponding point in the complex z-plane
    """
    # Map via the Z-plane
    big_z = 1 / 2 * (chi + 1 / chi)
    z = 1 / 2 * (big_z * (endpoints[1] - endpoints[0]) + endpoints[0] + endpoints[1])
    return z


@nb.njit(inline="always")
def map_z_circle_to_chi(z, r, center=0.0):
    """
    Function that maps a circle in the complex z-plane onto a unit circle in the complex chi-plane.

    Parameters
    ----------
    z : complex
        A point in the complex z-plane
    r : float
        Radius of the circle
    center : np.complex128
        Center point of the circle in the complex z-plane

    Return
    ------
    chi : complex
        The corresponding point in the complex chi-plane
    """
    return (z - center) / r


@nb.njit(inline="always")
def map_chi_to_z_circle(chi, r, center=0.0):
    """
    Function that maps the unit circle in the complex chi-plane to a circle in the complex z-plane.

    Parameters
    ----------
    chi : np.complex128
        A point in the complex chi-plane
    r : float
        Radius of the circle
    center : np.complex128
        Center point of the circle

    Return
    ------
    z : np.complex128
        The corresponding point in the complex z-plane
    """
    return chi * r + center


@nb.njit()
def map_2d_to_3d(self_, z, pnts):
    """
    Function that maps a point in the complex z-plane to a point in the 3D plane

    .. math::
            x_i = x u_i + y v_i + x_{i,0}

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    z : np.ndarray[np.complex128]
        A point in the complex z-plane
    pnts : np.ndarray[np.float64]
        An array to store the resulting points in the 3D plane

    Returns
    -------
    point : np.ndarray[np.float64]
        The corresponding point in the 3D plane
    """

    for i in range(len(z)):
        pnts[i] = (
            np.real(z[i]) * self_["x_vector"]
            + np.imag(z[i]) * self_["y_vector"]
            + self_["center"]
        )

    return pnts
