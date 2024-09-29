"""
Notes
-----
This module contains some geometrical functions.
"""

import numpy as np

__all__ = []


def map_z_line_to_chi(z, endpoints):
    """
    Function that maps the exterior of a line in the complex z-plane onto the exterior of the unit circle in the
    complex chi-plane.

    Parameters
    ----------
    z : complex or ndarray
        A complex point in the complex z-plane
    endpoints : list
        Endpoints of the line in the complex z-plane

    Returns
    -------
    chi : complex or ndarray
        The corresponding point in the complex chi-plane
    """
    # Map via the Z-plane
    big_z = (2 * z - endpoints[0] - endpoints[1]) / (endpoints[1] - endpoints[0])
    return big_z + np.sqrt(big_z - 1) * np.sqrt(big_z + 1)


def map_chi_to_z_line(chi, endpoints):
    """
    Function that maps the exterior of the unit circle in the complex chi-plane onto the exterior of a line in the
    complex z-plane.

    Parameters
    ----------
    chi : complex or ndarray
        A point in the complex chi-plane
    endpoints : list
        Endpoints of the line in the complex z-plane

    Returns
    -------
    z : complex or ndarray
        The corresponding point in the complex z-plane
    """
    # Map via the Z-plane
    big_z = 1 / 2 * (chi + 1 / chi)
    return 1 / 2 * (big_z * (endpoints[1] - endpoints[0]) + endpoints[0] + endpoints[1])


def map_z_circle_to_chi(z, r, center=0.0):
    """
    Function that maps a circle in the complex z-plane onto a unit circle in the complex chi-plane.

    Parameters
    ----------
    z : complex or ndarray
        A point in the complex z-plane
    r : float
        Radius of the circle
    center : complex or ndarray
        Center point of the circle in the complex z-plane

    Return
    ------
    chi : complex or ndarray
        The corresponding point in the complex chi-plane
    """
    return (z - center) / r


def map_chi_to_z_circle(chi, r, center=0.0):
    """
    Function that maps the unit circle in the complex chi-plane to a circle in the complex z-plane.
    
    Parameters
    ----------
    chi : complex or ndarray
        A point in the complex chi-plane 
    r : float
        Radius of the circle 
    center : complex or ndarray
        Center point of the circle 
    
    Return
    ------
    z : complex or ndarray
        The corresponding point in the complex z-plane
    """
    return chi * r + center


def get_chi_from_theta(nint, start, stop):
    """
    Function that creates an array with chi values for a given number of points along the unit circle.

    Parameters
    ----------
    nint : int
        Number of instances to generate
    start : float
        Start point
    stop : float
        Stop point

    Returns
    -------
    chi : ndarray
        Array with chi values
    """
    dtheta = (stop - start) / nint
    chi_temp = []
    for i in range(nint):
        theta = dtheta * i
        chi_temp.append(np.exp(1j * theta))
    return np.array(chi_temp)
