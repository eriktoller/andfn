"""
Notes
-----
This module contains some geometrical functions.
"""


import numpy as np

__all__ = []


def map_z_line_to_chi(z, z0, z1):
    """
    Function that maps the exterior of a line in the complex z-plane onto the exterior of the unit circle in the
    complex chi-plane.

    Parameters
    ----------
    z : complex
        A complex point in the complex z-plane
    z0 : complex
        Startpoint of the line in the complex z-plane
    z1 : complex
        Endpoint of the line in the complex z-plane

    Returns
    -------
    chi : complex
        The corresponding point in the complex chi-plane
    """
    # Map via the Z-plane
    big_z = (2 * z - z0 - z1) / (z1 - z0)
    return big_z + np.sqrt(big_z - 1) * np.sqrt(big_z + 1)


def map_chi_to_z_line(chi, z0, z1):
    """
    Function that maps the exterior of the unit circle in the complex chi-plane onto the exterior of a line in the
    complex z-plane.

    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane
    z0 : complex
        Startpoint of the line in the complex z-plane
    z1 : complex
        Endpoint of the line in the complex z-plane

    Returns
    -------
    z : complex
        The corresponding point in the complex z-plane
    """
    # Map via the Z-plane
    big_z = 1 / 2 * (chi + 1 / chi)
    return 1 / 2 * (big_z * (z1 - z0) + z0 + z1)


def map_z_circle_to_chi(z, r, center=0.0):
    """
    Function that maps a circle in the complex z-plane onto a unit circle in the complex chi-plane.

    Parameters
    ----------
    z : complex
        A point in the complex z-plane
    r : float
        Radius of the circle
    center : float
        Center point of the circle in the complex z-plane

    Return
    ------
    chi : complex
        The corresponding point in the complex chi-plane
    """
    return (z-center)/r


def map_chi_to_z_circle(chi, r, center=0.0):
    """
    Function that maps the unit circle in the complex chi-plane to a circle in the complex z-plane.
    
    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane 
    r : float
        Radius of the circle 
    center : float
        Center point of the circle 
    
    Return
    ------
    z : complex
        The corresponding point in the complex z-plane
    """
    return chi*r + center


