"""
Notes
-----
This module contains some geometrical functions.
"""
from typing import Any

import numpy as np
import numba as nb
from numpy import ndarray, dtype, floating

import andfn
from andfn import fracture
from andfn import intersection
from andfn import const_head
from . import NO_PYTHON


@nb.jit(nopython=NO_PYTHON, inline='always')
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

@nb.jit(nopython=NO_PYTHON, inline='always')
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

@nb.jit(nopython=NO_PYTHON, inline='always')
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

@nb.jit(nopython=NO_PYTHON, inline='always')
def map_chi_to_z_circle(chi, r, center=0.0):
    """
    Function that maps the unit circle in the complex chi-plane to a circle in the complex z-plane.
    
    Parameters
    ----------
    chi : complex
        A point in the complex chi-plane 
    r : float
        Radius of the circle 
    center : np.complex128
        Center point of the circle 
    
    Return
    ------
    z : complex
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
    chi : np.ndarray
        Array with chi values
    """
    dtheta = (stop - start) / nint
    chi_temp = []
    for i in range(nint):
        theta = dtheta * i
        chi_temp.append(np.exp(1j * theta))
    return np.array(chi_temp)


def map_2d_to_3d(z, fractures):
    """
    Function that maps a point in the complex z-plane to a point in the 3D plane

    Parameters
    ----------
    z : complex | np.ndarray
        A point in the complex z-plane
    fractures : Fracture
        The fracture object

    Returns
    -------
    point : np.ndarray
        The corresponding point in the 3D plane
    """
    if np.isscalar(z): # or z.size == 1:
        return np.real(z) * fractures.x_vector + np.imag(z) * fractures.y_vector + fractures.center
    return np.real(z)[:, np.newaxis] * fractures.x_vector + np.imag(z)[:, np.newaxis] * fractures.y_vector + fractures.center


def map_3d_to_2d(point, fractures):
    """
    Function that maps a point in the 3D plane to a point in the complex z-plane

    Parameters
    ----------
    point : np.ndarray
        A point in the 3D plane
    fractures : Fracture
        The fracture object

    Returns
    -------
    z : complex
        The corresponding point in the complex z-plane
    """
    x = np.dot((point - fractures.center), fractures.x_vector)
    y = np.dot((point - fractures.center), fractures.y_vector)
    return x + 1j * y

def fracture_intersection(frac0, frac1):
    # vector parallel to the intersection line
    n = np.cross(frac0.normal, frac1.normal)
    if n.sum() == 0:  # Check if the normals are parallel
        return None, None
    n = n / np.linalg.norm(n)

    # Calculate a point on the line of intersection
    n_1, n_2 = frac0.normal, frac1.normal
    p_1, p_2 = frac0.center, frac1.center
    a = np.matrix(np.array([[2, 0, 0, n_1[0], n_2[0]],
                            [0, 2, 0, n_1[1], n_2[1]],
                            [0, 0, 2, n_1[2], n_2[2]],
                            [n_1[0], n_1[1], n_1[2], 0, 0],
                            [n_2[0], n_2[1], n_2[2], 0, 0]]))
    b4 = p_1[0] * n_1[0] + p_1[1] * n_1[1] + p_1[2] * n_1[2]
    b5 = p_2[0] * n_2[0] + p_2[1] * n_2[1] + p_2[2] * n_2[2]
    b = np.matrix(np.array([[2.0 * p_1[0]], [2.0 * p_1[1]], [2.0 * p_1[2]], [b4], [b5]]))

    x = np.linalg.solve(a, b)
    xi_a = np.squeeze(np.asarray(x[0:3]))

    # Get two points on the intersection line and map them to each fracture
    xi_b = xi_a + n * 2.0
    z0_a, z0_b = map_3d_to_2d(xi_a, frac0), map_3d_to_2d(xi_b, frac0)
    z1_a, z1_b = map_3d_to_2d(xi_a, frac1), map_3d_to_2d(xi_b, frac1)

    # Get intersection points
    z0_0, z0_1 = line_circle_intersection(z0_a, z0_b, frac0.radius)
    z1_0, z1_1 = line_circle_intersection(z1_a, z1_b, frac1.radius)

    # Exit if there is no intersection with circle
    if z0_0 is None or z1_0 is None:
        return None, None

    # Get the shortest intersection line
    # See which intersection points are closest to the two centers of the fractures
    xi0_0, xi0_1 = map_2d_to_3d(z0_0, frac0), map_2d_to_3d(z0_1, frac0)
    xi1_0, xi1_1 = map_2d_to_3d(z1_0, frac1), map_2d_to_3d(z1_1, frac1)
    xis = [xi0_0, xi0_1, xi1_0, xi1_1]
    pos = [i for i, xi in enumerate(xis) if np.linalg.norm(xi - frac0.center) < frac0.radius + 1e-10 and np.linalg.norm(
        xi - frac1.center) < frac1.radius + 1e-10]
    if not pos:
        return None, None

    if len(pos) == 1:
        return None, None
    xi0, xi1 = xis[pos[0]], xis[pos[1]]

    endpoints0 = [map_3d_to_2d(xi0, frac0), map_3d_to_2d(xi1, frac0)]
    endpoints1 = [map_3d_to_2d(xi0, frac1), map_3d_to_2d(xi1, frac1)]

    return endpoints0, endpoints1


def line_circle_intersection(z0, z1, radius):
    # Get the components of the line equation y = mx + x0
    dx = np.real(z1 - z0)
    dy = np.imag(z1 - z0)
    if dx == 0:
        x = np.real(z0)
        y1 = np.sqrt(radius ** 2 - x ** 2)
        y2 = -y1
        return x + 1j * y1, x + 1j * y2

    m = dy / dx
    x0 = np.imag(z0) - m * np.real(z0)
    a = 1 + m ** 2
    b = 2 * x0 * m
    c = x0 ** 2 - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None, None
    sqrt_discriminant = np.sqrt(discriminant)
    x1 = (-b + sqrt_discriminant) / (2 * a)
    x2 = (-b - sqrt_discriminant) / (2 * a)
    y1 = m * x1 + x0
    y2 = m * x2 + x0

    return x1 + 1j * y1, x2 + 1j * y2


def line_line_intersection(z0, z1, z2, z3):

    determinant = ((np.conj(z1) - np.conj(z0))*(z3 - z2) - (z1 - z0)*(np.conj(z3) - np.conj(z2)))

    if determinant == 0:
        return None

    z = (np.conj(z1)*z0 - z1*np.conj(z0))*(z3 - z2) - (z1 - z0)*((np.conj(z3))*z2 - z3*np.conj(z2))
    z /= determinant

    return z



def generate_fractures(n_fractures, radius_factor=1.0, center_factor=10.0, ncoef=10, nint=20):
    fractures = []
    radii = np.random.rand(n_fractures) * radius_factor
    centers = np.random.rand(n_fractures, 3) * center_factor
    normals = np.random.rand(n_fractures, 3)
    for i in range(n_fractures):
        fractures.append(fracture.Fracture(f'{i + 1}', 1, radii[i], centers[i], normals[i], ncoef, nint))
        print(f'\r{i + 1} / {n_fractures}', end='')
    print('')
    return fractures


def get_connected_fractures(fractures, ncoef=5, nint=10, fracture_surface=None):
    connected_fractures = []
    fracture_list = fractures.copy()
    if fracture_surface is not None:
        fracture_list_it = [fracture_surface]
        connected_fractures.append(fracture_surface)
    else:
        fracture_list_it = [fracture_list[0]]
        connected_fractures.append(fracture_list[0])
        fracture_list.remove(fracture_list[0])
    fracture_list_it_temp = []
    cnt = 1
    while fracture_list_it:
        for i, fr in enumerate(fracture_list_it):
            for fr2 in fracture_list:
                if fr == fr2:
                    continue
                if np.linalg.norm(fr.center - fr2.center) > fr.radius + fr2.radius:
                    continue
                endpoints0, endpoints1 = fracture_intersection(fr, fr2)
                if endpoints0 is not None:
                    if fr2 not in []:
                        intersections = intersection.Intersection(f'{fr.label}_{fr2.label}', endpoints0, endpoints1, fr, fr2, ncoef, nint)
                        fr.add_element(intersections)
                        fr2.add_element(intersections)
                        if fr2 not in connected_fractures:
                            connected_fractures.append(fr2)
                            fracture_list_it_temp.append(fr2)
            print(
                f'\r{i + 1} / {len(fracture_list_it)}, iteration {cnt}, {len(fracture_list)} potential fractures left to analyze, {len(connected_fractures)} added to the DFN',
                end='')
        fracture_list_it = fracture_list_it_temp
        fracture_list_it_temp = []
        fracture_list = [f for f in fractures if f not in connected_fractures]
        cnt += 1
    print(f'\r{len(connected_fractures)} connected fractures found out of {len(fractures)} and took {cnt} iterations')
    return connected_fractures


def set_head_boundary(fractures, ncoef, nint, head, center, radius, normal, label):
    fracture_surface = andfn.Fracture(label, 1, radius, center, normal, ncoef, nint)
    fr = fracture_surface
    for fr2 in fractures:
        if fr == fr2:
            continue
        if np.linalg.norm(fr.center - fr2.center) > fr.radius + fr2.radius:
            continue
        endpoints0, endpoints1 = fracture_intersection(fr, fr2)
        if endpoints0 is not None:
            c_head = const_head.ConstantHeadLine(f'{label}_{fr2.label}', endpoints1, head, fr2, ncoef, nint)
            fr2.add_element(c_head)


def convert_trend_plunge_to_normal(trend, plunge):
    """
    Function that converts a trend and plunge to a normal vector

    Parameters
    ----------
    trend : float
        The trend of the fracture plane.
    plunge : float
        The plunge of the fracture plane.

    Returns
    -------
    normal : np.ndarray
        The normal vector of the fracture plane.
    """
    trend_rad = np.deg2rad(trend+90)
    plunge_rad = np.deg2rad(90-plunge)
    return np.array([-np.sin(plunge_rad)*np.cos(trend_rad), np.sin(plunge_rad)*np.sin(trend_rad), -np.cos(plunge_rad)])


def convert_strike_dip_to_normal(strike, dip):
    """
    Function that converts a strike and dip to a normal vector

    Parameters
    ----------
    strike : float
        The strike of the fracture plane.
    dip : float
        The dip of the fracture plane.

    Returns
    -------
    normal : np.ndarray
        The normal vector of the fracture plane.
    """
    if strike > 90:
        strike = 360 - (180 -strike)-90+90
    strike_rad = np.deg2rad(strike-90)
    dip_rad = np.deg2rad(dip)
    return np.array([-np.sin(dip_rad)*np.sin(strike_rad), np.cos(strike_rad)*np.sin(dip_rad), -np.cos(dip_rad)])


def convert_normal_to_strike_dip(normal):
    """
    Function that converts a normal vector to a strike and dip

    Parameters
    ----------
    normal : np.ndarray
        The normal vector of the fracture plane.

    Returns
    -------
    strike : float
        The strike of the fracture plane.
    dip : float
        The dip of the fracture plane.
    """
    strike = -np.arctan2(normal[0], normal[1])
    dip = -np.arcsin(normal[2])
    return np.rad2deg(strike), np.rad2deg(dip)
