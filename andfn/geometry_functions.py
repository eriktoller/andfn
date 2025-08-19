"""
Notes
-----
This module contains some geometrical functions for e.g. conformal mappings and mapping between 3D space and fracture planes.

The geometrical functions are used by the element classes and to create the DFN in the andfn module.
"""

import numpy as np

import andfn
from . import fracture
from . import intersection
from . import const_head


def map_z_line_to_chi(z, endpoints):
    """
    Function that maps the exterior of a line in the complex z-plane onto the exterior of the unit circle in the
    complex chi-plane.

    .. math::
            Z = \frac{ 2z - \text{endpoints}[0] - \text{endpoints}[1] }{ \text{endpoints}[1] - \text{endpoints}[0]}

    .. math::
            \chi = \frac{1}{2} \left( z + \sqrt{z - 1} \sqrt{z + 1} \right)

    Parameters
    ----------
    z : complex | np.ndarray
        A complex point in the complex z-plane
    endpoints : np.ndarray
        Endpoints of the line in the complex z-plane

    Returns
    -------
    chi : complex | np.ndarray
        The corresponding point in the complex chi-plane
    """
    # Map via the Z-plane
    big_z = np.vectorize(
        lambda zz: (2 * zz - endpoints[0] - endpoints[1])
        / (endpoints[1] - endpoints[0])
    )(z)
    return big_z + np.sqrt(big_z - 1) * np.sqrt(big_z + 1)


def map_chi_to_z_line(chi, endpoints):
    r"""
    Function that maps the exterior of the unit circle in the complex chi-plane onto the exterior of a line in the
    complex z-plane.

    .. math::
            Z = \frac{1}{2} \left( \chi + \frac{1}{\chi} \right)

    .. math::
            z = \frac{1}{2} \left( Z \left(\text{endpoints}[1] - \text{endpoints}[0] \right) + \text{endpoints}[0] + \text{endpoints}[1]\right)

    Parameters
    ----------
    chi : complex | np.ndarray
        A point in the complex chi-plane
    endpoints : list | np.ndarray
        Endpoints of the line in the complex z-plane

    Returns
    -------
    z : complex | np.ndarray
        The corresponding point in the complex z-plane
    """
    # Map via the Z-plane
    big_z = 1 / 2 * (chi + 1 / chi)
    return 1 / 2 * (big_z * (endpoints[1] - endpoints[0]) + endpoints[0] + endpoints[1])


# @nb.jit(nopython=NO_PYTHON)
def map_z_circle_to_chi(z, r, center=0.0):
    r"""
    Function that maps a circle in the complex z-plane onto a unit circle in the complex chi-plane.

    .. math::
            \chi = \frac{z - \text{center}}{r}


    Parameters
    ----------
    z : complex | np.ndarray
        A point in the complex z-plane
    r : float
        Radius of the circle
    center : complex | np.ndarray
        Center point of the circle in the complex z-plane

    Return
    ------
    chi : np.ndarray
        The corresponding point in the complex chi-plane
    """
    return (z - center) / r


def map_chi_to_z_circle(chi, r, center=0.0):
    r"""
    Function that maps the unit circle in the complex chi-plane to a circle in the complex z-plane.

    .. math::
            z = \chi r + \text{center}

    Parameters
    ----------
    chi : complex | np.ndarray
        A point in the complex chi-plane
    r : float
        Radius of the circle
    center : complex or np.ndarray
        Center point of the circle

    Return
    ------
    z : complex | np.ndarray
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

    .. math::
            x_i = x u_i + y v_i + x_{i,0}

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
    if np.isscalar(z):  # or z.size == 1:
        return (
            np.real(z) * fractures.x_vector
            + np.imag(z) * fractures.y_vector
            + fractures.center
        )
    return (
        np.real(z)[:, np.newaxis] * fractures.x_vector
        + np.imag(z)[:, np.newaxis] * fractures.y_vector
        + fractures.center
    )


def map_3d_to_2d(point, fractures):
    """
    Function that maps a point in the 3D plane to a point in the complex z-plane.

    .. math::
            x = \left( x_i - x_{i,0} \right) u_i

    .. math::
            y = \left( x_i - x_{i,0} \right) v_i

    .. math::
            z = x + iy

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
    """
    Function that calculates the intersection between two fractures.

    Parameters
    ----------
    frac0 : Fracture
        The first fracture.
    frac1 : Fracture
        The second fracture.

    Returns
    -------
    endpoints0 : np.ndarray
        The endpoints of the intersection line in the first fracture. If no intersection is found, None is returned.
    endpoints1 : np.ndarray
        The endpoints of the intersection line in the second fracture. If no intersection is found, None is returned.
    """
    # vector parallel to the intersection line
    n = np.cross(frac0.normal, frac1.normal)
    if n.sum() == 0:  # Check if the normals are parallel
        return None, None
    n = n / np.linalg.norm(n)

    # Calculate a point on the line of intersection
    n_1, n_2 = frac0.normal, frac1.normal
    p_1, p_2 = frac0.center, frac1.center
    a = np.matrix(
        np.array(
            [
                [2, 0, 0, n_1[0], n_2[0]],
                [0, 2, 0, n_1[1], n_2[1]],
                [0, 0, 2, n_1[2], n_2[2]],
                [n_1[0], n_1[1], n_1[2], 0, 0],
                [n_2[0], n_2[1], n_2[2], 0, 0],
            ]
        )
    )
    b4 = p_1[0] * n_1[0] + p_1[1] * n_1[1] + p_1[2] * n_1[2]
    b5 = p_2[0] * n_2[0] + p_2[1] * n_2[1] + p_2[2] * n_2[2]
    b = np.matrix(
        np.array([[2.0 * p_1[0]], [2.0 * p_1[1]], [2.0 * p_1[2]], [b4], [b5]])
    )

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
    pos = [
        i
        for i, xi in enumerate(xis)
        if np.linalg.norm(xi - frac0.center) < frac0.radius + 1e-10
        and np.linalg.norm(xi - frac1.center) < frac1.radius + 1e-10
    ]
    if not pos:
        return None, None

    if len(pos) == 1:
        return None, None
    xi0, xi1 = xis[pos[0]], xis[pos[1]]

    endpoints0 = np.array([map_3d_to_2d(xi0, frac0), map_3d_to_2d(xi1, frac0)])
    endpoints1 = np.array([map_3d_to_2d(xi0, frac1), map_3d_to_2d(xi1, frac1)])

    return endpoints0, endpoints1


def line_circle_intersection(z0, z1, radius):
    """
    Function that calculates the intersection between a line and a circle.

    Parameters
    ----------
    z0 : complex
        A point on the line.
    z1 : complex
        Another point on the line.
    radius : float
        The radius of the circle.

    Returns
    -------
    z_0 : complex
        The first intersection point. If no intersection is found, None is returned.
    z_1 : complex
        The second intersection point. If no intersection is found, None is returned.
    """
    # Get the components of the line equation y = mx + x0
    dx = np.real(z1 - z0)
    dy = np.imag(z1 - z0)
    if dx == 0:
        x = np.real(z0)
        y1 = np.sqrt(radius**2 - x**2)
        y2 = -y1
        return x + 1j * y1, x + 1j * y2

    m = dy / dx
    x0 = np.imag(z0) - m * np.real(z0)
    a = 1 + m**2
    b = 2 * x0 * m
    c = x0**2 - radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None
    sqrt_discriminant = np.sqrt(discriminant)
    x1 = (-b + sqrt_discriminant) / (2 * a)
    x2 = (-b - sqrt_discriminant) / (2 * a)
    y1 = m * x1 + x0
    y2 = m * x2 + x0

    return x1 + 1j * y1, x2 + 1j * y2


def line_line_intersection(z0, z1, z2, z3):
    """
    Function that calculates the intersection between two lines.

    Parameters
    ----------
    z0 : complex
        A point on the first line.
    z1 : complex
        Another point on the first line.
    z2 : complex
        A point on the second line.
    z3 : complex
        Another point on the second line.

    Returns
    -------
    z : complex
        The intersection point. If no intersection is found, None is returned.
    """

    determinant = (np.conj(z1) - np.conj(z0)) * (z3 - z2) - (z1 - z0) * (
        np.conj(z3) - np.conj(z2)
    )

    if determinant == 0:
        return None

    z = (np.conj(z1) * z0 - z1 * np.conj(z0)) * (z3 - z2) - (z1 - z0) * (
        (np.conj(z3)) * z2 - z3 * np.conj(z2)
    )
    z /= determinant

    return z


def generate_fractures(
    n_fractures, radius_factor=1.0, center_factor=10.0, ncoef=10, nint=20
):
    """
    Function that generates a number of fractures with random radii, centers and normals.

    Parameters
    ----------
    n_fractures : int
        Number of fractures to generate.
    radius_factor : float
        The maximum radius of the fractures.
    center_factor : float
        The maximum distance from the origin of the centers of the fractures.
    ncoef : int
        The number of coefficients for the bounding circle.
    nint : int
        The number of integration points for the bounding circle.

    Returns
    -------
    fractures : list
        A list of the generated fractures.
    """
    fractures = []
    radii = np.random.rand(n_fractures) * radius_factor
    centers = np.random.rand(n_fractures, 3) * center_factor
    normals = np.random.rand(n_fractures, 3)
    for i in range(n_fractures):
        fractures.append(
            fracture.Fracture(
                f"{i + 1}", 1, radii[i], centers[i], normals[i], ncoef, nint
            )
        )
        print(f"\r{i + 1} / {n_fractures}", end="")
    print("")
    return fractures


def get_connected_fractures(
    fractures, se_factor, ncoef=5, nint=10, fracture_surface=None
):
    """
    Function that finds all connected fractures in a list of fractures. Starting from the first fracture in the list, or
    a given fracture, the function iterates through the list of fractures and finds all connected fractures.

    Parameters
    ----------
    fractures : list
        A list of fractures.
    se_factor : float
        The shortening element factor. This is used to shorten the intersection line between two fractures.
    ncoef : int
        The number of coefficients for the intersection elements.
    nint : int
        The number of integration points for the intersection elements.
    fracture_surface : Fracture
        The fracture to start the search from. If None, the first fracture in the list is used.

    Returns
    -------
    connected_fractures : list
        A list of connected fractures.
    """
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
                        # length = np.linalg.norm(endpoints0[0] - endpoints0[1])
                        # if length < 1e-1*fr.radius or length < 1e-1*fr2.radius:
                        #    continue
                        endpoints01 = shorten_line(
                            endpoints0[0], endpoints0[1], se_factor
                        )
                        endpoints11 = shorten_line(
                            endpoints1[0], endpoints1[1], se_factor
                        )
                        intersections = intersection.Intersection(
                            f"{fr.label}_{fr2.label}",
                            endpoints01,
                            endpoints11,
                            fr,
                            fr2,
                            ncoef,
                            nint,
                        )
                        fr.add_element(intersections)
                        fr2.add_element(intersections)
                        if fr2 not in connected_fractures:
                            connected_fractures.append(fr2)
                            fracture_list_it_temp.append(fr2)
            print(
                f"\r{i + 1} / {len(fracture_list_it)}, iteration {cnt}, {len(fracture_list)} potential fractures left to analyze, {len(connected_fractures)} added to the DFN",
                end="",
            )
        fracture_list_it = fracture_list_it_temp
        fracture_list_it_temp = []
        fracture_list = [f for f in fractures if f not in connected_fractures]
        cnt += 1
    print(
        f"\r{len(connected_fractures)} connected fractures found out of {len(fractures)} and took {cnt} iterations"
    )
    return connected_fractures


def get_fracture_intersections(fractures, se_factor, ncoef=5, nint=10):
    """
    Function that finds all connected fractures in a list of fractures. Starting from the first fracture in the list, or
    a given fracture, the function iterates through the list of fractures and finds all connected fractures.

    Parameters
    ----------
    fractures : list
        A list of fractures.
    se_factor : float
        The shortening element factor. This is used to shorten the intersection line between two fractures.
    ncoef : int
        The number of coefficients for the intersection elements.
    nint : int
        The number of integration points for the intersection elements.

    Returns
    -------
    connected_fractures : list
        A list of connected fractures.
    """
    for i, fr in enumerate(fractures):
        for fr2 in fractures[i + 1 :]:
            if fr == fr2:
                continue
            if np.linalg.norm(fr.center - fr2.center) > fr.radius + fr2.radius:
                continue
            endpoints0, endpoints1 = fracture_intersection(fr, fr2)
            if endpoints0 is not None:
                if fr2 not in []:
                    endpoints01 = shorten_line(endpoints0[0], endpoints0[1], se_factor)
                    endpoints11 = shorten_line(endpoints1[0], endpoints1[1], se_factor)
                    intersections = intersection.Intersection(
                        f"{fr.label}_{fr2.label}",
                        endpoints01,
                        endpoints11,
                        fr,
                        fr2,
                        ncoef,
                        nint,
                    )
                    fr.add_element(intersections)
                    fr2.add_element(intersections)

    return fractures


def remove_isolated_fractures(fractures):
    """
    Function that removes isolated fractures from a list of fractures. An isolated fracture is a fracture that does not
    have any intersection with other fractures.

    Parameters
    ----------
    fractures : list
        A list of fractures.

    Returns
    -------
    fractures : list
        A list of fractures with isolated fractures removed.
    """
    return [fr for fr in fractures if any(isinstance(el, intersection.Intersection) for el in fr.elements)]


def set_head_boundary(
    fractures, ncoef, nint, head, center, radius, normal, label, se_factor
):
    """
    Function that sets a constant head boundary condition on the intersection line between a fracture and a defined
    fracture. The constant head lines are added to the fractures in the list.

    Parameters
    ----------
    fractures : list
        A list of fractures.
    ncoef : int
        The number of coefficients for the constant head line.
    nint : int
        The number of integration points for the constant head line.
    head : float
        The hydraulic head value.
    center : np.ndarray
        The center of the constant head fracture plane.
    radius : float
        The radius of the constant head fracture plane.
    normal : np.ndarray
        The normal vector of the constant head fracture plane.
    label : str
        The label of the constant head fracture plane.
    se_factor : float
        The shortening element factor. This is used to shorten the constant head line.

    Returns
    -------
    None
    """
    fracture_surface = andfn.Fracture(label, 1, radius, center, normal, ncoef, nint)
    fr = fracture_surface
    for fr2 in fractures:
        if fr == fr2:
            continue
        if np.linalg.norm(fr.center - fr2.center) > fr.radius + fr2.radius:
            continue
        endpoints0, endpoints1 = fracture_intersection(fr, fr2)
        if endpoints0 is not None:
            endpoints = shorten_line(endpoints1[0], endpoints1[1], se_factor)
            const_head.ConstantHeadLine(
                f"{label}_{fr2.label}", endpoints, head, fr2, ncoef, nint
            )


def shorten_line(z0, z1, se_factor):
    """
    Function that shortens a line segment by a given se_factor and keeps the same center point.

    Parameters
    ----------
    z0 : complex
    z1 : complex
    se_factor : float

    Returns
    -------
    np.ndarray
    """
    center = (z0 + z1) / 2
    z0 = center + (z0 - center) * se_factor
    z1 = center + (z1 - center) * se_factor
    return np.array([z0, z1])


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
    trend_rad = np.deg2rad(trend + 90)
    plunge_rad = np.deg2rad(90 - plunge)
    return np.array(
        [
            -np.sin(plunge_rad) * np.cos(trend_rad),
            np.sin(plunge_rad) * np.sin(trend_rad),
            -np.cos(plunge_rad),
        ]
    )


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
        strike = 360 - (180 - strike) - 90 + 90
    strike_rad = np.deg2rad(strike - 90)
    dip_rad = np.deg2rad(dip)
    return np.array(
        [
            -np.sin(dip_rad) * np.sin(strike_rad),
            np.cos(strike_rad) * np.sin(dip_rad),
            -np.cos(dip_rad),
        ]
    )


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
