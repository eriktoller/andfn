"""
Notes
-----
This module contains some geometrical functions for e.g. conformal mappings and mapping between 3D space and fracture planes.

The geometrical functions are used by the element classes and to create the DFN in the andfn module.
"""

import time

import numpy as np
import numba as nb
from scipy.spatial import KDTree

import andfn
from . import fracture
from . import intersection
from .const_head import ConstantHeadLine
from .well import Well
from .impermeable_object import ImpermeableLine
import andfn.hpc.hpc_geometry_functions as hpc_gf


def map_z_line_to_chi(z, endpoints):
    """
    Function that maps the exterior of a line in the complex z-plane onto the exterior of the unit circle in the
    complex chi-plane.

    .. math::
            Z = \frac{ 2z - \text{endpoints}[0] - \text{endpoints}[1] }{ \text{endpoints}[1] - \text{endpoints}[0]}

    .. math::
            \\chi = \\frac{1}{2} \\left( z + \\sqrt{z - 1} \\sqrt{z + 1} \\right)

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
    if np.isscalar(z):
        return hpc_gf.map_z_line_to_chi(z, endpoints)
    else:
        chi = np.empty_like(z, dtype=np.complex128)
        for i, z0 in enumerate(z):
            chi[i] = hpc_gf.map_z_line_to_chi(z0, endpoints)
        return chi


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
            x = \\left( x_i - x_{i,0} \\right) u_i

    .. math::
            y = \\left( x_i - x_{i,0} \\right) v_i

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


def fracture_intersection_org(frac0, frac1):
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
    if np.allclose(n, 0):  # Check if the normals are parallel
        return None, None
    n = n / np.linalg.norm(n)

    # Calculate a point on the line of intersection
    n_1, n_2 = frac0.normal, frac1.normal
    p_1, p_2 = frac0.center, frac1.center
    a = np.array(
        [
            [2, 0, 0, n_1[0], n_2[0]],
            [0, 2, 0, n_1[1], n_2[1]],
            [0, 0, 2, n_1[2], n_2[2]],
            [n_1[0], n_1[1], n_1[2], 0, 0],
            [n_2[0], n_2[1], n_2[2], 0, 0],
        ]
    )
    b4 = p_1[0] * n_1[0] + p_1[1] * n_1[1] + p_1[2] * n_1[2]
    b5 = p_2[0] * n_2[0] + p_2[1] * n_2[1] + p_2[2] * n_2[2]
    b = np.array(
        [[2.0 * p_1[0]], [2.0 * p_1[1]], [2.0 * p_1[2]], [b4], [b5]]
    )  # Get two points on the intersection line and map them to each fracture
    x = np.linalg.solve(a, b)
    xi_a = np.squeeze(np.asarray(x[0:3]))
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
        if np.linalg.norm(xi - frac0.center) < frac0.radius + 1e-7
        and np.linalg.norm(xi - frac1.center) < frac1.radius + 1e-7
    ]
    if not pos:
        return None, None

    if len(pos) == 1:
        return None, None
    xi0, xi1 = xis[pos[0]], xis[pos[1]]

    endpoints0 = np.array([map_3d_to_2d(xi0, frac0), map_3d_to_2d(xi1, frac0)])
    endpoints1 = np.array([map_3d_to_2d(xi0, frac1), map_3d_to_2d(xi1, frac1)])

    return endpoints0, endpoints1


def fracture_intersection(frac0, frac1):
    # Line direction = intersection of planes
    n0, n1 = frac0.normal, frac1.normal
    d = np.cross(n0, n1)
    d_norm2 = np.dot(d, d)

    if d_norm2 < 1e-14:
        return None, None  # planes parallel or coincident

    # Plane offsets: n · x = c
    c0 = np.dot(n0, frac0.center)
    c1 = np.dot(n1, frac1.center)

    # Point on intersection line (closed-form)
    x0 = (c0 * np.cross(n1, d) + c1 * np.cross(d, n0)) / d_norm2

    # Two points defining the line
    x1 = x0 + d

    # Project line to fracture-local 2D coordinates
    z00, z01 = map_3d_to_2d(x0, frac0), map_3d_to_2d(x1, frac0)
    z10, z11 = map_3d_to_2d(x0, frac1), map_3d_to_2d(x1, frac1)

    # Intersect with circular fracture boundaries
    a0, b0 = line_circle_intersection(z00, z01, frac0.radius)
    a1, b1 = line_circle_intersection(z10, z11, frac1.radius)

    if a0 is None or a1 is None:
        return None, None

    # Lift valid endpoints into 3D
    candidates = [
        map_2d_to_3d(a0, frac0),
        map_2d_to_3d(b0, frac0),
        map_2d_to_3d(a1, frac1),
        map_2d_to_3d(b1, frac1),
    ]

    # Keep points inside both fractures
    valid = [
        x
        for x in candidates
        if np.linalg.norm(x - frac0.center) <= frac0.radius + 1e-8
        and np.linalg.norm(x - frac1.center) <= frac1.radius + 1e-8
    ]

    if len(valid) != 2:
        return None, None

    p0, p1 = valid

    endpoints0 = np.array([map_3d_to_2d(p0, frac0), map_3d_to_2d(p1, frac0)])
    endpoints1 = np.array([map_3d_to_2d(p0, frac1), map_3d_to_2d(p1, frac1)])

    return endpoints0, endpoints1  # endpoints0, endpoints1


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
    fractures, se_factor, ncoef=5, nint=10, fracture_surface=None, tolerance=-1
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
                        length = np.linalg.norm(endpoints0[0] - endpoints0[1])
                        if (
                            length < tolerance * fr.radius
                            or length < tolerance * fr2.radius
                        ):
                            continue
                        endpoints01 = shorten_line(endpoints0, se_factor)
                        endpoints11 = shorten_line(endpoints1, se_factor)
                        intersection.Intersection(
                            f"{fr.label}_{fr2.label}",
                            endpoints01,
                            endpoints11,
                            fr,
                            fr2,
                            ncoef,
                            nint,
                        )
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


def get_fracture_intersections_org(
    fractures, se_factor, ncoef=5, nint=10, tolerance=-1
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
    tolerance : float, optional
        The minimum length of the intersection line as a fraction of the fracture radius. Intersections shorter than
        this value will be ignored. If -1, no tolerance is applied. Default is -1.


    Returns
    -------
    connected_fractures : list
        A list of connected fractures.
    """
    for i, fr in enumerate(fractures):
        print(f"\r{i + 1} / {len(fractures)} fractures processed", end="")
        # Get the celltreeboxes
        # celltree = numba_celltree.CellTree3d(vertices3d, boxes)
        # overlapping_i, overlapping_j = celltree.locate_boxes(box_bbounds)
        for fr2 in fractures[i + 1 :]:
            if fr == fr2:
                continue
            if np.linalg.norm(fr.center - fr2.center) > fr.radius + fr2.radius:
                continue
            endpoints0, endpoints1 = fracture_intersection(fr, fr2)
            if endpoints0 is not None:
                if fr2 not in []:
                    length = np.linalg.norm(endpoints0[0] - endpoints0[1])
                    if (
                        length < tolerance * fr.radius
                        or length < tolerance * fr2.radius
                    ):
                        continue
                    endpoints01 = shorten_line(endpoints0, se_factor)
                    endpoints11 = shorten_line(endpoints1, se_factor)
                    intersection.Intersection(
                        f"{fr.label}_{fr2.label}",
                        endpoints01,
                        endpoints11,
                        fr,
                        fr2,
                        ncoef,
                        nint,
                    )
                    # fr.add_element(intersections)
                    # fr2.add_element(intersections)

    return fractures


def get_fracture_intersections(
    fractures, se_factor, ncoef=5, nint=10, tolerance=-1, tree=None
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
    tolerance : float, optional
        The minimum length of the intersection line as a fraction of the fracture radius. Intersections shorter than
        this value will be ignored. If -1, no tolerance is applied. Default is -1.
    tree : KDTree, optional
        A KDTree object for spatial indexing of the fracture centers. If None, a new KDtree will be created. Default is None.


    Returns
    -------
    connected_fractures : list
        A list of connected fractures.
    """
    # 1. Build a spatial index based on fracture centers
    if tree is None:
        # Sort fractures by radius to optimize the search
        fractures = sorted(fractures, key=lambda f: f.radius, reverse=True)
        centers = np.array([fr.center for fr in fractures])
        tree = KDTree(centers)

    # Check if fractures are sorted by radius
    if not all(
        fractures[i].radius >= fractures[i + 1].radius
        for i in range(len(fractures) - 1)
    ):
        raise ValueError(
            "Fractures must be sorted by radius in descending order for the spatial index to work correctly."
        )

    # 2. Query the tree for pairs within a possible intersection distance
    # This will give us candidate pairs of fractures that might intersect, significantly reducing the number of expensive intersection checks.
    pairs = set()
    for i, fr in enumerate(fractures):
        r = fr.radius * 2  # Maximum distance for potential intersection
        neighbors = tree.query_ball_point(fr.center, r)
        for j in neighbors:
            if j > i:
                pairs.add((i, j))

    n_pairs = len(pairs)
    for cnt, (i, j) in enumerate(pairs, start=1):
        if cnt % 50000 == 0:
            print(f"\r{cnt} / {n_pairs} fracture pairs processed", end="")
        cnt += 1
        fr, fr2 = fractures[i], fractures[j]

        # 3. Quick Squared Distance Check
        dx = fr.center[0] - fr2.center[0]
        dy = fr.center[1] - fr2.center[1]
        dz = fr.center[2] - fr2.center[2]
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq > (fr.radius + fr2.radius) ** 2:
            continue

        # 4. Perform expensive geometric intersection
        endpoints0, endpoints1 = fracture_intersection(fr, fr2)
        if endpoints0 is not None:
            dx = endpoints0[0].real - endpoints0[1].real
            dy = endpoints0[0].imag - endpoints0[1].imag
            length = (dx * dx + dy * dy) ** 0.5
            if length < tolerance * fr.radius or length < tolerance * fr2.radius:
                continue
            endpoints01 = shorten_line(endpoints0, se_factor)
            endpoints11 = shorten_line(endpoints1, se_factor)
            intersection.Intersection(
                f"{fr.label}_{fr2.label}",
                endpoints01,
                endpoints11,
                fr,
                fr2,
                ncoef,
                nint,
            )
    print(f"\r{len(pairs)} fracture pairs processed. Intersection detection complete.")
    return fractures


def split_crossing_elements(fractures):
    """
    Function that splits crossing intersection elements in a list of fractures. If two intersection elements cross each
    other, they are split into four new intersection elements.

    Parameters
    ----------
    fractures : list
        A list of fractures.

    Returns
    -------
    fractures : list
        A list of fractures with crossing intersection elements split.
    """

    def check_crossing(el, el2, frac):
        if el.frac0 == frac:
            z0 = el.endpoints0[0]
            z1 = el.endpoints0[1]
        else:
            z0 = el.endpoints1[0]
            z1 = el.endpoints1[1]
        if el2.frac0 == frac:
            z2 = el2.endpoints0[0]
            z3 = el2.endpoints0[1]
        else:
            z2 = el2.endpoints1[0]
            z3 = el2.endpoints1[1]

        z = line_line_intersection(
            el.endpoints0[0], el.endpoints0[1], el2.endpoints0[0], el2.endpoints0[1]
        )
        if z is None:
            return False
        atol = 1e-12
        if np.abs(np.abs(z - z0) + np.abs(z1 - z) - np.abs(z0 - z1)) > atol:
            return False

        if np.abs(np.abs(z - z2) + np.abs(z - z3) - np.abs(z2 - z3)) > atol:
            return False
        ltol = 1e-10
        if (
            (np.abs(z - z0) < ltol)
            or (np.abs(z - z1) < ltol)
            or (np.abs(z - z2) < ltol)
            or (np.abs(z - z3) < ltol)
        ):
            return False
        return z

    def create_new_element(frac, el, new_endpoints0, new_endpoints1):
        if isinstance(el, intersection.Intersection):
            # map endpoints to correct fractures
            if el.frac0 == frac:
                z3d = map_2d_to_3d(new_endpoints0, frac)
                new_endpoints1 = map_3d_to_2d(z3d, el.frac1)
            else:
                z3d = map_2d_to_3d(new_endpoints1, frac)
                new_endpoints0 = map_3d_to_2d(z3d, el.frac0)
            intersection.Intersection(
                f"{el.label}_part",
                new_endpoints0,
                new_endpoints1,
                el.frac0,
                el.frac1,
                el.ncoef,
                el.nint,
            )
        elif isinstance(el, ConstantHeadLine):
            ConstantHeadLine(
                f"{el.label}_part",
                new_endpoints0,
                el.head,
                el.frac0,
                el.ncoef,
                el.nint,
            )
        elif isinstance(el, ImpermeableLine):
            ImpermeableLine(
                f"{el.label}_part",
                new_endpoints0,
                el.frac0,
                el.ncoef,
                el.nint,
            )

    def split_element_at_point(frac, el, el2, z):
        if el.frac0 == frac:
            z0 = el.endpoints0[0]
            z1 = el.endpoints0[1]
        else:
            z0 = el.endpoints1[0]
            z1 = el.endpoints1[1]
        if el2.frac0 == frac:
            z2 = el2.endpoints0[0]
            z3 = el2.endpoints0[1]
        else:
            z2 = el2.endpoints1[0]
            z3 = el2.endpoints1[1]
        # Split el
        new_el1_endpoints0 = np.array([z0, z])
        new_el2_endpoints0 = np.array([z, z1])
        # Split el2
        new_el1_endpoints1 = np.array([z2, z])
        new_el2_endpoints1 = np.array([z, z3])
        create_new_element(frac, el, new_el1_endpoints0, new_el1_endpoints1)
        create_new_element(frac, el, new_el2_endpoints0, new_el2_endpoints1)
        create_new_element(frac, el2, new_el1_endpoints1, new_el1_endpoints0)
        create_new_element(frac, el2, new_el2_endpoints1, new_el2_endpoints0)

        # Remove old elements
        frac.delete_element(el)
        frac.delete_element(el2)

    for fr in fractures:
        cond = True
        while cond:
            cond = False
            elements = [
                el
                for el in fr.elements
                if isinstance(el, intersection.Intersection)
                or isinstance(el, ConstantHeadLine)
                or isinstance(el, ImpermeableLine)
            ]
            n_elements = len(elements)
            for i in range(n_elements):
                el = elements[i]
                for j in range(i + 1, n_elements):
                    el2 = elements[j]
                    z = check_crossing(el, el2, fr)
                    if z is not False:
                        cond = True
                        split_element_at_point(fr, el, el2, z)
                        break

    print("coming here")


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
    return [
        fr
        for fr in fractures
        if any(isinstance(el, intersection.Intersection) for el in fr.elements)
    ]


def set_head_boundary(
    fractures, ncoef, nint, head, center, radius, normal, label, se_factor, tolerance
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
    tolerance : float
        The minimum length of the intersection line as a fraction of the fracture radius. Intersections shorter than
        this value will be ignored. If -1, no tolerance is applied. Default is -1.

    Returns
    -------
    None
    """
    fracture_surface = andfn.Fracture(label, 1, radius, center, normal, ncoef, nint)
    fr = fracture_surface
    for fr2 in fractures:
        # Quick Squared Distance Check
        dist_sq = np.sum((fr.center - fr2.center) ** 2)
        if dist_sq > (fr.radius + fr2.radius) ** 2:
            continue
        endpoints0, endpoints1 = fracture_intersection(fr, fr2)
        if endpoints0 is not None:
            endpoints = shorten_line(endpoints1, se_factor)
            length = np.linalg.norm(endpoints1[0] - endpoints1[1])
            if length < tolerance * fr2.radius:
                continue
            ConstantHeadLine(f"{label}_{fr2.label}", endpoints, head, fr2, ncoef, nint)


def set_impermeable_boundary(
    fractures, ncoef, nint, center, radius, normal, label, se_factor
):
    """
    Function that sets an impermeable boundary condition on the intersection line between a fracture and a defined
    fracture. The impermeable lines are added to the fractures in the list.

    Parameters
    ----------
    fractures : list
        A list of fractures.
    ncoef : int
        The number of coefficients for the constant head line.
    nint : int
        The number of integration points for the constant head line.
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
            endpoints = shorten_line(endpoints1, se_factor)
            if np.linalg.norm(endpoints0[0] - endpoints0[1]) < 1e-10:
                continue
            ImpermeableLine(f"{label}_{fr2.label}", endpoints, fr2, ncoef, nint)


def shorten_line(endpoints, se_factor):
    """
    Function that shortens a line segment by a given se_factor and keeps the same center point.

    Parameters
    ----------
    endpoints : np.ndarray
        The endpoints of the line segment.
    se_factor : float

    Returns
    -------
    np.ndarray
    """
    z0 = endpoints[0]
    z1 = endpoints[1]
    center = (z0 + z1) / 2
    z0 = center + (z0 - center) * se_factor
    z1 = center + (z1 - center) * se_factor
    return np.array([z0, z1])


def check_connectivity_org(fractures):
    """
    Function that checks the connectivity of a list of fractures. The function returns
    any fractures that are not connected to a constant head boundary.

    Parameters
    ----------
    fractures : list
        A list of fractures.

    Returns
    -------
    bool
        True if all fractures are connected, False otherwise.
    """
    boundary_fracs = []
    for f in fractures:
        for el in f.elements:
            if isinstance(el, (ConstantHeadLine, Well)):
                boundary_fracs.append(f)
                break

    # Get all connected fractures to the boundary fractures
    fracture_list_it = boundary_fracs.copy()
    connected_fractures = boundary_fracs.copy()
    fracture_list_it_temp = []
    fracture_list = [f for f in fractures if f not in connected_fractures]
    while fracture_list_it:
        for i, fr in enumerate(fracture_list_it):
            if any(isinstance(el, intersection.Intersection) for el in fr.elements):
                fracture_list_it_temp.extend(
                    [
                        el.frac1
                        for el in fr.elements
                        if isinstance(el, intersection.Intersection)
                        and el.frac1 in fracture_list
                        and el.frac1 not in fracture_list_it_temp
                    ]
                )
                fracture_list_it_temp.extend(
                    [
                        el.frac0
                        for el in fr.elements
                        if isinstance(el, intersection.Intersection)
                        and el.frac0 in fracture_list
                        and el.frac0 not in fracture_list_it_temp
                    ]
                )

        fracture_list_it = [
            f for f in fracture_list_it_temp if f not in connected_fractures
        ]
        connected_fractures.extend(fracture_list_it)
        fracture_list_it_temp = []
        fracture_list = [f for f in fractures if f not in connected_fractures]

    remove_list = [f for f in fractures if f not in connected_fractures]

    return len(remove_list) == 0, remove_list


def check_connectivity(fractures):
    """
    Function that checks the connectivity of a list of fractures. The function returns
    any fractures that are not connected to a constant head boundary.

    Parameters
    ----------
    fractures : list
        A list of fractures.

    Returns
    -------
    bool
        True if all fractures are connected, False otherwise.
    """
    from collections import deque

    n = len(fractures)

    # Map fractures to integer IDs (major speedup)
    id_of = {f: i for i, f in enumerate(fractures)}

    # Neighbor adjacency list (only intersections)
    neighbors = [[] for _ in range(n)]

    # Connected flag (bitset for speed & memory)
    connected = bytearray(n)

    # BFS queue
    queue = deque()

    # ---------- Single preprocessing pass ----------
    for f in fractures:
        fid = id_of[f]
        has_boundary = False

        for el in f.elements:
            if isinstance(el, intersection.Intersection):
                neighbors[fid].append(id_of[el.frac0])
                neighbors[fid].append(id_of[el.frac1])

            elif isinstance(el, (ConstantHeadLine, Well)):
                has_boundary = True

        # Seed BFS immediately
        if has_boundary:
            connected[fid] = 1
            queue.append(fid)

    # ---------- BFS traversal ----------
    while queue:
        fid = queue.popleft()

        for nbr in neighbors[fid]:
            if not connected[nbr]:
                connected[nbr] = 1
                queue.append(nbr)

    # ---------- Fractures to remove ----------
    remove_list = [fractures[i] for i in range(n) if not connected[i]]

    return len(remove_list) == 0, remove_list


@nb.njit(parallel=True, cache=True)
def count_fracture_adjacency_from_fractures(fractures, elements):
    n_fractures = fractures.size
    counts = np.zeros(n_fractures, dtype=np.int32)

    for i in nb.prange(n_fractures):
        nel = fractures[i]["nelements"]
        el_ids = fractures[i]["elements"][:nel]
        el = elements[el_ids]
        counts[i] = sum(el["_type"] == 0)

    return counts


@nb.njit(cache=True)
def build_indptr(counts):
    n = counts.size
    indptr = np.empty(n + 1, dtype=np.int32)

    s = 0
    indptr[0] = 0
    for i in range(n):
        s += counts[i]
        indptr[i + 1] = s

    return indptr, s  # total adjacency entries


@nb.njit(cache=True)
def fill_fracture_adjacency_from_elements(elements, adj_indptr, adj_indices):
    write_ptr = adj_indptr.copy()
    frac0 = elements["frac0"]
    frac1 = elements["frac1"]
    types = elements["_type"]

    for i in range(elements.size):
        if types[i] != 0:
            continue

        f0 = frac0[i]
        f1 = frac1[i]

        p0 = write_ptr[f0]
        p1 = write_ptr[f1]

        adj_indices[p0] = f1
        adj_indices[p1] = f0

        write_ptr[f0] += 1
        write_ptr[f1] += 1


@nb.njit(cache=True)
def build_fracture_adjacency(fractures_struc_array, elements_struc_array):
    counts = count_fracture_adjacency_from_fractures(
        fractures_struc_array, elements_struc_array
    )

    adj_indptr, total = build_indptr(counts)

    adj_indices = np.empty(total, dtype=np.int32)

    fill_fracture_adjacency_from_elements(
        elements_struc_array,
        adj_indptr,
        adj_indices,
    )

    return adj_indptr, adj_indices


@nb.njit(parallel=True, cache=True)
def compute_boundary_fractures_from_elements(elements, n_fractures):
    """
    Determine which fractures are connected to boundary conditions.

    Parameters
    ----------
    elements : structured array (element_dtype_hpc)
        All DFN elements.
    n_fractures : int
        Total number of fractures.

    Returns
    -------
    boundary : uint8 array of shape (n_fractures,)
        boundary[f] == 1 if fracture f has a boundary condition.
    """
    boundary = np.zeros(n_fractures, dtype=np.uint8)

    for i in nb.prange(elements.size):
        t = elements[i]["_type"]

        # Boundary condition elements
        if t == 2 or t == 3:
            f = elements[i]["frac0"]
            boundary[f] = 1  # safe: idempotent write

    return boundary


@nb.njit(cache=True)
def bfs_connectivity_from_adjacency(adj_indptr, adj_indices, boundary):
    n_fr = boundary.size

    connected = np.zeros(n_fr, dtype=np.uint8)
    queue = np.empty(n_fr, dtype=np.int32)

    q0 = 0
    q1 = 0

    # seed queue and copy boundary
    for i in range(n_fr):
        c = boundary[i]
        connected[i] = c
        if c == 1:
            queue[q1] = i
            q1 += 1

    adj_ptr = adj_indptr
    adj_idx = adj_indices
    conn = connected
    queue_loc = queue

    while q0 < q1:
        f = queue_loc[q0]
        q0 += 1

        start = adj_ptr[f]
        end = adj_ptr[f + 1]

        for k in range(start, end):
            nbr = adj_idx[k]
            if conn[nbr] == 0:
                conn[nbr] = 1
                queue_loc[q1] = nbr
                q1 += 1

    return connected


def check_connectivity_hpc(fractures_struc_array, elements_struc_array):
    """
    Drop-in replacement for check_connectivity using HPC arrays.
    """

    count_fracture_adjacency_from_fractures(
        fractures_struc_array, elements_struc_array
    )  # warmup JIT
    s = time.time()
    adj_indptr, adj_indices = build_fracture_adjacency(
        fractures_struc_array,
        elements_struc_array,
    )
    s1 = time.time()
    boundary = compute_boundary_fractures_from_elements(
        elements_struc_array,
        fractures_struc_array.size,
    )
    s2 = time.time()
    connected = bfs_connectivity_from_adjacency(
        adj_indptr,
        adj_indices,
        boundary,
    )
    print(
        f"Adjacency: {s1 - s:.2f} sec, Boundary: {s2 - s1:.2f} sec, BFS: {time.time() - s2:.2f} sec"
    )

    # fractures to remove (Python side)
    remove_ids = np.flatnonzero(connected == 0).tolist()

    all_connected = len(remove_ids) == 0
    return all_connected, remove_ids


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
