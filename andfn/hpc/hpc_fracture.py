"""
Notes
-----
This module contains the HPC fracture functions.
"""

import numpy as np
import numba as nb
from andfn.hpc import (
    hpc_intersection,
    hpc_const_head_line,
    hpc_well,
    hpc_bounding_circle,
    hpc_imp_object,
    CACHE,
)
from andfn.hpc import hpc_geometry_functions as gf


@nb.njit(cache=CACHE)
def sunflower_spiral(n_in, n_bnd):
    """
    Generate n points in a sunflower spiral pattern.

    Parameters
    ----------
    n_in : int
        Number of points to generate.
    n_bnd : int
        Number of boundary points to add along the unit circle.

    Returns
    -------
    z : np.ndarray[np.complex128]
        An array of shape (n,) containing the complex coordinates of the points in the sunflower spiral.
    """
    indices = np.arange(0, n_in, dtype=np.float64) + 0.5

    r = np.sqrt(indices / n_in)
    theta = np.pi * (1 + 5**0.5) * indices

    # Convert polar coordinates to complex numbers
    z = r * np.cos(theta) + 1j * r * np.sin(theta)

    # Add points along the boundary of the unit circle
    z = np.concatenate((z, np.exp(1j * np.linspace(0, 2 * np.pi, n_bnd))))

    return z


@nb.njit()
def calc_omega(self_, z, element_struc_array, exclude=-1):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    z : complex
        A point in the complex z plane.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    omega : complex
        The complex potential for the fracture.
    """
    # Initialize omega with the constant value
    omega = self_["constant"] + 0.0j

    # Loop through the elements of the fracture
    for e in range(self_["nelements"]):
        el = self_["elements"][e]
        if el != exclude:
            element = element_struc_array[el]
            if element["_type"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega(element, z, self_["_id"])
            elif element["_type"] == 1:  # Bounding circle
                omega += hpc_bounding_circle.calc_omega(element, z)
            elif element["_type"] == 2:  # Well
                omega += hpc_well.calc_omega(element, z)
            elif element["_type"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega(element, z)
            elif element["_type"] == 4:  # Impermeable circle
                omega += hpc_imp_object.calc_omega_circle(element, z)
            elif element["_type"] == 5:  # Impermeable line
                omega += hpc_imp_object.calc_omega_line(element, z)
    return omega


@nb.njit()
def calc_omega_array(
    omega,
    z,
    constant,
    nelements,
    fid,
    element_idx,
    frac0,
    ep0a,
    ep0b,
    ep1a,
    ep1b,
    coef,
    ncoef,
    q,
    radius,
    center,
    etype,
    exclude=-1,
):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    omega : np.ndarray[np.complex128]
        Array to store the resulting omega values for each point in z.
    z : np.ndarray[np.complex128]
        A list of points in the complex z plane.
    constant : float
        The constant value for the fracture.
    nelements : int
        Number of elements in the fracture.
    fid : int
        Fracture ID.
    element_idx : np.ndarray[int]
        Array of element indices for the fracture.
    frac0 : np.ndarray[float]
        Array of frac0 values for the elements.
    ep0a : np.ndarray[float]
        Array of ep0a values for the elements.
    ep0b : np.ndarray[float]
        Array of ep0b values for the elements.
    ep1a : np.ndarray[float]
        Array of ep1a values for the elements.
    ep1b : np.ndarray[float]
        Array of ep1b values for the elements.
    coef : np.ndarray[float]
        Array of coef values for the elements.
    ncoef : np.ndarray[int]
        Array of ncoef values for the elements.
    q : np.ndarray[float]
        Array of q values for the elements.
    radius : np.ndarray[float]
        Array of radius values for the elements.
    center : np.ndarray[float]
        Array of center values for the elements.
    etype : np.ndarray[int]
        Array of element types for the elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    None
    """

    # Split once
    idx0, idx1, idx2, idx3, idx4, idx5 = split_elements_by_type(
        element_idx, etype, nelements, exclude
    )

    # Initialize omega with the constant value
    omega[:] = constant + 0.0j

    # ---- Intersections (etype 0)
    for k in range(idx0.size):
        el = idx0[k]
        hpc_intersection.calc_omega_array(
            omega,
            z,
            fid,
            frac0[el],
            ep0a[el],
            ep0b[el],
            ep1a[el],
            ep1b[el],
            coef[el],
            ncoef[el],
            q[el],
        )

    # ---- Bounding circles (etype 1)
    for k in range(idx1.size):
        el = idx1[k]
        hpc_bounding_circle.calc_omega_array(omega, z, radius[el], coef[el], ncoef[el])

    # ---- Wells (etype 2)
    for k in range(idx2.size):
        el = idx2[k]
        hpc_well.calc_omega_array(omega, z, radius[el], center[el], q[el])

    # ---- Constant head lines (etype 3)
    for k in range(idx3.size):
        el = idx3[k]
        hpc_const_head_line.calc_omega_array(
            omega, z, ep0a[el], ep0b[el], coef[el], ncoef[el], q[el]
        )

    # ---- Impermeable circles (etype 4)
    for k in range(idx4.size):
        el = idx4[k]
        hpc_imp_object.calc_omega_circle_array(
            omega, z, radius[el], center[el], coef[el], ncoef[el]
        )

    # ---- Impermeable lines (etype 5)
    for k in range(idx5.size):
        el = idx5[k]
        hpc_imp_object.calc_omega_line_array(
            omega, z, ep0a[el], ep0b[el], coef[el], ncoef[el]
        )


@nb.njit()
def calc_omega_mean(
    z,
    constant,
    nelements,
    fid,
    element_idx,
    frac0,
    ep0a,
    ep0b,
    ep1a,
    ep1b,
    coef,
    ncoef,
    q,
    radius,
    center,
    etype,
    exclude=-1,
):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    z : np.ndarray[np.complex128]
        A list of points in the complex z plane.
    constant : float
        The constant value for the fracture.
    nelements : int
        Number of elements in the fracture.
    fid : int
        Fracture ID.
    element_idx : np.ndarray[int]
        Array of element indices for the fracture.
    frac0 : np.ndarray[float]
        Array of frac0 values for the elements.
    ep0a : np.ndarray[float]
        Array of ep0a values for the elements.
    ep0b : np.ndarray[float]
        Array of ep0b values for the elements.
    ep1a : np.ndarray[float]
        Array of ep1a values for the elements.
    ep1b : np.ndarray[float]
        Array of ep1b values for the elements.
    coef : np.ndarray[float]
        Array of coef values for the elements.
    ncoef : np.ndarray[int]
        Array of ncoef values for the elements.
    q : np.ndarray[float]
        Array of q values for the elements.
    radius : np.ndarray[float]
        Array of radius values for the elements.
    center : np.ndarray[float]
        Array of center values for the elements.
    etype : np.ndarray[int]
        Array of element types for the elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    None
    """

    # Split once
    idx0, idx1, idx2, idx3, idx4, idx5 = split_elements_by_type(
        element_idx, etype, nelements, exclude
    )

    # Initialize omega with the constant value
    omega = constant + 0.0j

    # ---- Intersections (etype 0)
    for k in range(idx0.size):
        el = idx0[k]
        omega += hpc_intersection.calc_omega_sum(
            z,
            fid,
            frac0[el],
            ep0a[el],
            ep0b[el],
            ep1a[el],
            ep1b[el],
            coef[el],
            ncoef[el],
            q[el],
        )

    # ---- Bounding circles (etype 1)
    for k in range(idx1.size):
        el = idx1[k]
        omega += hpc_bounding_circle.calc_omega_sum(z, radius[el], coef[el], ncoef[el])

    # ---- Wells (etype 2)
    for k in range(idx2.size):
        el = idx2[k]
        omega += hpc_well.calc_omega_sum(z, radius[el], center[el], q[el])

    # ---- Constant head lines (etype 3)
    for k in range(idx3.size):
        el = idx3[k]
        omega += hpc_const_head_line.calc_omega_sum(
            z, ep0a[el], ep0b[el], coef[el], ncoef[el], q[el]
        )

    # ---- Impermeable circles (etype 4)
    for k in range(idx4.size):
        el = idx4[k]
        omega += hpc_imp_object.calc_omega_circle_sum(
            z, radius[el], center[el], coef[el], ncoef[el]
        )

    # ---- Impermeable lines (etype 5)
    for k in range(idx5.size):
        el = idx5[k]
        omega += hpc_imp_object.calc_omega_line_sum(
            z, ep0a[el], ep0b[el], coef[el], ncoef[el]
        )

    omega_mean = omega / z.size
    return omega_mean


def calc_w(self_, z, element_struc_array, exclude=-1):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    z : complex
        A point in the complex z plane.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    w : complex
        The complex potential for the fracture.
    """
    w = 0.0 + 0.0j

    for e in range(self_["nelements"]):
        el = self_["elements"][e]
        if el != exclude:
            element = element_struc_array[el]
            if element["_type"] == 0:  # Intersection
                w += hpc_intersection.calc_w(element, z, self_["_id"])
            elif element["_type"] == 1:  # Bounding circle
                w += hpc_bounding_circle.calc_w(element, z)
            elif element["_type"] == 2:  # Well
                w += hpc_well.calc_w(element, z)
            elif element["_type"] == 3:  # Constant head line
                w += hpc_const_head_line.calc_w(element, z)

    return w


@nb.njit(cache=CACHE)
def calc_flow_net(self_, flow_net, n_points, z_array, element_struc_array):
    """
    Calculates the flow net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    flow_net : np.ndarray[complex]
        The flow net to be calculated.
    n_points : int
        Number of points in the flow net.
    z_array : np.ndarray[np.complex128]
        Array of complex coordinates for the points in the sunflower spiral multiplied with the fracture radius.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    flow_net : np.ndarray[complex]
        The flow net for the fracture.
    """
    for i in range(n_points):
        flow_net[i] = calc_omega(self_, z_array[i], element_struc_array)


@nb.njit(cache=CACHE)
def calc_heads(self_, heads, n_points, z_array, element_struc_array):
    """
    Calculates the head net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    heads : np.ndarray[np.float64]
        Array to store the head net for the fracture.
    n_points : int
        Number of points in the flow net.
    z_array : np.ndarray[np.complex128]
        Array of complex coordinates for the points in the sunflower spiral multiplied with the fracture radius.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    None
         Modifies the heads array in place.
    """
    # Calculate the head net for the fracture
    for i in range(n_points):
        phi = np.real(calc_omega(self_, z_array[i], element_struc_array))
        heads[i] = head_from_phi(self_, phi)


@nb.njit(inline="always")
def head_from_phi(self_, phi):
    """
    Calculate the head from the potential.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    phi : float
        The potential.

    Returns
    -------
    head : float
        The head.
    """
    return phi / self_["t"]


@nb.njit(cache=CACHE, parallel=True)
def get_flow_nets(
    fracture_struc_array, n_points, n_boundary_points, element_struc_array
):
    """
    Get the flow nets for all fractures.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        The fracture structure array.
    n_points : int
        Number of points in the flow net.
    n_boundary_points : int
        Number of points along the boundary of the unit circle.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    flow_nets : list[np.ndarray[complex]]
        List of flow nets for each fracture.
    """
    n = n_points + n_boundary_points

    # Create the flow nets arrays
    flow_nets = np.zeros((len(fracture_struc_array), n, n), dtype=np.complex128)

    # Create the 3D points arrays and its working z arrays
    pnts_3d = np.zeros((len(fracture_struc_array), n, 3), dtype=np.float64)
    z_arrays = np.zeros((len(fracture_struc_array), n), dtype=np.complex128)
    z_array = sunflower_spiral(n_points, n_boundary_points)

    # Calculate the flow nets for each fracture
    for i in nb.prange(len(fracture_struc_array)):
        z_arrays[i] = z_array * fracture_struc_array[i]["radius"]
        calc_flow_net(
            fracture_struc_array[i],
            flow_nets[i],
            n,
            z_arrays[i],
            element_struc_array,
        )
        # Map the 2D points to 3D
        gf.map_2d_to_3d(fracture_struc_array[i], z_arrays[i], pnts_3d[i])

    return flow_nets, pnts_3d


@nb.njit(cache=CACHE, parallel=True)
def get_heads(fracture_struc_array, element_struc_array, z_array):
    """
    Get the heads for all fractures.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        The fracture structure array.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.
    z_array : np.ndarray[np.complex128]
        Array of complex coordinates for the points in the disk.

    Returns
    -------
    heads : list[np.ndarray[complex]]
        List of heads for each fracture.
    """
    n = len(z_array)

    # Create the heads arrays
    heads = np.zeros((len(fracture_struc_array), n), dtype=np.float64)

    # Create the 3D points arrays and its working z arrays
    pnts_3d = np.zeros((len(fracture_struc_array), n, 3), dtype=np.float64)
    z_arrays = np.zeros((len(fracture_struc_array), n), dtype=np.complex128)

    # Calculate the heads for each fracture
    for i in nb.prange(len(fracture_struc_array)):
        z_arrays[i] = z_array * fracture_struc_array[i]["radius"]
        calc_heads(
            fracture_struc_array[i],
            heads[i],
            n,
            z_arrays[i],
            element_struc_array,
        )
        # Map the 2D points to 3D
        gf.map_2d_to_3d(fracture_struc_array[i], z_arrays[i], pnts_3d[i])

    return heads, pnts_3d


@nb.njit()
def split_elements_by_type(element_idx, etype, nelements, exclude):
    n0 = n1 = n2 = n3 = n4 = n5 = 0

    # First pass: count
    for i in range(nelements):
        el = element_idx[i]
        if el == exclude:
            continue
        t = etype[el]
        if t == 0:
            n0 += 1
        elif t == 1:
            n1 += 1
        elif t == 2:
            n2 += 1
        elif t == 3:
            n3 += 1
        elif t == 4:
            n4 += 1
        elif t == 5:
            n5 += 1

    idx0 = np.empty(n0, np.int64)
    idx1 = np.empty(n1, np.int64)
    idx2 = np.empty(n2, np.int64)
    idx3 = np.empty(n3, np.int64)
    idx4 = np.empty(n4, np.int64)
    idx5 = np.empty(n5, np.int64)

    i0 = i1 = i2 = i3 = i4 = i5 = 0

    # Second pass: fill
    for i in range(nelements):
        el = element_idx[i]
        if el == exclude:
            continue
        t = etype[el]
        if t == 0:
            idx0[i0] = el
            i0 += 1
        elif t == 1:
            idx1[i1] = el
            i1 += 1
        elif t == 2:
            idx2[i2] = el
            i2 += 1
        elif t == 3:
            idx3[i3] = el
            i3 += 1
        elif t == 4:
            idx4[i4] = el
            i4 += 1
        elif t == 5:
            idx5[i5] = el
            i5 += 1

    return idx0, idx1, idx2, idx3, idx4, idx5
