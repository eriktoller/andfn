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
        heads[i] = phi / self_["t"]


@nb.njit(cache=CACHE, parallel=True)
def get_flow_nets(fracture_struc_array, n_points, n_boundary_points, element_struc_array):
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
    flow_nets = np.zeros(
        (len(fracture_struc_array), n, n), dtype=np.complex128
    )

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
def get_heads(fracture_struc_array, n_points, n_boundary_points, element_struc_array):
    """
    Get the heads for all fractures.

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
    heads : list[np.ndarray[complex]]
        List of heads for each fracture.
    """
    n = n_points + n_boundary_points

    # Create the heads arrays
    heads = np.zeros((len(fracture_struc_array), n), dtype=np.float64)

    # Create the 3D points arrays and its working z arrays
    pnts_3d = np.zeros((len(fracture_struc_array), n, 3), dtype=np.float64)
    z_arrays = np.zeros((len(fracture_struc_array), n), dtype=np.complex128)
    z_array = sunflower_spiral(n_points, n_boundary_points)

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
