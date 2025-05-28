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
    omega = self_["constant"] + 0.0j

    for e in range(self_["nelements"]):
        el = self_["elements"][e]
        if el != exclude:
            element = element_struc_array[el]
            if element["type_"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega(element, z, self_["id_"])
            elif element["type_"] == 1:  # Bounding circle
                omega += hpc_bounding_circle.calc_omega(element, z)
            elif element["type_"] == 2:  # Well
                omega += hpc_well.calc_omega(element, z)
            elif element["type_"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega(element, z)
            elif element["type_"] == 4:  # Impermeable circle
                omega += hpc_imp_object.calc_omega_circle(element, z)
            elif element["type_"] == 5:  # Impermeable line
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
            if element["type_"] == 0:  # Intersection
                w += hpc_intersection.calc_w(element, z, self_["id_"])
            elif element["type_"] == 1:  # Bounding circle
                w += hpc_bounding_circle.calc_w(element, z)
            elif element["type_"] == 2:  # Well
                w += hpc_well.calc_w(element, z)
            elif element["type_"] == 3:  # Constant head line
                w += hpc_const_head_line.calc_w(element, z)

    return w


@nb.njit(cache=CACHE)
def calc_flow_net(self_, n_points, margin, element_struc_array):
    """
    Calculates the flow net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    n_points : int
        Number of points in the flow net.
    margin : float
        Margin for the flow net.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    flow_net : np.ndarray[complex]
        The flow net for the fracture.
    """
    # Create the z array
    flow_net = np.zeros((n_points, n_points), dtype=np.complex128)
    radius_margin = self_["radius"] * (1 + margin)
    radius2 = self_["radius"] * self_["radius"]
    x_array = np.linspace(-radius_margin, radius_margin, n_points)
    y_array = np.linspace(-radius_margin, radius_margin, n_points)
    for i in range(n_points):
        for j in range(n_points):
            z = x_array[i] + 1j * y_array[j]
            if np.real(z * np.conj(z)) > radius2:
                flow_net[j, i] = np.nan
            else:
                flow_net[j, i] = calc_omega(self_, z, element_struc_array)

    return flow_net, x_array, y_array


def head_from_phi(self_, phi):
    """
    Calculates the head net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    phi : float
        The discharge potential for the fracture.

    Returns
    -------
    head : np.ndarray[complex]
        The head net for the fracture.
    """
    # Create the z array
    head = phi / self_["t"]

    return head


@nb.njit(cache=CACHE)
def calc_heads(self_, n_points, margin, element_struc_array):
    """
    Calculates the head net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    n_points : int
        Number of points in the flow net.
    margin : float
        Margin for the flow net.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    heads : np.ndarray[complex]
        The head net for the fracture.
    """
    # Create the z array
    heads = np.zeros((n_points, n_points), dtype=np.float64)
    radius_margin = self_["radius"] * (1 + margin)
    radius2 = self_["radius"] * self_["radius"]
    x_array = np.linspace(-radius_margin, radius_margin, n_points)
    y_array = np.linspace(-radius_margin, radius_margin, n_points)
    t = self_["t"]
    for i in range(n_points):
        for j in range(n_points):
            z = x_array[i] + 1j * y_array[j]
            if np.real(z * np.conj(z)) > radius2:
                heads[j, i] = np.nan
            else:
                phi = np.real(calc_omega(self_, z, element_struc_array))
                heads[j, i] = phi / t

    return heads, x_array, y_array


@nb.njit(cache=CACHE, parallel=True)
def get_flow_nets(fracture_struc_array, n_points, margin, element_struc_array):
    """
    Get the flow nets for all fractures.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        The fracture structure array.
    n_points : int
        Number of points in the flow net.
    margin : float
        Margin for the flow net.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    flow_nets : list[np.ndarray[complex]]
        List of flow nets for each fracture.
    """
    # Create the flow nets arrays
    flow_nets = np.zeros(
        (len(fracture_struc_array), n_points, n_points), dtype=np.complex128
    )
    x_arrays = np.zeros((len(fracture_struc_array), n_points), dtype=np.float64)
    y_arrays = np.zeros((len(fracture_struc_array), n_points), dtype=np.float64)
    for i in nb.prange(len(fracture_struc_array)):
        flow_net, x_array, y_array = calc_flow_net(
            fracture_struc_array[i], n_points, margin, element_struc_array
        )
        flow_nets[i] = flow_net
        x_arrays[i] = x_array
        y_arrays[i] = y_array

    return flow_nets, x_arrays, y_arrays


@nb.njit(cache=CACHE, parallel=True)
def get_heads(fracture_struc_array, n_points, margin, element_struc_array):
    """
    Get the heads for all fractures.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        The fracture structure array.
    n_points : int
        Number of points in the flow net.
    margin : float
        Margin for the flow net.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    heads : list[np.ndarray[complex]]
        List of heads for each fracture.
    """
    # Create the heads arrays
    heads = np.zeros((len(fracture_struc_array), n_points, n_points), dtype=np.float64)
    x_arrays = np.zeros((len(fracture_struc_array), n_points), dtype=np.float64)
    y_arrays = np.zeros((len(fracture_struc_array), n_points), dtype=np.float64)
    for i in nb.prange(len(fracture_struc_array)):
        head, x_array, y_array = calc_heads(
            fracture_struc_array[i], n_points, margin, element_struc_array
        )
        heads[i] = head
        x_arrays[i] = x_array
        y_arrays[i] = y_array

    return heads, x_arrays, y_arrays
