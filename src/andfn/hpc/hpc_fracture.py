"""
Notes
-----
This module contains the HPC fracture functions.
"""

import numba as nb
import numpy as np
from numba.extending import overload

from andfn.storage import connectivity_from_hpc, fracture_metadata_from_hpc

from . import (
    CACHE,
    hpc_bounding_circle,
    hpc_const_head_line,
    hpc_imp_object,
    hpc_intersection,
    hpc_well,
)
from . import hpc_geometry_functions as gf


def fracture_element_ids(fracture, ids=None, indptr=None):
    """Compatibility view with compile-time dispatch for compact records."""
    if ids is None:
        return fracture["elements"][: fracture["nelements"]]
    if indptr is None:
        return ids
    i = fracture["_id"]
    start, end = indptr[i], indptr[i + 1]
    return ids[start:end]


@overload(fracture_element_ids)
def _overload_fracture_element_ids(fracture, ids=None, indptr=None):
    # Numba must not type-check legacy fields on a compact metadata record.
    if ids is None or isinstance(ids, (nb.types.NoneType, nb.types.Omitted)):

        def legacy(fracture, ids=None, indptr=None):
            return fracture["elements"][: fracture["nelements"]]

        return legacy
    if indptr is None or isinstance(indptr, (nb.types.NoneType, nb.types.Omitted)):

        def supplied(fracture, ids=None, indptr=None):
            return ids

        return supplied

    def csr(fracture, ids=None, indptr=None):
        i = fracture["_id"]
        start, end = indptr[i], indptr[i + 1]
        return ids[start:end]

    return csr


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
def calc_omega(
    self_,
    z,
    element_struc_array,
    exclude=-1,
    element_ids=None,
    coef_data=None,
    coef_start=None,
):
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
    ids = fracture_element_ids(self_, element_ids)
    for el in ids:
        if el != exclude:
            element = element_struc_array[el]
            if element["_type"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega(
                    element,
                    z,
                    self_["_id"],
                    self_["radius"],
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            elif element["_type"] == 1:  # Bounding circle
                omega += hpc_bounding_circle.calc_omega(
                    element, z, coef_data=coef_data, coef_start=coef_start
                )
            elif element["_type"] == 2:  # Well
                omega += hpc_well.calc_omega(element, z)
            elif element["_type"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega(
                    element,
                    z,
                    self_["radius"],
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            elif element["_type"] == 4:  # Impermeable circle
                omega += hpc_imp_object.calc_omega_circle(
                    element, z, coef_data=coef_data, coef_start=coef_start
                )
            elif element["_type"] == 5:  # Impermeable line
                omega += hpc_imp_object.calc_omega_line(
                    element, z, coef_data=coef_data, coef_start=coef_start
                )
        else:
            element = element_struc_array[el]
            # If the element is the one to exclude, we still need to add  the contribution from the mirror image of the element if it is an intersection or constant head line
            if element["_type"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega(
                    element,
                    z,
                    self_["_id"],
                    self_["radius"],
                    mirror=True,
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            elif element["_type"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega(
                    element,
                    z,
                    self_["radius"],
                    mirror=True,
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
    return omega


@nb.njit()
def calc_omega_error(self_, z, element_struc_array, exclude=-1):
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
    omega = self_["error_constant"] + 0.0j

    # Loop through the elements of the fracture
    for e in range(self_["nelements"]):
        el = self_["elements"][e]
        if el != exclude:
            element = element_struc_array[el]
            if element["_type"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega_error(
                    element, z, self_["_id"], self_["radius"]
                )
            elif element["_type"] == 1:  # Bounding circle
                omega += hpc_bounding_circle.calc_omega_error(element, z)
            elif element["_type"] == 2:  # Well
                omega += hpc_well.calc_omega_error(element, z)
            elif element["_type"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega_error(
                    element, z, self_["radius"]
                )
            elif element["_type"] == 4:  # Impermeable circle
                omega += hpc_imp_object.calc_omega_circle_error(element, z)
            elif element["_type"] == 5:  # Impermeable line
                omega += hpc_imp_object.calc_omega_line_error(element, z)
        else:
            element = element_struc_array[el]
            # If the element is the one to exclude, we still need to add  the contribution from the mirror image of the element if it is an intersection or constant head line
            if element["_type"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega_error(
                    element, z, self_["_id"], self_["radius"], mirror=True
                )
            elif element["_type"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega_error(
                    element, z, self_["radius"], mirror=True
                )
    return omega


@nb.njit()
def calc_omega_error_org(self_, z, error_struc_array, exclude=-1):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    z : complex
        A point in the complex z plane.
    error_struc_array : np.ndarray[element_dtype]
        Array of elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    omega : complex
        The complex potential for the fracture.
    """
    # Initialize omega with the constant value
    omega = self_["error_constant"]

    # Loop through the elements of the fracture
    for e in range(self_["nelements"]):
        el = self_["elements"][e]
        if el != exclude:
            element = error_struc_array[el]
            if element["_type"] == 0:  # Intersection
                omega += hpc_intersection.calc_omega_error(element, z, self_["_id"])
            elif element["_type"] == 1:  # Bounding circle
                omega += hpc_bounding_circle.calc_omega_error(element, z)
            elif element["_type"] == 2:  # Well
                omega += hpc_well.calc_omega_error(element, z)
            elif element["_type"] == 3:  # Constant head line
                omega += hpc_const_head_line.calc_omega_error(element, z)
            elif element["_type"] == 4:  # Impermeable circle
                omega += hpc_imp_object.calc_omega_circle(element, z)
            elif element["_type"] == 5:  # Impermeable line
                omega += hpc_imp_object.calc_omega_line(element, z)
    return omega


@nb.njit()
def calc_omega_array(
    self_,
    omega,
    z,
    element_struc_array,
    exclude=-1,
    element_ids=None,
    coef_data=None,
    coef_start=None,
):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    omega : np.ndarray[np.complex128]
        Array to store the resulting omega values for each point in z.
    z : np.ndarray[np.complex128]
        A point in the complex z plane.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    None
    """
    # Initialize omega with the constant value
    omega[:] = self_["constant"] + 0.0j

    # Loop through the elements of the fracture
    ids = fracture_element_ids(self_, element_ids)
    for el in ids:
        if el != exclude:
            element = element_struc_array[el]
            if element["_type"] == 0:  # Intersection
                hpc_intersection.calc_omega_array(
                    element,
                    omega,
                    z,
                    self_["_id"],
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            elif element["_type"] == 1:  # Bounding circle
                hpc_bounding_circle.calc_omega_array(
                    element, omega, z, coef_data=coef_data, coef_start=coef_start
                )
            elif element["_type"] == 2:  # Well
                hpc_well.calc_omega_array(element, omega, z)
            elif element["_type"] == 3:  # Constant head line
                hpc_const_head_line.calc_omega_array(
                    element, omega, z, coef_data=coef_data, coef_start=coef_start
                )
            elif element["_type"] == 4:  # Impermeable circle
                hpc_imp_object.calc_omega_circle_array(
                    element, omega, z, coef_data=coef_data, coef_start=coef_start
                )
            elif element["_type"] == 5:  # Impermeable line
                hpc_imp_object.calc_omega_line_array(
                    element, omega, z, coef_data=coef_data, coef_start=coef_start
                )


@nb.njit()
def calc_w(
    self_,
    z,
    element_struc_array,
    exclude=-1,
    element_ids=None,
    coef_data=None,
    coef_start=None,
):
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

    ids = fracture_element_ids(self_, element_ids)
    for el in ids:
        if el != exclude:
            element = element_struc_array[el]
            if element["_type"] == 0:  # Intersection
                w += hpc_intersection.calc_w(
                    element,
                    z,
                    self_["_id"],
                    self_["radius"],
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            elif element["_type"] == 1:  # Bounding circle
                w += hpc_bounding_circle.calc_w(
                    element, z, coef_data=coef_data, coef_start=coef_start
                )
            elif element["_type"] == 2:  # Well
                w += hpc_well.calc_w(element, z)
            elif element["_type"] == 3:  # Constant head line
                w += hpc_const_head_line.calc_w(
                    element,
                    z,
                    self_["radius"],
                    coef_data=coef_data,
                    coef_start=coef_start,
                )

    return w


@nb.njit(cache=CACHE)
def calc_omega_csr(
    fracture,
    z,
    fractures,
    elements,
    ids,
    indptr,
    exclude=-1,
    coef_data=None,
    coef_start=None,
):
    """CSR entry point; all arguments are numeric arrays or scalars."""
    start, end = indptr[fracture], indptr[fracture + 1]
    return calc_omega(
        fractures[fracture],
        z,
        elements,
        exclude,
        ids[start:end],
        coef_data=coef_data,
        coef_start=coef_start,
    )


@nb.njit(cache=CACHE)
def calc_w_csr(
    fracture,
    z,
    fractures,
    elements,
    ids,
    indptr,
    exclude=-1,
    coef_data=None,
    coef_start=None,
):
    start, end = indptr[fracture], indptr[fracture + 1]
    return calc_w(
        fractures[fracture],
        z,
        elements,
        exclude,
        ids[start:end],
        coef_data=coef_data,
        coef_start=coef_start,
    )


@nb.njit(cache=CACHE, parallel=True)
def get_heads_csr(
    fractures,
    elements,
    ids,
    indptr,
    z_array,
    fracture_indices=None,
    coef_data=None,
    coef_start=None,
):
    """Evaluate the same scaled fracture points as get_heads, without padding."""
    size = fractures.size if fracture_indices is None else fracture_indices.size
    heads = np.zeros((size, z_array.size), dtype=np.float64)
    points = np.zeros((size, z_array.size, 3), dtype=np.float64)
    for row in nb.prange(size):
        i = np.int64(row) if fracture_indices is None else fracture_indices[row]
        start, end = indptr[i], indptr[i + 1]
        element_ids = ids[start:end]
        z = z_array * fractures[i]["radius"]
        for j in range(z_array.size):
            heads[row, j] = head_from_phi(
                fractures[i],
                np.real(
                    calc_omega(
                        fractures[i],
                        z[j],
                        elements,
                        -1,
                        element_ids,
                        coef_data=coef_data,
                        coef_start=coef_start,
                    )
                ),
            )
        gf.map_2d_to_3d(fractures[i], z, points[row])
    return heads, points


def get_heads_from_hpc(fractures, elements, z_array, fracture_indices=None):
    """Python compatibility boundary; build CSR/metadata once per evaluation."""
    if fracture_indices is not None and (
        not isinstance(fracture_indices, np.ndarray)
        or fracture_indices.ndim != 1
        or fracture_indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or not fracture_indices.flags.c_contiguous
        or np.any(fracture_indices < 0)
        or np.any(fracture_indices >= fractures.size)
    ):
        raise ValueError("Invalid fracture indices")
    connectivity = connectivity_from_hpc(fractures, elements.size)
    metadata = fracture_metadata_from_hpc(fractures)
    return get_heads_csr(
        metadata,
        elements,
        connectivity.ids,
        connectivity.indptr,
        z_array,
        fracture_indices,
    )


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


@nb.njit(cache=CACHE)
def calc_errors(self_, errors, n_points, z_array, error_struc_array):
    """
    Calculates the head net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    errors : np.ndarray[np.complex128]
        Array to store the head net for the fracture.
    n_points : int
        Number of points in the flow net.
    z_array : np.ndarray[np.complex128]
        Array of complex coordinates for the points in the sunflower spiral multiplied with the fracture radius.
    error_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    None
         Modifies the heads array in place.
    """
    # Calculate the head net for the fracture
    for i in range(n_points):
        errors[i] = calc_omega_error(self_, z_array[i], error_struc_array)


@nb.njit(cache=CACHE)
def calc_error_heads(self_, errors, n_points, z_array, error_struc_array):
    """
    Calculates the head net for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[fracture_dtype]
        The fracture element.
    errors : np.ndarray[np.complex128]
        Array to store the head net for the fracture.
    n_points : int
        Number of points in the flow net.
    z_array : np.ndarray[np.complex128]
        Array of complex coordinates for the points in the sunflower spiral multiplied with the fracture radius.
    error_struc_array : np.ndarray[element_dtype]
        Array of elements.

    Returns
    -------
    None
         Modifies the heads array in place.
    """
    # Calculate the head net for the fracture
    for i in range(n_points):
        error = np.real(calc_omega_error(self_, z_array[i], error_struc_array))
        errors[i] = head_from_phi(self_, error)


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


@nb.njit(cache=CACHE, parallel=True)
def get_errors(fracture_struc_array, error_struc_array, z_array, heads=False):
    """
    Get the heads for all fractures.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        The fracture structure array.
    error_struc_array : np.ndarray[element_dtype]
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
    errors = np.zeros((len(fracture_struc_array), n), dtype=np.complex128)

    # Create the 3D points arrays and its working z arrays
    pnts_3d = np.zeros((len(fracture_struc_array), n, 3), dtype=np.float64)
    z_arrays = np.zeros((len(fracture_struc_array), n), dtype=np.complex128)

    # Calculate the errors
    if heads:
        for i in nb.prange(len(fracture_struc_array)):
            z_arrays[i] = z_array * fracture_struc_array[i]["radius"]
            calc_error_heads(
                fracture_struc_array[i],
                errors[i],
                n,
                z_arrays[i],
                error_struc_array,
            )
            # Map the 2D points to 3D
            gf.map_2d_to_3d(fracture_struc_array[i], z_arrays[i], pnts_3d[i])
    else:
        for i in nb.prange(len(fracture_struc_array)):
            z_arrays[i] = z_array * fracture_struc_array[i]["radius"]
            calc_errors(
                fracture_struc_array[i],
                errors[i],
                n,
                z_arrays[i],
                error_struc_array,
            )
            # Map the 2D points to 3D
            gf.map_2d_to_3d(fracture_struc_array[i], z_arrays[i], pnts_3d[i])

    return errors, pnts_3d
