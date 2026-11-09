"""
Notes
-----
This module contains the HPC solve functions.
"""

import logging
import time

import numba as nb
import numpy as np
import scipy as sp

from andfn.storage import Connectivity

from . import (
    CACHE,
    PARALLEL,
    hpc_bounding_circle,
    hpc_const_head_line,
    hpc_fracture,
    hpc_imp_object,
    hpc_intersection,
    hpc_well,
)
from . import hpc_math_functions as mf
from .hpc_coefficients import candidate_at, coefficient_view
from .hpc_quadrature import allocate_quadrature, quadrature_row, validate_quadrature
from .hpc_work import boundary_work, integration_view, work_record

logger = logging.getLogger(__name__)


def solve(
    fracture_arrays,
    discharge_int,
    constants,
    *,
    adaptive_owner,
    iteration_callback=None,
):
    """Run the adaptive solver over compact coefficient and fracture arrays."""
    if adaptive_owner is None:
        raise ValueError("Adaptive solving requires compact owner state")
    storage = adaptive_owner.state.storage
    element_struc_array = storage.elements
    if not element_struc_array.size or not fracture_arrays.size:
        raise ValueError("Solving requires at least one element and fracture")
    max_error = constants["MAX_ERROR"]
    max_iterations = constants["MAX_ITERATIONS"]
    coef_ratio = constants["COEF_RATIO"]
    max_coef = constants["MAX_COEFFICIENT_ORDER"]
    coef_increase = constants["COEF_INCREASE"]
    damping = float(constants["DAMPING"])

    logger.info("Compiling HPC code...")
    discharge_indices = get_discharge_indices(element_struc_array)
    connectivity_ids, connectivity_indptr = fracture_arrays.ids, fracture_arrays.indptr
    work_array = adaptive_owner.state.work
    num_elements = element_struc_array.size
    size = discharge_indices.size + fracture_arrays.size
    head_matrix = np.zeros(size)
    discharges = get_old_discharges(
        element_struc_array,
        fracture_arrays,
        element_struc_array,
        discharge_indices,
    )
    discharges_old = np.zeros(size)
    z_int = allocate_quadrature(discharge_indices.size, discharge_int)
    get_z_int_array(z_int, element_struc_array, discharge_int, discharge_indices)

    # Discharge matrix
    logger.info("Building discharge matrix...")
    startdm = time.time()
    discharge_matrix = build_discharge_matrix_csr(
        fracture_arrays,
        element_struc_array,
        connectivity_ids,
        connectivity_indptr,
        discharge_indices,
        discharge_int,
        z_int,
    )
    logger.debug(f"Discharge matrix build time: {time.time() - startdm}")

    # LU-factorization
    startlu = time.time()
    lu_matrix = sp.sparse.linalg.splu(discharge_matrix)
    logger.debug(f"LU factorization time: {time.time() - startlu}")

    # Set old error
    for i in nb.prange(num_elements):
        e = element_struc_array[i]
        e["error_old"] = 1e30
        e["error"] = 1e30

    # fill work array ex_array
    for i, e in enumerate(element_struc_array):
        n = e["nint"]
        record = work_record(work_array, i)
        thetas = integration_view(e, record, "thetas")
        mf.calc_thetas(n, e["_type"], thetas[:n])
        mf.fill_exp_array(n, thetas, record["exp_array_m"], -1)
        mf.fill_exp_array(n, thetas, record["exp_array_p"], 1)
        mf.fill_z_integral(e, record)

    logger.info(f"Number of elements: {len(element_struc_array)}")
    logger.info(f"Number of fractures: {fracture_arrays.size}")
    logger.info(f"Number of entries in discharge matrix: {discharge_matrix.getnnz()}")

    error = np.float64(1.0)
    error_coef = np.float64(1.0)
    nit = 0
    cnt_error = 0
    cnt_bnd = 0
    error_q = 1.0
    start = time.time()
    sum_timee = 0.0
    sum_timeq = 0.0
    while cnt_error < 2 and nit < max_iterations:
        nit += 1
        # Solve the discharge matrix
        startq = time.time()
        if error_q > max_error / 1e30:
            discharges_old[:] = discharges[:]
            solve_discharge_matrix(
                fracture_arrays,
                element_struc_array,
                element_struc_array,
                discharge_int,
                head_matrix,
                discharges,
                discharges_old,
                z_int,
                lu_matrix,
                connectivity_ids,
                connectivity_indptr,
                discharge_indices,
                storage.coef_data,
                storage.coef_start,
            )
            error_q = np.max(np.abs(discharges - discharges_old))
            error_q = np.mean(np.abs(discharges - discharges_old))
            error_q = np.max(
                np.abs(discharges - discharges_old)
                / (np.max(np.abs(discharges_old)) + 1e-16)
            )
        timeq = time.time() - startq

        # Solve the elements
        starte = time.time()
        from .hpc_adaptive import adaptive_step

        adaptive_step(
            adaptive_owner,
            fracture_arrays,
            max_error,
            nit,
            cnt_error,
            coef_ratio,
            max_coef,
            coef_increase,
            damping,
        )
        storage = adaptive_owner.state.storage
        element_struc_array = storage.elements
        work_array = adaptive_owner.state.work
        cnt_set_zero = 0
        timee = time.time() - starte
        sum_timee += timee
        sum_timeq += timeq

        error, _id = get_error(element_struc_array)
        error_coef = np.max(element_struc_array["error_coef"])

        # print info
        if nit < 10:
            logger.info(
                f"Iteration: 0{nit}, Error E: {error:.3e}, Coef: {error_coef:.3e}, Q: {error_q:.3e}, Element: {_id}, N of Coef: {element_struc_array[_id]['ncoef']}, Type: {element_struc_array[_id]['_type']}"
            )
            logger.debug(
                f"Solve time {(timee + timeq):.2f} sec (E: {timee:.2f}, Q: {timeq:.2f}), "
                f"Elements set to zero: {cnt_set_zero}"
            )
        else:
            logger.info(
                f"Iteration: {nit}, Error E: {error:.3e}, Coef: {error_coef:.3e}, Q: {error_q:.3e}, Element: {_id}, N of Coef: {element_struc_array[_id]['ncoef']}, Type: {element_struc_array[_id]['_type']}"
            )
            logger.debug(
                f"Solve time {(timee + timeq):.2f} sec (E: {timee:.2f}, Q: {timeq:.2f}), "
                f"Elements set to zero: {cnt_set_zero}"
            )
        if cnt_bnd > 1 or nit == max_iterations:
            element_solver(
                num_elements,
                element_struc_array,
                fracture_arrays,
                work_array,
                max_error,
                nit,
                cnt_error,
                damping,
                storage.coef_data,
                storage.coef_start,
                storage.candidate_coef_data,
            )
            # error = 1.0

        if error_coef < max_error and error_q < max_error:
            cnt_error += 1
            error_q = 1e30
            # error = 1.0
        if iteration_callback is not None:
            iteration_callback(
                nit,
                element_struc_array,
                fracture_arrays,
                work_array,
                discharges,
                error_q,
                cnt_error,
            )

    # Print the solver results
    logger.info("---------------------------------------")
    logger.info("Solver results")
    logger.info("---------------------------------------")
    logger.info(f"Iterations: {nit}, Error E: {error:.3e}, Q: {error_q:.3e}, ")
    logger.debug(f"Total element solve time: {sum_timee:.2f} sec")
    logger.debug(f"Total matrix solve time: {sum_timeq:.2f} sec")
    solve_time = time.time() - start
    days, rem = divmod(solve_time, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(
        f"Solve time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds\n"
    )

    return element_struc_array, work_array


@nb.njit(cache=CACHE)
def get_error(element_struc_array):
    """
    Get the maximum error from the elements and the element that it is associated with.

    Parameters
    ----------
    element_struc_array : np.ndarray[element_dtype]
        The array of elements

    Returns
    -------
    error : float
        The maximum error
    _id : int
        The id of the element with the maximum error
    """
    error = np.max(element_struc_array["error"])
    _id = np.argmax(element_struc_array["error"])
    return error, _id


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_discharge_elements(element_struc_array):
    """
    Get the discharge elements from the element array.

    Parameters
    ----------
    element_struc_array : np.ndarray[element_dtype]
        The array of elements

    Returns
    -------
    discharge_elements : np.ndarray[element_dtype]
        The array of discharge elements
    """
    # get the discharge elements
    el = np.zeros(len(element_struc_array), dtype=np.bool_)
    for i in nb.prange(len(element_struc_array)):
        if element_struc_array[i]["_type"] in {
            0,
            2,
            3,
        }:  # Intersection, Well, Constant head line
            el[i] = 1
    discharge_elements = element_struc_array[el]
    return discharge_elements


@nb.njit(cache=CACHE)
def get_discharge_indices(elements):
    """Ordered array positions; unlike the legacy API, no records are copied."""
    count = 0
    for e in elements:
        if e["_type"] in (0, 2, 3):
            count += 1
    indices = np.empty(count, dtype=np.int64)
    j = 0
    for i in range(elements.size):
        if elements[i]["_type"] in (0, 2, 3):
            indices[j] = i
            j += 1
    return indices


@nb.njit(parallel=PARALLEL, cache=CACHE)
def calculate_candidates(
    num_elements,
    element_struc_array,
    fracture_struc_array,
    work_array,
    max_error,
    nit,
    cnt_error,
    damping,
    coef_data=None,
    coef_start=None,
    candidate_coef_data=None,
    selected=None,
):
    """
    Calculate candidates and convergence metadata without changing current coefficients.

    Parameters
    ----------
    num_elements : int
        The number of elements
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    work_array : WorkArrays
        Compact worker-owned solver state.
    max_error : float
        The maximum error
    nit : int
        The number of iterations
    cnt_error : int
        The number of completed iterations
    damping : float
        The damping factor for the solver (default 0.5)

    Returns
    -------
    cnt : int
        The number of elements that were solved

    """

    cnt = 0

    # Solve the elements
    for i in nb.prange(num_elements):
        if selected is not None and not selected[i]:
            continue
        e = element_struc_array[i]
        if e["error"] < max_error * 0 and nit > 30 and cnt_error == 0:
            cnt += 1
            continue
        record = work_record(work_array, i)
        if e["_type"] == 0:  # Intersection
            hpc_intersection.solve(
                e,
                fracture_struc_array,
                element_struc_array,
                record,
                coef_data,
                coef_start,
                candidate_coef_data,
            )
        elif e["_type"] == 1:  # Bounding circle
            hpc_bounding_circle.solve(
                e,
                fracture_struc_array,
                element_struc_array,
                record,
                coef_data,
                coef_start,
                candidate_coef_data,
            )
        elif e["_type"] == 2:  # Well
            e["error"] = 0.0
            cnt += 1
        elif e["_type"] == 3:  # Constant head line
            hpc_const_head_line.solve(
                e,
                fracture_struc_array,
                element_struc_array,
                record,
                coef_data,
                coef_start,
                candidate_coef_data,
            )
        elif e["_type"] == 4:  # Impermeable circle
            hpc_imp_object.solve_circle(
                e,
                fracture_struc_array,
                element_struc_array,
                record,
                coef_data,
                coef_start,
                candidate_coef_data,
            )
        elif e["_type"] == 5:  # Impermeable line
            hpc_imp_object.solve_line(
                e,
                fracture_struc_array,
                element_struc_array,
                record,
                coef_data,
                coef_start,
                candidate_coef_data,
            )

    return cnt


@nb.njit(parallel=PARALLEL, cache=CACHE)
def commit_coefficients(
    elements, work, damping, coef_data=None, coef_start=None, candidate_coef_data=None
):
    for i in nb.prange(elements.size):
        e = elements[i]
        current = coefficient_view(e, coef_data, coef_start)
        candidate = candidate_at(e, work, i, candidate_coef_data, coef_start)
        current[:] = damping * candidate + (1 - damping) * current


@nb.njit(cache=CACHE)
def element_solver(
    num_elements,
    element_struc_array,
    fracture_struc_array,
    work_array,
    max_error,
    nit,
    cnt_error,
    damping,
    coef_data=None,
    coef_start=None,
    candidate_coef_data=None,
):
    cnt = calculate_candidates(
        num_elements,
        element_struc_array,
        fracture_struc_array,
        work_array,
        max_error,
        nit,
        cnt_error,
        damping,
        coef_data,
        coef_start,
        candidate_coef_data,
    )
    commit_coefficients(
        element_struc_array,
        work_array,
        damping,
        coef_data,
        coef_start,
        candidate_coef_data,
    )
    return cnt


def solve_discharge_matrix(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    discharges,
    discharges_old,
    z_int,
    lu_matrix,
    connectivity_ids=None,
    connectivity_indptr=None,
    discharge_indices=None,
    coef_data=None,
    coef_start=None,
):
    """
    Solves the discharge matrix for the DFN and stores the discharges and constants in the elements and fractures.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_elements : np.ndarray[element_dtype]
        The discharge elements
    discharge_int : int
        The number of integration points
    head_matrix : np.ndarray[dtype_head_matrix]
        The head matrix
    discharges : np.ndarray
        The discharges to be solved
    discharges_old : np.ndarray
        The old discharges
    z_int : np.ndarray
        Selected-row quadrature, dense (2, selected, count).
    lu_matrix : scipy.sparse.linalg.splu
        The LU factorization of the discharge matrix

    Returns
    -------
    fractures_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    """

    # pre solver
    start0 = time.time()
    pre_matrix_solve(
        fractures_struc_array,
        element_struc_array,
        discharge_elements,
        discharge_int,
        head_matrix,
        z_int,
        connectivity_ids,
        connectivity_indptr,
        discharge_indices,
        coef_data=coef_data,
        coef_start=coef_start,
    )
    logger.debug(f"Pre solve time: {time.time() - start0}")

    # Solve the discharge matrix
    start0 = time.time()
    discharges[:] = lu_matrix.solve(head_matrix)
    logger.debug(f"Solve matrix time: {time.time() - start0}")

    # post solver
    start0 = time.time()
    post_matrix_solve(
        fractures_struc_array,
        element_struc_array,
        discharge_elements,
        discharges,
        discharges_old,
        discharge_indices,
    )
    logger.debug(f"Post solve time: {time.time() - start0}")


@nb.njit(parallel=PARALLEL, cache=CACHE)
def pre_matrix_solve(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    z_int,
    connectivity_ids=None,
    connectivity_indptr=None,
    discharge_indices=None,
    coef_data=None,
    coef_start=None,
):
    """
    Solves the discharge matrix for the DFN and stores the discharges and constants in the elements and fractures.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_elements : np.ndarray[element_dtype]
        The discharge elements
    discharge_int : int
        The number of integration points
    head_matrix : np.ndarray[dtype_head_matrix]
        The head matrix
    z_int : np.ndarray
        Selected-row quadrature, dense (2, selected, count).

    Returns
    -------
    fractures_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    """

    # Set the discharges equal to zero
    for i in nb.prange(len(element_struc_array)):
        element_struc_array[i]["q"] = 0.0

    # Set the constants equal to zero
    for i in nb.prange(fractures_struc_array.size):
        fractures_struc_array[i]["constant"] = 0.0

    # Get the head matrix
    build_head_matrix(
        fractures_struc_array,
        element_struc_array,
        discharge_elements,
        discharge_int,
        head_matrix,
        z_int,
        connectivity_ids,
        connectivity_indptr,
        discharge_indices,
        coef_data=coef_data,
        coef_start=coef_start,
    )


@nb.njit(parallel=PARALLEL, cache=CACHE)
def post_matrix_solve(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharges,
    discharges_old,
    discharge_indices=None,
):
    """
    Solves the discharge matrix for the DFN and stores the discharges and constants in the elements and fractures.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_elements : np.ndarray[element_dtype]
        The discharge elements
    discharges : np.ndarray
        The discharges
    discharges_old : np.ndarray
        The old discharges

    Returns
    -------
    fractures_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    """
    # TODO: Should I use damping here too?
    # Set the discharges for each element
    n_de = (
        len(discharge_elements) if discharge_indices is None else len(discharge_indices)
    )
    for i in nb.prange(n_de):
        e = (
            discharge_elements[i]
            if discharge_indices is None
            else element_struc_array[discharge_indices[i]]
        )
        element_struc_array[e["_id"]]["q"] = discharges[i]
        # element_struc_array[e["_id"]]["q"] = (discharges[i] + discharges_old[i]) / 2.0

    # Set the constants for each fracture
    for i in nb.prange(fractures_struc_array.size):
        fractures_struc_array[i]["constant"] = discharges[n_de + i]
        # fractures_struc_array[i]["constant"] = (discharges[len(discharge_elements) + i] + discharges_old[len(discharge_elements) + i]) / 2.0


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_old_discharges(
    element_struc_array,
    fractures_struc_array,
    discharge_elements,
    discharge_indices=None,
):
    n_de = (
        len(discharge_elements) if discharge_indices is None else len(discharge_indices)
    )
    discharges_old = np.zeros(n_de + np.max(element_struc_array["frac0"]) + 1)
    for i in nb.prange(n_de):
        e = (
            discharge_elements[i]
            if discharge_indices is None
            else element_struc_array[discharge_indices[i]]
        )
        discharges_old[i] = element_struc_array[e["_id"]]["q"]
    for i in nb.prange(fractures_struc_array.size):
        pos = n_de + i
        discharges_old[pos] = fractures_struc_array[i]["constant"]
    return discharges_old


@nb.njit(parallel=PARALLEL, cache=CACHE)
def build_head_matrix(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    z_int,
    connectivity_ids=None,
    connectivity_indptr=None,
    discharge_indices=None,
    coef_data=None,
    coef_start=None,
):
    """
    Builds the head matrix for the DFN and stores it.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_elements : np.ndarray[element_dtype]
        The discharge elements
    discharge_int : int
        The number of integration points
    head_matrix : np.ndarray[dtype_head_matrix]
        The head matrix
    z_int : np.ndarray
        Selected-row quadrature, dense (2, selected, count).

    Returns
    -------
    matrix : np.ndarray
        The head matrix
    """

    # Add the head for each discharge element
    n_de = (
        discharge_elements.size if discharge_indices is None else discharge_indices.size
    )
    for j in nb.prange(n_de):
        e = (
            discharge_elements[j]
            if discharge_indices is None
            else element_struc_array[discharge_indices[j]]
        )
        frac0 = fractures_struc_array[e["frac0"]]
        ids0 = hpc_fracture.fracture_element_ids(
            frac0, connectivity_ids, connectivity_indptr
        )
        z0 = quadrature_row(z_int, 0, j, discharge_int)
        omega = 0.0 + 0.0j
        for i in range(discharge_int):
            omega += hpc_fracture.calc_omega(
                frac0,
                z0[i],
                element_struc_array,
                -1,
                ids0,
                coef_data=coef_data,
                coef_start=coef_start,
            )
        omega = omega / discharge_int
        if e["_type"] == 0:  # Intersection
            frac1 = fractures_struc_array[e["frac1"]]
            ids1 = hpc_fracture.fracture_element_ids(
                frac1, connectivity_ids, connectivity_indptr
            )
            z1 = quadrature_row(z_int, 1, j, discharge_int)
            omega1 = 0.0 + 0.0j
            for i in range(discharge_int):
                omega1 += hpc_fracture.calc_omega(
                    frac1,
                    z1[i],
                    element_struc_array,
                    -1,
                    ids1,
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            omega1 = omega1 / discharge_int
            head_matrix[j] = np.real(omega1) / frac1["t"] - np.real(omega) / frac0["t"]
        elif e["_type"] in [2, 3]:  # Well or Constant head line
            head_matrix[j] = e["phi"] - np.real(omega)


def build_discharge_matrix_csr(
    fractures_struc_array,
    element_struc_array,
    connectivity_ids,
    connectivity_indptr,
    discharge_indices,
    discharge_int,
    z_int,
):
    """CSR connectivity and ordered positions; no element-record gathers."""
    Connectivity(connectivity_ids, connectivity_indptr).validate(
        fractures_struc_array.size, element_struc_array.size
    )
    if not np.array_equal(
        fractures_struc_array["_id"], np.arange(fractures_struc_array.size)
    ) or not np.array_equal(
        element_struc_array["_id"], np.arange(element_struc_array.size)
    ):
        raise ValueError("IDs must equal array positions")
    if (
        not isinstance(discharge_indices, np.ndarray)
        or discharge_indices.ndim != 1
        or discharge_indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or not discharge_indices.flags.c_contiguous
        or not np.array_equal(
            discharge_indices, get_discharge_indices(element_struc_array)
        )
    ):
        raise ValueError(
            "Discharge indices must contain all discharge elements in order"
        )
    validate_quadrature(z_int, discharge_indices.size, discharge_int)
    for name in ("frac0", "frac1"):
        references = element_struc_array[name]
        if np.any(references < -1) or np.any(references >= fractures_struc_array.size):
            raise ValueError(f"Invalid {name} fracture reference")
    if np.any(element_struc_array["frac0"][discharge_indices] < 0):
        raise ValueError("Discharge elements require frac0")
    intersections = element_struc_array["_type"] == 0
    if np.any(element_struc_array["frac1"][intersections] < 0):
        raise ValueError("Intersections require frac1")
    id_to_pos = np.full(element_struc_array.size, -1, dtype=np.int64)
    id_to_pos[discharge_indices] = np.arange(discharge_indices.size)

    nnz_per_row = count_discharge_nnz(
        fractures_struc_array,
        element_struc_array,
        element_struc_array,
        connectivity_ids,
        connectivity_indptr,
        discharge_indices,
    )
    row_offsets, total_nnz = exclusive_prefix_sum(nnz_per_row)

    rows = np.empty(total_nnz, np.int64)
    cols = np.empty(total_nnz, np.int64)
    data = np.empty(total_nnz, np.float64)
    size = len(nnz_per_row)

    fill_discharge_matrix(
        fractures_struc_array,
        element_struc_array,
        element_struc_array,
        id_to_pos,
        z_int,
        discharge_int,
        row_offsets,
        rows,
        cols,
        data,
        connectivity_ids,
        connectivity_indptr,
        discharge_indices,
    )

    logger.info(f"Dicharge matrix arrays size: {size}")

    # create the csr sparse matrix
    matrix = sp.sparse.csc_matrix((data, (rows, cols)), shape=(size, size))

    return matrix


@nb.njit(parallel=True, cache=CACHE)
def count_discharge_nnz(
    fractures,
    elements,
    discharge_elements,
    connectivity_ids=None,
    connectivity_indptr=None,
    discharge_indices=None,
):
    n_de = (
        discharge_elements.size if discharge_indices is None else discharge_indices.size
    )
    n_fr = fractures.size
    nnz_per_row = np.zeros(n_de + n_fr, np.int64)

    # Discharge element equations
    for j in nb.prange(n_de):
        e = (
            discharge_elements[j]
            if discharge_indices is None
            else elements[discharge_indices[j]]
        )
        cnt = 0

        if e["_type"] == 0:
            for f_id in (e["frac0"], e["frac1"]):
                f = fractures[f_id]
                element_ids = hpc_fracture.fracture_element_ids(
                    f, connectivity_ids, connectivity_indptr
                )
                for element_id in element_ids:
                    ee = elements[element_id]
                    t = ee["_type"]
                    if ee["_id"] == e["_id"] + 99 * 99 * 99 or (
                        t != 0 and t != 2 and t != 3
                    ):
                        continue
                    cnt += 1
            cnt += 2  # fracture continuity terms
        else:
            f = fractures[e["frac0"]]
            element_ids = hpc_fracture.fracture_element_ids(
                f, connectivity_ids, connectivity_indptr
            )
            for element_id in element_ids:
                ee = elements[element_id]
                t = ee["_type"]
                if ee["_id"] == e["_id"] + 99 * 99 * 99 or (
                    t != 0 and t != 2 and t != 3
                ):
                    continue
                cnt += 1
            cnt += 1

        nnz_per_row[j] = cnt

    # Fracture continuity equations
    for j in nb.prange(n_fr):
        f = fractures[j]
        cnt = 0
        element_ids = hpc_fracture.fracture_element_ids(
            f, connectivity_ids, connectivity_indptr
        )
        for element_id in element_ids:
            e = elements[element_id]
            if e["_type"] in (0, 2, 3):
                cnt += 1
        nnz_per_row[n_de + j] = cnt

    return nnz_per_row


@nb.njit(cache=CACHE)
def exclusive_prefix_sum(arr):
    out = np.empty_like(arr)
    s = 0
    for i in range(arr.size):
        out[i] = s
        s += arr[i]
    return out, s  # offsets, total nnz


@nb.njit(parallel=True, cache=CACHE)
def fill_discharge_matrix(
    fractures,
    elements,
    discharge_elements,
    id_to_pos,
    z_int,
    discharge_int,
    row_offsets,
    rows,
    cols,
    data,
    connectivity_ids=None,
    connectivity_indptr=None,
    discharge_indices=None,
):
    n_de = (
        discharge_elements.size if discharge_indices is None else discharge_indices.size
    )
    n_fr = fractures.size

    # ---- discharge element equations ----
    for j in nb.prange(n_de):
        e = (
            discharge_elements[j]
            if discharge_indices is None
            else elements[discharge_indices[j]]
        )
        row = j
        ptr = row_offsets[j]

        if e["_type"] == 0:
            z0 = quadrature_row(z_int, 0, j, discharge_int)
            z1 = quadrature_row(z_int, 1, j, discharge_int)

            for f_id, sign in ((e["frac0"], 1.0), (e["frac1"], -1.0)):
                f = fractures[f_id]
                element_ids = hpc_fracture.fracture_element_ids(
                    f, connectivity_ids, connectivity_indptr
                )
                for element_id in element_ids:
                    ee = elements[element_id]
                    t = ee["_type"]
                    if ee["_id"] == e["_id"] + 99 * 99 * 99 or (
                        t != 0 and t != 2 and t != 3
                    ):
                        continue

                    rows[ptr] = row
                    cols[ptr] = id_to_pos[ee["_id"]]
                    data[ptr] = hpc_fracture.head_from_phi(
                        f,
                        sign
                        * get_discharge_term(
                            ee, z0 if sign > 0 else z1, f_id, f["radius"], e["_id"]
                        ),
                    )
                    ptr += 1

            rows[ptr] = row
            cols[ptr] = n_de + e["frac0"]
            data[ptr] = hpc_fracture.head_from_phi(fractures[e["frac0"]], 1.0)
            ptr += 1

            rows[ptr] = row
            cols[ptr] = n_de + e["frac1"]
            data[ptr] = hpc_fracture.head_from_phi(fractures[e["frac1"]], -1.0)

        else:
            f = fractures[e["frac0"]]
            z0 = quadrature_row(z_int, 0, j, discharge_int)

            element_ids = hpc_fracture.fracture_element_ids(
                f, connectivity_ids, connectivity_indptr
            )
            for element_id in element_ids:
                ee = elements[element_id]
                t = ee["_type"]
                if ee["_id"] == e["_id"] + 99 * 99 * 99 or (
                    t != 0 and t != 2 and t != 3
                ):
                    continue

                rows[ptr] = row
                cols[ptr] = id_to_pos[ee["_id"]]
                data[ptr] = get_discharge_term(
                    ee, z0, e["frac0"], f["radius"], e["_id"]
                )
                ptr += 1

            rows[ptr] = row
            cols[ptr] = n_de + e["frac0"]
            data[ptr] = 1.0

    # ---- fracture continuity equations ----
    for j in nb.prange(n_fr):
        f = fractures[j]
        row = n_de + j
        ptr = row_offsets[row]

        element_ids = hpc_fracture.fracture_element_ids(
            f, connectivity_ids, connectivity_indptr
        )
        for element_id in element_ids:
            e = elements[element_id]
            t = e["_type"]
            if t not in (0, 2, 3):
                continue

            rows[ptr] = row
            cols[ptr] = id_to_pos[e["_id"]]
            data[ptr] = 1.0 if t != 0 or e["frac0"] == f["_id"] else -1.0
            ptr += 1


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_bnd_error(
    num_elements,
    fracture_struc_array,
    element_struc_array,
    work_array,
    discharge_int,
    bnd_error,
    z_int,
    nit,
    max_error,
    constants,
    coef_data=None,
    coef_start=None,
    prepared=False,
):
    """
    Builds the head matrix for the DFN and stores it.

    Parameters
    ----------
    num_elements : int
        The number of elements
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    work_array : WorkArrays
        Compact worker-owned solver state.
    discharge_int : int
        The number of integration points
    bnd_error : np.ndarray[dtype_head_matrix]
        The error matrix for the boundary conditions
    z_int : np.ndarray
        All-element quadrature, dense (2, elements, count).
    nit : int
        The number of iterations
    max_error : float
        The maximum error
    constants : np.ndarray[constants_dtype]
        The constants for the solver and dfn.

    Returns
    -------
    matrix : np.ndarray
        The head matrix
    """

    cnt_bnd = 0

    # Add the head for each discharge element
    for j in range(num_elements):
        e = element_struc_array[j]
        record = work_record(work_array, j)
        if e["_type"] == 2:  # Well
            bnd_error[j, 0] = 0.0
            bnd_error[j, 1] = e["_type"]
            bnd_error[j, 2] = e["q"]
            continue
        if e["_type"] in [0, 3]:  # Intersection, Constant head line
            frac0 = fracture_struc_array[e["frac0"]]
            z0 = quadrature_row(z_int, 0, j, discharge_int)
            omega_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega_vec[i] = hpc_fracture.calc_omega(
                    frac0,
                    z0[i],
                    element_struc_array,
                    coef_data=coef_data,
                    coef_start=coef_start,
                )
            head0 = np.real(omega_vec) / frac0["t"]
            # omega = np.sum(omega_vec) / discharge_int
            if e["_type"] == 0:  # Intersection
                frac1 = fracture_struc_array[e["frac1"]]
                z1 = quadrature_row(z_int, 1, j, discharge_int)
                omega1_vec = np.zeros(discharge_int, dtype=np.complex128)
                for i in range(discharge_int):
                    omega1_vec[i] = hpc_fracture.calc_omega(
                        frac1,
                        z1[i],
                        element_struc_array,
                        coef_data=coef_data,
                        coef_start=coef_start,
                    )
                # omega1 = np.sum(omega1_vec) / discharge_int
                head1 = np.real(omega1_vec) / frac1["t"]
                bnd_error[j, 0] = np.max(
                    np.abs(head1 - head0) / np.mean(head0)
                )  # np.abs((head1 - head0) / head1)
                bnd_error[j, 1] = e["_type"]
                bnd_error[j, 2] = e["q"]
            else:  # Well or Constant head line
                head = e["phi"] / frac0["t"]
                bnd_error[j, 0] = np.max(np.abs(head - head0) / head)
                bnd_error[j, 1] = e["_type"]
                bnd_error[j, 2] = e["q"]
        elif e["_type"] == 1:  # Bounding circle
            # The bounding circle is impermeable: ψ = Im(ω) must be constant
            # around the circle.  ψ is multi-valued: every time the integration
            # path crosses a branch cut of another element the raw Im(ω) jumps
            # by ±q.  find_branch_cuts locates these crossings and gives the
            # correction dpsi_corr[ii] for each interval [ii, ii+1].
            # After subtracting those jumps, the corrected ψ should be flat;
            # the peak-to-peak range of the corrected ψ is the true error.
            frac0 = fracture_struc_array[e["frac0"]]
            nint = int(e["nint"]) * 0 + discharge_int

            # nint evenly-spaced points on the bounding circle in z-plane
            # theta = np.linspace(0.0, 2.0 * np.pi, nint, endpoint=False)
            # z_pos = gf.map_chi_to_z_circle(
            #    np.exp(1j * theta), e["radius"], e["center"]
            # )
            # theta = np.arange(nint) * (2.0 * np.pi / nint)
            z_pos = quadrature_row(z_int, 0, j, nint)

            # Locate branch cuts and fill work_array[j] fields
            if not prepared:
                mf.find_branch_cuts(
                    e, z_pos, fracture_struc_array, element_struc_array, record, nint
                )

            # Build dpsi_corr vector (length nint-1) from work_array results
            dpsi_corr = record["dpsi"][:nint]
            dpsi_corr[:] = 0.0
            for k in range(record["len_discharge_element"]):
                ek = element_struc_array[record["discharge_element"][k]]
                pos = int(record["element_pos"][k])
                dpsi_corr[pos] += ek["q"] * record["sign_array"][k]

            # Evaluate ω at each of the nint points
            omega_pts = np.zeros(nint, dtype=np.complex128)
            for i in range(nint):
                omega_pts[i] = hpc_fracture.calc_omega(
                    frac0,
                    z_pos[i],
                    element_struc_array,
                    coef_data=coef_data,
                    coef_start=coef_start,
                )

            # Reconstruct corrected ψ by integrating branch-cut-corrected increments
            psi = record["psi"][:nint]
            psi[:] = 0.0
            psi[0] = np.imag(omega_pts[0])
            for ii in range(nint - 1):
                raw_dpsi = np.imag(omega_pts[ii + 1]) - np.imag(omega_pts[ii])
                psi[ii + 1] = psi[ii] + (raw_dpsi - dpsi_corr[ii])

            # Normalisation strategy
            # ----------------------
            # The error is Δψ / norm, aiming for a dimensionless relative
            # measure of how well ψ = const is satisfied.
            #
            # Primary scale: phi_range = max(Re ω) - min(Re ω) on the circle.
            #   * phi_range = T · h_range, so Δψ/phi_range = Δψ/(T·h_range)
            #     is a true dimensionless ratio independent of T.
            #   * It is already computed from omega_pts and naturally reflects
            #     the actual conditions on the fracture.
            #
            # Floor: T · radius · max_error
            #   * When T is very small, phi_range = T·h_range is inherently
            #     tiny even under a large head gradient, so phi_range alone
            #     can give a misleading (huge) error for a well-converged
            #     low-T fracture.
            #   * T · radius is the natural ψ-scale of the fracture (same
            #     units as φ).  Multiplying by max_error means: "if the
            #     potential variation is already below what T could produce
            #     at the target accuracy, the fracture is effectively at rest
            #     and the BC is trivially satisfied."
            #   * This floor is proportional to T, so it shrinks correctly
            #     for low-T fractures without being as aggressive as T·radius
            #     alone.
            phi_range = np.max(np.real(omega_pts)) - np.min(np.real(omega_pts))
            t_floor = phi_range / frac0["t"]

            nel = frac0["nelements"]
            el_ids = frac0["elements"][:nel]
            sum_q = 0.0
            for k in range(nel):
                ek = element_struc_array[el_ids[k]]
                if ek["_type"] in (0, 2, 3):
                    sum_q += np.abs(ek["q"])
            norm = max(sum_q, 1e-21)
            if norm < frac0["t"] / e["radius"]:
                norm = max(norm, t_floor)

            bnd_error[j, 0] = np.max(np.abs(psi)) / norm if norm > 1e-20 else 0.0
            bnd_error[j, 1] = e["_type"]
            bnd_error[j, 2] = sum_q
            bnd_error[j, 3] = phi_range
            bnd_error[j, 4] = t_floor
            bnd_error[j, 5] = norm

    return cnt_bnd


def get_max_min_phi(
    element_struc_array, fracture_struc_array, ids, frac_id, z_int, discharge_int
):
    """
    Get the maximum and minimum phi values from the elements.

    Parameters
    ----------
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    ids : np.ndarray
        The ids of the elements to check
    frac_id : int
        The id of the fracture to check
    z_int : np.ndarray
        Runtime quadrature points for discharge elements.
    discharge_int : int
        The number of integration points

    Returns
    -------
    max_phi : float
        The maximum phi value
    min_phi : float
        The minimum phi value
    """
    max_phi = 1e-300
    min_phi = 1e300
    z_max = 0.0 + 0.0j
    z_min = 0.0 + 0.0j
    l_max = 0.0
    l_min = 0.0
    for element_id in ids:
        e = element_struc_array[element_id]
        if e["_type"] in [0, 2, 3]:  # Well, Constant head line
            z0 = z_int["z0"][element_id][:discharge_int]
            frac0 = fracture_struc_array[e["frac0"]]
            endpoints = e["endpoints0"]
            if e["_type"] == 0 and e["frac1"] == frac_id:  # Intersection
                z0 = z_int["z1"][element_id][:discharge_int]
                frac0 = fracture_struc_array[e["frac1"]]
                endpoints = e["endpoints1"]
            omega_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega_vec[i] = hpc_fracture.calc_omega(
                    frac0, z0[i], element_struc_array
                )
            phi = np.sum(np.real(omega_vec)) / discharge_int
            phi_min = np.min(np.real(omega_vec))
            phi_max = np.max(np.real(omega_vec))
            if phi > max_phi:
                max_phi = phi
                z_max = z0[np.argmax(np.real(omega_vec))]
            if phi < min_phi:
                min_phi = phi
                z_min = z0[np.argmin(np.real(omega_vec))]
            if e["_type"] == 0:  # Intersection
                if phi_min < min_phi:
                    min_phi = phi_min
                    z_min = z0[np.argmin(np.real(omega_vec))]
                    l_min = np.abs(endpoints[1] - endpoints[0])
                if phi_max > max_phi:
                    max_phi = phi_max
                    z_max = z0[np.argmax(np.real(omega_vec))]
                    l_max = np.abs(endpoints[1] - endpoints[0])

    return max_phi, min_phi, z_max, z_min, l_min, l_max


@nb.njit(cache=CACHE)
def get_z_int_array(z_int, elements, discharge_int, element_indices=None):
    # Add the head for each discharge element
    size = elements.size if element_indices is None else element_indices.size
    for j in range(size):
        e = elements[j] if element_indices is None else elements[element_indices[j]]
        if e["_type"] == 0:  # Intersection
            quadrature_row(z_int, 0, j, discharge_int)[:] = hpc_intersection.z_array(
                e, discharge_int, e["frac0"]
            )
            quadrature_row(z_int, 1, j, discharge_int)[:] = hpc_intersection.z_array(
                e, discharge_int, e["frac1"]
            )
        elif e["_type"] == 1:  # Bounding circle
            quadrature_row(z_int, 0, j, discharge_int)[:] = hpc_bounding_circle.z_array(
                e, discharge_int
            )
        elif e["_type"] == 2:  # Well
            quadrature_row(z_int, 0, j, discharge_int)[:] = hpc_well.z_array(
                e, discharge_int
            )
        elif e["_type"] == 3:  # Constant head line
            quadrature_row(z_int, 0, j, discharge_int)[:] = hpc_const_head_line.z_array(
                e, discharge_int
            )


@nb.njit(cache=CACHE)
def get_discharge_term(element, z, frac, radius, e_is):
    if element["_type"] == 0:  # Intersection
        return hpc_intersection.discharge_term(
            element, z, frac, radius, element["_id"] == e_is
        )
    elif element["_type"] == 2:  # Well
        return hpc_well.discharge_term(element, z)
    elif element["_type"] == 3:  # Constant head line
        return hpc_const_head_line.discharge_term(
            element, z, radius, element["_id"] == e_is
        )
    else:
        return 0.0


@nb.njit()
def set_new_ncoef(self_, n, nint_mult=2):
    """
    Increase the number of coefficients in the asymptotic expansion.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The element to increase the number of coefficients.
    n : int
        The new number of coefficients.
    nint_mult : int
        The multiplier for the number of integration points.
    """
    if self_["_type"] == 0 or self_["_type"] == 3:  # Intersection
        self_["ncoef"] = n
        self_["nint"] = n * nint_mult
        stop = 2 * np.pi + 2 * np.pi / self_["nint"]
        self_["thetas"] = np.linspace(
            start=np.pi / (2 * self_["nint"]),
            stop=stop - stop / self_["nint"],
            num=self_["nint"],
        )
    elif self_["_type"] == 1:  # Bounding Circle
        self_["ncoef"] = n
        self_["nint"] = n * nint_mult
        self_["thetas"][: self_["nint"]] = np.linspace(
            start=0, stop=2 * np.pi - 2 * np.pi / self_["nint"], num=self_["nint"]
        )


def prepare_boundary_work(fractures, elements, count, source=None):
    """Prepare operation-private topology on the actual diagnostic coordinates."""
    from .hpc_branch import prepare_direct, publish_direct

    z_int = allocate_quadrature(elements.size, count)
    work = boundary_work(elements, count, source)
    get_z_int_array(z_int, elements, count)
    ids = np.flatnonzero(elements["_type"] == 1)
    lengths = np.full(elements.size, count, dtype=np.int64)
    prepared, stats = prepare_direct(
        work,
        elements,
        fractures,
        ids,
        lengths,
        quadrature=z_int,
    )
    work = prepared.work
    publish_direct(
        work, prepared.ids, prepared.starts, prepared.counts, prepared.values
    )
    return work, z_int, stats


def compute_bnd_error(
    fracture_struc_array,
    element_struc_array,
    discharge_int,
    constants,
    coef_data=None,
    coef_start=None,
    *,
    work=None,
):
    """
    Wrapper that calls get_bnd_error after allocating the required scratch arrays.

    For bounding circle elements, get_bnd_error internally calls find_branch_cuts
    to compute the branch-cut corrections and reconstruct the corrected ψ.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures (from dfn.fractures_struc_array_hpc)
    element_struc_array : np.ndarray[element_dtype]
        Array of elements (from dfn.elements_struc_array, the solved result)
    discharge_int : int
        Number of integration points
    constants : np.ndarray[constants_dtype]
        Solver constants

    Returns
    -------
    bnd_error : np.ndarray[float64]
        Per-element boundary condition error
    """
    num_elements = len(element_struc_array)

    if coef_data is None or coef_start is None:
        raise ValueError("Boundary diagnostics require compact coefficient storage")
    work_array, z_int, _ = prepare_boundary_work(
        fracture_struc_array,
        element_struc_array,
        discharge_int,
        work,
    )

    max_error = float(constants["MAX_ERROR"])
    bnd_error = np.zeros([num_elements, 7], dtype=np.float64)

    get_bnd_error(
        num_elements,
        fracture_struc_array,
        element_struc_array,
        work_array,
        discharge_int,
        bnd_error,
        z_int,
        0,  # nit – not used inside get_bnd_error
        max_error,
        constants,
        coef_data,
        coef_start,
        True,
    )

    return bnd_error
