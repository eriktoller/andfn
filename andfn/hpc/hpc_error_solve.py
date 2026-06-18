"""
Notes
-----
This module contains the HPC solve functions.
"""

import logging
import time

import numpy as np
import numba as nb
import scipy as sp
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_geometry_functions as gf
from andfn.hpc import (
    hpc_intersection,
    hpc_fracture,
    hpc_const_head_line,
    hpc_well,
    hpc_bounding_circle,
    hpc_imp_object,
    PARALLEL,
    CACHE,
)
from andfn.element import MAX_NCOEF, MAX_ELEMENTS

dtype_work = np.dtype(
    [
        ("phi", np.float64, MAX_NCOEF * 2),
        ("psi", np.float64, MAX_NCOEF * 2),
        ("coef", np.complex128, MAX_NCOEF),
        ("coef0", np.complex128, MAX_NCOEF),
        ("coef1", np.complex128, MAX_NCOEF),
        ("coef_error", np.complex128, MAX_NCOEF),
        ("a0_error", np.complex128),
        ("old_coef", np.complex128, MAX_NCOEF),
        ("dpsi", np.float64, MAX_NCOEF * 2),
        ("error", np.float64),
        ("integral", np.complex128, MAX_NCOEF * 2),
        ("sign_array", np.int64, MAX_ELEMENTS),
        ("discharge_element", np.int64, MAX_ELEMENTS),
        ("element_pos", np.int64, MAX_ELEMENTS),
        ("len_discharge_element", np.int64),
        ("exp_array_m", np.complex128, MAX_NCOEF * 2),
        ("exp_array_p", np.complex128, MAX_NCOEF * 2),
        ("z_integral", np.complex128, MAX_NCOEF * 2),
        ("set_zero", np.bool),
    ]
)

dtype_z_arrays = np.dtype(
    [("z0", complex, MAX_NCOEF * 2), ("z1", complex, MAX_NCOEF * 2)]
)

logger = logging.getLogger("andfn")


def solve_error(
    fracture_struc_array,
    element_struc_array,
    error_struc_array,
    discharge_int,
    constants,
    ntype_elements,
):
    """
    Solves the DFN.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    error_struc_array : np.ndarray[element_dtype]
        Array of elements to compute the error for
    discharge_int : int
        The number of integration points
    constants : np.ndarray[constants_dtype]
        The constants for the solver and dfn.
    ntype_elements : np.ndarray[int]
        A dictionary with the number of elements of each type.

    Returns
    -------
    error_struc_array : np.ndarray[element_dtype]
        The array of elements

    """
    # Get the constants, this is necessary for Numba parallelization to work
    max_error = constants["MAX_ERROR"]
    max_iterations = constants["MAX_ITERATIONS"] * 0 + 20
    damping = float(constants["DAMPING"])

    # get the discharge elements
    logger.info("Compiling HPC code...")

    # Allocate memory for the work array
    discharge_elements = get_discharge_elements(error_struc_array)
    num_elements = len(error_struc_array)
    work_array = np.zeros(num_elements, dtype=dtype_work)
    # head matrix
    size = discharge_elements.size + fracture_struc_array.size
    head_matrix = np.zeros(size)
    discharges = get_old_discharges(
        error_struc_array, fracture_struc_array, discharge_elements
    )
    discharges_old = np.zeros(size)
    z_int = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int, discharge_elements, discharge_int)
    z_int_error = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int_error, error_struc_array, discharge_int)

    # Discharge matrix
    logger.info("Building discharge matrix...")
    startdm = time.time()
    discharge_matrix = build_discharge_matrix(
        fracture_struc_array,
        error_struc_array,
        discharge_elements,
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
        e = error_struc_array[i]
        e["error_old"] = 1e30
        e["error"] = 1e30

    # fill work array ex_array
    for i, e in enumerate(error_struc_array):
        n = e["nint"]
        mf.calc_thetas(n, e["_type"], e["thetas"][:n])
        thetas = e["thetas"]
        mf.fill_exp_array(n, thetas, work_array[i]["exp_array_m"], -1)
        mf.fill_exp_array(n, thetas, work_array[i]["exp_array_p"], 1)
        mf.fill_z_integral(e, work_array[i])

    logger.info(f"Number of elements: {len(error_struc_array)}")
    logger.info(f"Number of fractures: {len(fracture_struc_array)}")

    error = np.float64(1.0)
    nit = 0
    cnt_error = 0
    error_q = 1.0
    start = time.time()
    sum_timee = 0.0
    sum_timeq = 0.0
    while cnt_error < 2 and nit < max_iterations:
        nit += 1
        # Solve the discharge matrix
        # startq = time.time()
        if error_q > max_error / 1e30:
            discharges_old[:] = discharges[:]
            solve_discharge_matrix_error(
                fracture_struc_array,
                element_struc_array,
                error_struc_array,
                discharge_elements,
                discharge_int,
                head_matrix,
                discharges,
                discharges_old,
                z_int,
                lu_matrix,
            )
            error_q = np.max(
                np.abs(discharges - discharges_old)
                / (np.max(np.abs(discharges_old)) + 1e-16)
            )
        # timeq = time.time() - startq

        # Solve the elements
        starte = time.time()
        element_solver_error(
            num_elements,
            element_struc_array,
            error_struc_array,
            fracture_struc_array,
            work_array,
            max_error,
            nit,
            cnt_error,
            damping,
        )
        timee = time.time() - starte
        sum_timee += timee

        error, _id = get_error(error_struc_array)
        error_coef = np.max(error_struc_array["error_coef"])

        # print info
        if nit < 10:
            logger.info(
                f"Iteration: 0{nit}, Error E: {error:.3e}, Coef: {error_coef:.3e}, Q: {error_q:.3e}, Element: {_id}, N of Coef: {error_struc_array[_id]['ncoef']}, Type: {error_struc_array[_id]['_type']}"
            )
        else:
            logger.info(
                f"Iteration: {nit}, Error E: {error:.3e}, Coef: {error_coef:.3e}, Q: {error_q:.3e}, Element: {_id}, N of Coef: {error_struc_array[_id]['ncoef']}, Type: {error_struc_array[_id]['_type']}"
            )

        if error_coef < max_error and error_q < max_error:
            cnt_error += 1
            error_q = 1e30
            # error = 1.0

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

    num_elements = len(element_struc_array)

    # Scratch arrays – work_array is zero-initialised;
    # find_branch_cuts resets len_discharge_element itself.
    work_array = np.zeros(num_elements, dtype=dtype_work)
    z_int = np.zeros(num_elements, dtype=dtype_z_arrays)

    # z_int is only used for intersection / well / const-head rows
    get_z_int_array(z_int, error_struc_array, discharge_int * 20)

    max_error = float(constants["MAX_ERROR"])
    bnd_error = np.zeros((num_elements, 7), dtype=np.float64)

    get_bnd_error(
        num_elements,
        fracture_struc_array,
        element_struc_array,
        error_struc_array,
        work_array,
        discharge_int * 20,
        bnd_error,
        z_int,
    )

    return error_struc_array, work_array


def solve_discharge_matrix_error(
    fractures_struc_array,
    element_struc_array,
    error_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    discharges,
    discharges_old,
    z_int,
    lu_matrix,
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
    z_int : np.ndarray[dtype_z_arrays]
        The z arrays for the discharge elements
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
        error_struc_array,
        discharge_elements,
        discharge_int,
        head_matrix,
        z_int,
    )
    logger.debug(f"Pre solve time: {time.time() - start0}")

    # Solve the discharge matrix
    start0 = time.time()
    discharges[:] = lu_matrix.solve(head_matrix)
    print(discharges)
    logger.debug(f"Solve matrix time: {time.time() - start0}")

    # post solver
    start0 = time.time()
    post_matrix_solve(
        fractures_struc_array,
        error_struc_array,
        discharge_elements,
        discharges,
        discharges_old,
    )
    logger.debug(f"Post solve time: {time.time() - start0}")


@nb.njit(parallel=PARALLEL, cache=CACHE)
def pre_matrix_solve(
    fractures_struc_array,
    element_struc_array,
    error_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    z_int,
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
    z_int : np.ndarray[dtype_z_arrays]
        The z arrays for the discharge elements

    Returns
    -------
    fractures_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    """

    # Set the discharges equal to zero
    for i in nb.prange(len(error_struc_array)):
        error_struc_array[i]["q"] = 0.0

    # Set the constants equal to zero
    for i in nb.prange(len(fractures_struc_array)):
        fractures_struc_array[i]["error_constant"] = 0.0

    # Get the head matrix
    build_head_matrix(
        fractures_struc_array,
        element_struc_array,
        error_struc_array,
        discharge_elements,
        discharge_int,
        head_matrix,
        z_int,
    )


@nb.njit(parallel=PARALLEL, cache=CACHE)
def post_matrix_solve(
    fractures_struc_array,
    error_struc_array,
    discharge_elements,
    discharges,
    discharges_old,
):
    """
    Solves the discharge matrix for the DFN and stores the discharges and constants in the elements and fractures.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    error_struc_array : np.ndarray[element_dtype]
        Array of elements error
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
    for i in nb.prange(len(discharge_elements)):
        e = discharge_elements[i]
        error_struc_array[e["_id"]]["q"] = discharges[i]

    # Set the constants for each fracture
    for i in nb.prange(len(fractures_struc_array)):
        fractures_struc_array[i]["error_constant"] = discharges[
            len(discharge_elements) + i
        ]


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_old_discharges(error_struc_array, fractures_struc_array, discharge_elements):
    discharges_old = np.zeros(
        len(discharge_elements) + np.max(error_struc_array["frac0"]) + 1
    )
    for i in nb.prange(len(discharge_elements)):
        e = discharge_elements[i]
        discharges_old[i] = error_struc_array[e["_id"]]["q"]
    for i in nb.prange(len(fractures_struc_array)):
        pos = len(discharge_elements) + i
        discharges_old[pos] = fractures_struc_array[i]["error_constant"]
    return discharges_old


@nb.njit(parallel=PARALLEL, cache=CACHE)
def build_head_matrix(
    fractures_struc_array,
    element_struc_array,
    error_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    z_int,
):
    """
    Builds the head matrix for the DFN and stores it.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    error_struc_array : np.ndarray[element_dtype]
        Array of elements to compute the error for
    discharge_elements : np.ndarray[element_dtype]
        The discharge elements
    discharge_int : int
        The number of integration points
    head_matrix : np.ndarray[dtype_head_matrix]
        The head matrix
    z_int : np.ndarray[dtype_z_arrays]
        The z arrays for the discharge elements

    Returns
    -------
    matrix : np.ndarray
        The head matrix
    """

    # Add the head for each discharge element
    for j in nb.prange(discharge_elements.size):
        e = discharge_elements[j]
        frac0 = fractures_struc_array[e["frac0"]]
        z0 = z_int["z0"][j][:discharge_int]
        omega = 0.0 + 0.0j
        er = 0.0 + 0.0j
        diff = 0.0
        fi0 = 0.0
        for i in range(discharge_int):
            fi_tmp = np.real(hpc_fracture.calc_omega(frac0, z0[i], element_struc_array))
            er_tmp = np.real(
                hpc_fracture.calc_omega_error(frac0, z0[i], error_struc_array)
            )
            omega += fi_tmp
            er += er_tmp
            diff += e["phi"] - fi_tmp
            fi0 += fi_tmp
        omega = omega / discharge_int
        er = er / discharge_int
        diff = (e["phi"] - omega) + er
        if e["_type"] == 0:  # Intersection
            frac1 = fractures_struc_array[e["frac1"]]
            z1 = z_int["z1"][j][:discharge_int]
            omega1 = 0.0 + 0.0j
            er1 = 0.0 + 0.0j
            diff1 = 0.0
            diff_er = 0.0
            er00 = 0.0
            er11 = 0.0
            fi00 = 0.0
            fi11 = 0.0
            for i in range(discharge_int):
                omega1 += hpc_fracture.calc_omega_error(
                    frac1, z1[i], element_struc_array
                )
                er0 = (
                    np.real(
                        hpc_fracture.calc_omega_error(frac0, z0[i], error_struc_array)
                    )
                    / frac0["t"]
                )
                er1 = (
                    np.real(
                        hpc_fracture.calc_omega_error(frac1, z1[i], error_struc_array)
                    )
                    / frac1["t"]
                )
                fi_tmp = (
                    np.real(hpc_fracture.calc_omega(frac0, z0[i], element_struc_array))
                    / frac0["t"]
                )
                fi_tmp1 = (
                    np.real(hpc_fracture.calc_omega(frac1, z1[i], element_struc_array))
                    / frac1["t"]
                )
                diff1 += fi_tmp1 - fi_tmp
                diff_er += er1 - er0
                er00 += er0
                er11 += er1
                fi00 += fi_tmp
                fi11 += fi_tmp1
            er1 = er1 / discharge_int
            er11 = er11 / discharge_int
            er00 = er00 / discharge_int
            fi00 = fi00 / discharge_int
            fi11 = fi11 / discharge_int
            omega1 = omega1 / discharge_int
            diff1 = fi11 - fi00 + er11 - (fi11 - fi00 + er00)
            head_matrix[j] = diff1
        elif e["_type"] in [2, 3]:  # Well or Constant head line
            head_matrix[j] = (e["phi"] - np.real(omega)) + np.real(er)
            head_matrix[j] = -diff


def build_discharge_matrix(
    fractures_struc_array,
    error_struc_array,
    discharge_elements,
    discharge_int,
    z_int,
):
    """
    Builds the discharge matrix for the DFN and adds it to the DFN.

    """
    # Estimate the maximum number of non-zero entries in the discharge matrix
    max_id = int(np.max(error_struc_array["_id"]))
    id_to_pos = np.full(max_id + 1, -1, dtype=np.int32)
    for i, e in enumerate(discharge_elements):
        id_to_pos[e["_id"]] = i

    nnz_per_row = count_discharge_nnz(
        fractures_struc_array, error_struc_array, discharge_elements
    )
    row_offsets, total_nnz = exclusive_prefix_sum(nnz_per_row)

    rows = np.empty(total_nnz, np.int64)
    cols = np.empty(total_nnz, np.int64)
    data = np.empty(total_nnz, np.float64)
    size = len(nnz_per_row)

    fill_discharge_matrix(
        fractures_struc_array,
        error_struc_array,
        discharge_elements,
        id_to_pos,
        z_int,
        discharge_int,
        row_offsets,
        rows,
        cols,
        data,
    )

    logger.info(f"Dicharge matrix arrays size: {size}")

    # create the csr sparse matrix
    matrix = sp.sparse.csc_matrix((data, (rows, cols)), shape=(size, size))

    return matrix


@nb.njit(parallel=True, cache=CACHE)
def count_discharge_nnz(fractures, elements, discharge_elements):
    n_de = discharge_elements.size
    n_fr = fractures.size
    nnz_per_row = np.zeros(n_de + n_fr, np.int64)

    # Discharge element equations
    for j in nb.prange(n_de):
        e = discharge_elements[j]
        cnt = 0

        if e["_type"] == 0:
            for f_id in (e["frac0"], e["frac1"]):
                f = fractures[f_id]
                for k in range(f["nelements"]):
                    ee = elements[f["elements"][k]]
                    t = ee["_type"]
                    if ee["_id"] == e["_id"] or (t != 0 and t != 2 and t != 3):
                        continue
                    cnt += 1
            cnt += 2  # fracture continuity terms
        else:
            f = fractures[e["frac0"]]
            for k in range(f["nelements"]):
                ee = elements[f["elements"][k]]
                t = ee["_type"]
                if ee["_id"] == e["_id"] or (t != 0 and t != 2 and t != 3):
                    continue
                cnt += 1
            cnt += 1

        nnz_per_row[j] = cnt

    # Fracture continuity equations
    for j in nb.prange(n_fr):
        f = fractures[j]
        cnt = 0
        for k in range(f["nelements"]):
            e = elements[f["elements"][k]]
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
):
    n_de = discharge_elements.size
    n_fr = fractures.size

    # ---- discharge element equations ----
    for j in nb.prange(n_de):
        e = discharge_elements[j]
        row = j
        ptr = row_offsets[j]

        if e["_type"] == 0:
            z0 = z_int["z0"][j][:discharge_int]
            z1 = z_int["z1"][j][:discharge_int]

            for f_id, sign in ((e["frac0"], 1.0), (e["frac1"], -1.0)):
                f = fractures[f_id]
                for k in range(f["nelements"]):
                    ee = elements[f["elements"][k]]
                    t = ee["_type"]
                    if ee["_id"] == e["_id"] or (t != 0 and t != 2 and t != 3):
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
            z0 = z_int["z0"][j][:discharge_int]

            for k in range(f["nelements"]):
                ee = elements[f["elements"][k]]
                t = ee["_type"]
                if ee["_id"] == e["_id"] or (t != 0 and t != 2 and t != 3):
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

        for k in range(f["nelements"]):
            e = elements[f["elements"][k]]
            t = e["_type"]
            if t not in (0, 2, 3):
                continue

            rows[ptr] = row
            cols[ptr] = id_to_pos[e["_id"]]
            data[ptr] = 1.0 if t != 0 or e["frac0"] == f["_id"] else -1.0
            ptr += 1


@nb.njit(cache=CACHE)
def get_error(error_struc_array):
    """
    Get the maximum error from the elements and the element that it is associated with.

    Parameters
    ----------
    error_struc_array : np.ndarray[element_dtype]
        The array of elements

    Returns
    -------
    error : float
        The maximum error
    _id : int
        The id of the element with the maximum error
    """
    error = np.max(error_struc_array["error"])
    _id = np.argmax(error_struc_array["error"])
    return error, _id


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_discharge_elements(error_struc_array):
    """
    Get the discharge elements from the element array.

    Parameters
    ----------
    error_struc_array : np.ndarray[element_dtype]
        The array of elements

    Returns
    -------
    discharge_elements : np.ndarray[element_dtype]
        The array of discharge elements
    """
    # get the discharge elements
    el = np.zeros(len(error_struc_array), dtype=np.bool_)
    for i in nb.prange(len(error_struc_array)):
        if error_struc_array[i]["_type"] in {
            0,
            2,
            3,
        }:  # Intersection, Well, Constant head line
            el[i] = 1
    discharge_elements = error_struc_array[el]
    return discharge_elements


@nb.njit(parallel=PARALLEL, cache=CACHE)
def element_solver_error(
    num_elements,
    element_struc_array,
    error_struc_array,
    fracture_struc_array,
    work_array,
    max_error,
    nit,
    cnt_error,
    damping,
):
    """
    Solves the elements and updates the coefficients in the work array.

    Parameters
    ----------
    num_elements : int
        The number of elements
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    error_struc_array : np.ndarray[element_dtype]
        Array of elements to compute the error for
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    work_array : np.ndarray[dtype_work]
        The work array
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
        e = error_struc_array[i]
        if e["error"] < max_error * 0 and nit > 30 and cnt_error == 0:
            cnt += 1
            continue
        if e["_type"] == 0:  # Intersection
            hpc_intersection.solve_error(
                e,
                fracture_struc_array,
                element_struc_array,
                error_struc_array,
                work_array[i],
            )
        elif e["_type"] == 1:  # Bounding circle
            hpc_bounding_circle.solve_error(
                e,
                fracture_struc_array,
                element_struc_array,
                error_struc_array,
                work_array[i],
            )
        elif e["_type"] == 2:  # Well
            e["error"] = 0.0
            cnt += 1
        elif e["_type"] == 3:  # Constant head line
            hpc_const_head_line.solve_error(
                e,
                fracture_struc_array,
                element_struc_array,
                error_struc_array,
                work_array[i],
            )
        elif e["_type"] == 4:  # Impermeable circle
            hpc_imp_object.solve_circle(
                e,
                fracture_struc_array,
                element_struc_array,
                error_struc_array,
                work_array[i],
            )
        elif e["_type"] == 5:  # Impermeable line
            hpc_imp_object.solve_line(
                e,
                fracture_struc_array,
                element_struc_array,
                error_struc_array,
                work_array[i],
            )

    # Get the coefficients from the work array
    for i in nb.prange(num_elements):
        e = error_struc_array[i]
        e["coef"][: e["ncoef"]] = (
            damping * work_array[i]["coef"][: e["ncoef"]]
            + (1 - damping) * e["coef"][: e["ncoef"]]
        )

    return cnt


@nb.njit(cache=CACHE)
def get_z_int_array(z_int, elements, discharge_int):
    # Add the head for each discharge element
    for j in range(elements.size):
        e = elements[j]
        if e["_type"] == 0:  # Intersection
            z_int["z0"][j][:discharge_int] = hpc_intersection.z_array(
                e, discharge_int, e["frac0"]
            )
            z_int["z1"][j][:discharge_int] = hpc_intersection.z_array(
                e, discharge_int, e["frac1"]
            )
        elif e["_type"] == 1:  # Bounding circle
            z_int["z0"][j][:discharge_int] = hpc_bounding_circle.z_array(
                e, discharge_int
            )
        elif e["_type"] == 2:  # Well
            z_int["z0"][j][:discharge_int] = hpc_well.z_array(e, discharge_int)
        elif e["_type"] == 3:  # Constant head line
            z_int["z0"][j][:discharge_int] = hpc_const_head_line.z_array(
                e, discharge_int
            )


@nb.njit(cache=CACHE)
def get_discharge_term(element, z, frac, radius, e_is):
    if element["_type"] == 0:  # Intersection
        return hpc_intersection.discharge_term_error(
            element,
            z,
            frac,
            radius,
        )
    elif element["_type"] == 2:  # Well
        return hpc_well.discharge_term(element, z)
    elif element["_type"] == 3:  # Constant head line
        return hpc_const_head_line.discharge_term_error(
            element,
            z,
            radius,
        )
    else:
        return 0.0


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_bnd_error(
    num_elements,
    fracture_struc_array,
    element_struc_array,
    error_struc_array,
    work_array,
    discharge_int,
    bnd_error,
    z_int,
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
    work_array : np.ndarray[dtype_work]
        The work array
    discharge_int : int
        The number of integration points
    bnd_error : np.ndarray[dtype_head_matrix]
        The error matrix for the boundary conditions
    z_int : np.ndarray[dtype_z_arrays]
        The z arrays for the discharge elements
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
    for e in error_struc_array:
        e["q"] += 0.0
    # fracture_struc_array['error_constant'][1] = 0.0
    cnt_discharge = -1
    # Add the head for each discharge element
    for j in range(num_elements):
        e = element_struc_array[j]
        nint = int(e["nint"])
        nint = discharge_int
        frac0 = fracture_struc_array[e["frac0"]]
        if e["_type"] == 2:  # Well
            bnd_error[j, 0] = 0.0
            bnd_error[j, 1] = e["_type"]
            bnd_error[j, 2] = e["q"]
            continue
        if e["_type"] in [0, 3]:  # Intersection, Constant head line
            dphi = np.zeros(nint, dtype=np.float64)
            dphi_only = np.zeros(nint, dtype=np.float64)

            if e["_type"] == 0:  # Intersection
                frac1 = fracture_struc_array[e["frac1"]]
                omega_er = np.zeros(nint, dtype=np.complex128)
                omega_er1 = np.zeros(nint, dtype=np.complex128)
                z0 = z_int["z0"][j][:discharge_int]
                z1 = z_int["z1"][j][:discharge_int]
                for ii in range(nint):
                    chi = work_array["exp_array_p"][j][ii]

                    omega0 = (
                        hpc_fracture.calc_omega(frac0, z0[ii], element_struc_array)
                        / frac0["t"]
                    )
                    omega_error0 = (
                        hpc_fracture.calc_omega_error(frac0, z0[ii], error_struc_array)
                        / frac0["t"]
                    )
                    omega1 = (
                        hpc_fracture.calc_omega(frac1, z1[ii], element_struc_array)
                        / frac1["t"]
                    )
                    omega_error1 = (
                        hpc_fracture.calc_omega_error(frac1, z1[ii], error_struc_array)
                        / frac1["t"]
                    )
                    omega_er[ii] = omega_error0
                    omega_er1[ii] = omega_error1
                    print(
                        f"omega0={omega0.real}, omega1={omega1.real}, phi_error={omega1.real - omega0.real}, omega_error0={omega_error0.real}, omega_error1={omega_error1.real}"
                    )
                    dphi[ii] = (np.real(omega0) - np.real(omega1)) + (
                        np.real(omega_error0) - np.real(omega_error1)
                    )
                    dphi_only[ii] = np.real(omega1) - np.real(omega0)

                import matplotlib.pyplot as plt

                plt.figure()
                plt.title(
                    f"Boundary condition error for element {j} (type {e['_type']})"
                )
                plt.plot(dphi_only, label="BC")
                plt.plot(omega_er.real, label="Re E(z)")
                plt.plot(omega_er1.real, label="Re E(z) frac1", linestyle="dashed")
                plt.plot(dphi_only + omega_er.real, label="Diff")
                plt.legend()
            else:  # Well or Constant head line
                omega_er = np.zeros(nint, dtype=np.complex128)
                cnt_discharge += 1
                z0 = z_int["z0"][j][:discharge_int]
                for ii in range(nint):
                    chi = work_array["exp_array_p"][j][ii]
                    z = gf.map_chi_to_z_line(chi, e["endpoints0"])
                    z = z0[ii]
                    omega = hpc_fracture.calc_omega(frac0, z, element_struc_array)
                    omega_error = hpc_fracture.calc_omega_error(
                        frac0, z, error_struc_array
                    )
                    omega_er[ii] = omega_error
                    dphi[ii] = e["phi"] - np.real(omega) + np.real(omega_error)
                    dphi_only[ii] = e["phi"] - np.real(omega)

                import matplotlib.pyplot as plt

                print(f"mean={np.mean(dphi_only)}")
                print(f"mean error={np.mean(omega_er.real)}")
                print(f"diff mean ={np.mean(dphi_only) - np.mean(omega_er.real)}")

                plt.figure()
                plt.title(
                    f"Boundary condition error for element {j} (type {e['_type']})"
                )
                plt.plot(dphi_only, label="BC")
                plt.plot(omega_er.real, label="Re E(z)")
                plt.plot(dphi_only + omega_er.real, label="Diff")
                plt.plot(
                    np.mean(dphi_only) - np.mean(omega_er.real),
                    label="Diff mean",
                    marker="o",
                )
                plt.plot(np.mean(dphi_only), label="BC mean", marker="o")
                plt.plot(
                    np.mean(omega_er.real), label="E(z) mean", marker="s", zorder=0
                )
                plt.legend()
        elif e["_type"] == 1:  # Bounding circle
            # dpsi_corr = e["dpsi_corr"][: nint - 1]
            # dpsi = np.zeros(nint, dtype=np.float64)
            # dpsi_only = np.zeros(nint, dtype=np.float64)
            om_error = np.zeros(nint, dtype=np.complex128)
            z0 = z_int["z0"][j][:discharge_int]
            # Locate branch cuts and fill work_array[j] fields
            mf.find_branch_cuts(
                e, z0, fracture_struc_array, element_struc_array, work_array[j], nint
            )

            # Build dpsi_corr vector (length nint-1) from work_array results
            dpsi_corr = np.zeros(nint)
            for k in range(work_array[j]["len_discharge_element"]):
                ek = element_struc_array[work_array[j]["discharge_element"][k]]
                pos = int(work_array[j]["element_pos"][k])
                dpsi_corr[pos] += ek["q"] * work_array[j]["sign_array"][k]

            # Evaluate ω at each of the nint points
            omega_pts = np.zeros(nint, dtype=np.complex128)
            for i in range(nint):
                omega_pts[i] = hpc_fracture.calc_omega(
                    frac0, z0[i], element_struc_array
                )
                om_error[i] = hpc_fracture.calc_omega_error(
                    frac0, z0[i], error_struc_array
                )

            # Reconstruct corrected ψ by integrating branch-cut-corrected increments
            psi = np.zeros(nint)
            psi[0] = np.imag(omega_pts[0])
            for ii in range(nint - 1):
                raw_dpsi = np.imag(omega_pts[ii + 1]) - np.imag(omega_pts[ii])
                psi[ii + 1] = psi[ii] + (raw_dpsi - dpsi_corr[ii])

            import matplotlib.pyplot as plt

            plt.figure()
            plt.title(f"Boundary condition error for element {j} (type {e['_type']})")
            plt.plot(psi, label="BC")
            plt.plot(-om_error.imag, label="Im E(z)")
            plt.plot(-om_error.real, label="Re E(z)")
            plt.legend()

    plt.show()
