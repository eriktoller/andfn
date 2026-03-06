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


def solve(
    fracture_struc_array,
    element_struc_array,
    discharge_int,
    constants,
):
    """
    Solves the DFN.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_int : int
        The number of integration points
    constants : np.ndarray[constants_dtype]
        The constants for the solver and dfn.

    Returns
    -------
    element_struc_array : np.ndarray[element_dtype]
        The array of elements

    """
    # Get the constants, this is necessary for Numba parallelization to work
    max_error = constants["MAX_ERROR"]
    max_iterations = constants["MAX_ITERATIONS"]
    coef_ratio = constants["COEF_RATIO"]
    max_coef = constants["MAX_NCOEF"]
    coef_increase = constants["COEF_INCREASE"]
    damping = float(constants["DAMPING"])

    # get the discharge elements
    logger.info("Compiling HPC code...")
    discharge_elements = get_discharge_elements(element_struc_array)

    # Allocate memory for the work array
    num_elements = len(element_struc_array)
    work_array = np.zeros(num_elements, dtype=dtype_work)
    # head matrix
    size = discharge_elements.size + fracture_struc_array.size
    head_matrix = np.zeros(size)
    discharges = get_old_discharges(
        element_struc_array, fracture_struc_array, discharge_elements
    )
    discharges_old = np.zeros(size)
    z_int = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int, discharge_elements, discharge_int)
    z_int_error = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int_error, element_struc_array, discharge_int)

    # Discharge matrix
    logger.info("Building discharge matrix...")
    startdm = time.time()
    discharge_matrix = build_discharge_matrix(
        fracture_struc_array,
        element_struc_array,
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
        e = element_struc_array[i]
        e["error_old"] = 1e30
        e["error"] = 1e30

    # fill work array ex_array
    for i, e in enumerate(element_struc_array):
        n = e["nint"]
        mf.calc_thetas(n, e["_type"], e["thetas"][:n])
        thetas = e["thetas"]
        mf.fill_exp_array(n, thetas, work_array[i]["exp_array_m"], -1)
        mf.fill_exp_array(n, thetas, work_array[i]["exp_array_p"], 1)
        mf.fill_z_integral(e, work_array[i])

    logger.info(f"Number of elements: {len(element_struc_array)}")
    logger.info(f"Number of fractures: {len(fracture_struc_array)}")
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
                fracture_struc_array,
                element_struc_array,
                discharge_elements,
                discharge_int,
                head_matrix,
                discharges,
                discharges_old,
                z_int,
                lu_matrix,
            )
            error_q = np.max(np.abs(discharges - discharges_old))
            error_q = np.mean(np.abs((discharges - discharges_old)))
            error_q = np.max(
                np.abs(discharges - discharges_old)
                / (np.max(np.abs(discharges_old)) + 1e-16)
            )
        timeq = time.time() - startq

        # Solve the elements
        starte = time.time()
        cnt_set_zero = element_solver2(
            num_elements,
            element_struc_array,
            fracture_struc_array,
            work_array,
            max_error,
            nit,
            cnt_error,
            coef_ratio,
            max_coef,
            coef_increase,
            damping,
        )
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
            e = element_struc_array[_id]
            # print(
            #    f"Error increasing for element {e['_id']}, type {e['_type']}, fractures {e['frac0']} and {e['frac1']}, "
            #    f"error: {e['error']}, error_old: {e['error_old']}, error_old2: {e['error_old2']}, ncoef: {e['ncoef']}, "
            #    f"\n   coef[:5]:    {e['coef'][:5]}, \nold_coef[:5]: {work_array[_id]['old_coef'][:5]}"
            #    f"\ncoef[:-5]:    {e['coef'][e['ncoef']-5:e['ncoef']]}, \nold_coef[:-5]: {work_array[_id]['old_coef'][e['ncoef']-5:e['ncoef']]}"
            #    f"\npercent change: {np.abs(np.sum(np.abs(e['coef']))-np.sum(np.abs(work_array[_id]['old_coef'])))/np.sum(np.abs(work_array[_id]['old_coef']))*100 if np.sum(np.abs(work_array[_id]['old_coef'])) > 0 else np.nan}")

        if cnt_bnd > 1 or nit == max_iterations:
            element_solver(
                num_elements,
                element_struc_array,
                fracture_struc_array,
                work_array,
                max_error,
                nit,
                cnt_error,
                damping,
            )
            # error = 1.0

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

    return element_struc_array


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


@nb.njit(parallel=PARALLEL, cache=CACHE)
def element_solver(
    num_elements,
    element_struc_array,
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
        e = element_struc_array[i]
        if e["error"] < max_error * 0 and nit > 30 and cnt_error == 0:
            cnt += 1
            continue
        if e["_type"] == 0:  # Intersection
            hpc_intersection.solve(
                e, fracture_struc_array, element_struc_array, work_array[i]
            )
        elif e["_type"] == 1:  # Bounding circle
            hpc_bounding_circle.solve(
                e, fracture_struc_array, element_struc_array, work_array[i]
            )
        elif e["_type"] == 2:  # Well
            e["error"] = 0.0
            cnt += 1
        elif e["_type"] == 3:  # Constant head line
            hpc_const_head_line.solve(
                e, fracture_struc_array, element_struc_array, work_array[i]
            )
        elif e["_type"] == 4:  # Impermeable circle
            hpc_imp_object.solve_circle(
                e, fracture_struc_array, element_struc_array, work_array[i]
            )
        elif e["_type"] == 5:  # Impermeable line
            hpc_imp_object.solve_line(
                e, fracture_struc_array, element_struc_array, work_array[i]
            )

    # Get the coefficients from the work array
    for i in nb.prange(num_elements):
        e = element_struc_array[i]
        e["coef"][: e["ncoef"]] = (
            damping * work_array[i]["coef"][: e["ncoef"]]
            + (1 - damping) * e["coef"][: e["ncoef"]]
        )

    return cnt


@nb.njit(parallel=PARALLEL, cache=CACHE)
def element_solver2(
    num_elements,
    element_struc_array,
    fracture_struc_array,
    work_array,
    max_error,
    nit,
    cnt_error,
    max_coef_ratio,
    max_coef,
    coef_increase,
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
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    work_array : np.ndarray[dtype_work]
        The work array
    max_error : float
        The maximum error
    nit : int
        The number of iterations
    cnt_error : int
        The number of errors
    max_coef_ratio : float
        The coefficient ratio
    max_coef : int
        The maximum number of coefficients
    coef_increase : int
        The coefficient increase
    damping : float
        The damping factor for the solver (default 0.5)

    Returns
    -------
    cnt : int
        The number of elements that were solved
    """
    error = 1.0
    nit_el = 0
    while error > max_error and nit_el < 1:
        nit_el += 1

        # Solve the elements
        if nit <= 2000000000000:
            for _ in range(1):
                element_solver(
                    num_elements,
                    element_struc_array,
                    fracture_struc_array,
                    work_array,
                    max_error,
                    nit,
                    cnt_error,
                    damping,
                )

        error, _id = get_error(element_struc_array)

        ncc = 3  # Number of coefficients to check the ratio

        # Get the coefficients from the work array
        if nit < 2:
            continue
        for i in nb.prange(num_elements):
            # TODO: check if I can use sum(coef) and look if the change in sum due to the last 3 is significant,
            #  maybe also check if the error is continously increasing for an element (possibly track error history)
            work_array[i]["set_zero"] = False
            e = element_struc_array[i]
            if e["_type"] == 2:  # Well
                continue
            coefs = e["coef"][: e["ncoef"]]
            coef0 = np.mean(np.abs(coefs[1:ncc])) + 1e-30  # CHANGED
            coef1 = np.max(np.abs(coefs[-ncc:]))
            if coef0 == 0.0:
                coef_ratio = 0.0
            else:
                coef_ratio = coef1 / coef0 * 0
            if np.max(np.abs(coefs[1:2])) < max_error / 1e10:
                coef_ratio = 0.0
            if (
                e["error"] > e["error_old"]
                and e["error"] > e["error_old2"]
                and e["error"] > max_error
                and nit > 5
            ):
                work_array[i]["set_zero"] = True
                work_array[i]["set_zero"] = False
                coef_ratio = max_coef_ratio + 1.0
            if (
                np.abs(np.sum(coefs[-ncc:]) / (np.sum(coefs[:-ncc]) + 1e-30))
                > max_coef_ratio
                and e["error"] > max_error
            ):
                coef_ratio = max_coef_ratio + 1.0
            cnt = 0

            # Increase the number of coefficients if the ratio is too high
            max_cnt = 1
            while (
                coef_ratio > max_coef_ratio
                and e["ncoef"] < max_coef
                and cnt < max_cnt
                and nit > 1
            ):
                cnt += 1

                e["ncoef"] = int(e["ncoef"] + coef_increase)
                e["nint"] = e["ncoef"] * 2

                mf.calc_thetas(e["nint"], e["_type"], e["thetas"][: e["nint"]])
                work_array[i]["len_discharge_element"] = 0
                mf.fill_exp_array(
                    e["nint"], e["thetas"], work_array[i]["exp_array_m"], -1
                )
                mf.fill_exp_array(
                    e["nint"], e["thetas"], work_array[i]["exp_array_p"], 1
                )
                mf.fill_z_integral(e, work_array[i])

                if e["_type"] == 0:  # Intersection
                    hpc_intersection.solve(
                        e, fracture_struc_array, element_struc_array, work_array[i]
                    )
                elif e["_type"] == 1:  # Bounding circle
                    hpc_bounding_circle.solve(
                        e, fracture_struc_array, element_struc_array, work_array[i]
                    )
                elif e["_type"] == 2:  # Well
                    e["error"] = 0.0
                elif e["_type"] == 3:  # Constant head line
                    hpc_const_head_line.solve(
                        e, fracture_struc_array, element_struc_array, work_array[i]
                    )
                elif e["_type"] == 4:  # Impermeable circle
                    hpc_imp_object.solve_circle(
                        e, fracture_struc_array, element_struc_array, work_array[i]
                    )
                elif e["_type"] == 5:  # Impermeable line
                    hpc_imp_object.solve_line(
                        e, fracture_struc_array, element_struc_array, work_array[i]
                    )
                coefs = work_array[i]["coef"][: e["ncoef"]]
                coef0 = np.mean(np.abs(coefs[1:ncc])) + 1e-30
                coef1 = np.max(np.abs(coefs[-ncc:]))
                coef_ratio = coef1 / coef0
                # if e["error"] > e["error_old"] and e["error"] > e["error_old2"] and e["error"] > max_error and nit > 5:
                #    coef_ratio = max_coef_ratio + 1.0
                if (
                    np.abs(np.sum(coefs[-ncc:]) / (np.sum(coefs[:-ncc]) + 1e-30))
                    > max_coef_ratio
                ):
                    coef_ratio = max_coef_ratio + 1.0
                # The coefficients needs to be reset to zero if the number of coefficients is increased, otherwise the
                # error might grow if the discharge are computed with faulty coefficients
                if cnt == max_cnt:
                    work_array[i]["set_zero"] = True
                    work_array[i]["set_zero"] = False
                # print(f"Element {e['_id']} of type {e['_type']} has been set to zero coefficients due to high coefficient ratio.")

        for i in nb.prange(num_elements):
            e = element_struc_array[i]
            if work_array[i]["set_zero"]:
                work_array[i]["coef"][: e["ncoef"]] = (
                    work_array[i]["coef"][: e["ncoef"]] / 100.0 * 0
                )
                e["error_old"] = 1e30
            # if e["_type"] == 0:  # Intersection
            #    work_array[i]["coef"][: e["ncoef"]] = (
            #            work_array[i]["coef"][: e["ncoef"]]* 0
            #    )
            e["coef"][: e["ncoef"]] = (
                damping * work_array[i]["coef"][: e["ncoef"]]
                + (1 - damping) * e["coef"][: e["ncoef"]]
            )

        # Solve the elements
        if nit <= -2000:
            for _ in range(1):
                element_solver(
                    num_elements,
                    element_struc_array,
                    fracture_struc_array,
                    work_array,
                    max_error,
                    nit,
                    cnt_error,
                    damping,
                )

    return np.sum(work_array["set_zero"])


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
        discharge_elements,
        discharge_int,
        head_matrix,
        z_int,
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
    for i in nb.prange(len(element_struc_array)):
        element_struc_array[i]["q"] = 0.0

    # Set the constants equal to zero
    for i in nb.prange(len(fractures_struc_array)):
        fractures_struc_array[i]["constant"] = 0.0

    # Get the head matrix
    build_head_matrix(
        fractures_struc_array,
        element_struc_array,
        discharge_elements,
        discharge_int,
        head_matrix,
        z_int,
    )


@nb.njit(parallel=PARALLEL, cache=CACHE)
def post_matrix_solve(
    fractures_struc_array,
    element_struc_array,
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
    for i in nb.prange(len(discharge_elements)):
        e = discharge_elements[i]
        element_struc_array[e["_id"]]["q"] = discharges[i]
        # element_struc_array[e["_id"]]["q"] = (discharges[i] + discharges_old[i]) / 2.0

    # Set the constants for each fracture
    for i in nb.prange(len(fractures_struc_array)):
        fractures_struc_array[i]["constant"] = discharges[len(discharge_elements) + i]
        # fractures_struc_array[i]["constant"] = (discharges[len(discharge_elements) + i] + discharges_old[len(discharge_elements) + i]) / 2.0


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_old_discharges(element_struc_array, fractures_struc_array, discharge_elements):
    discharges_old = np.zeros(
        len(discharge_elements) + np.max(element_struc_array["frac0"]) + 1
    )
    for i in nb.prange(len(discharge_elements)):
        e = discharge_elements[i]
        discharges_old[i] = element_struc_array[e["_id"]]["q"]
    for i in nb.prange(len(fractures_struc_array)):
        pos = len(discharge_elements) + i
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
        for i in range(discharge_int):
            omega += hpc_fracture.calc_omega(frac0, z0[i], element_struc_array)
        omega = omega / discharge_int
        if e["_type"] == 0:  # Intersection
            frac1 = fractures_struc_array[e["frac1"]]
            z1 = z_int["z1"][j][:discharge_int]
            omega1 = 0.0 + 0.0j
            for i in range(discharge_int):
                omega1 += hpc_fracture.calc_omega(frac1, z1[i], element_struc_array)
            omega1 = omega1 / discharge_int
            head_matrix[j] = np.real(omega1) / frac1["t"] - np.real(omega) / frac0["t"]
        elif e["_type"] in [2, 3]:  # Well or Constant head line
            head_matrix[j] = e["phi"] - np.real(omega)


def build_discharge_matrix(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharge_int,
    z_int,
):
    """
    Builds the discharge matrix for the DFN and adds it to the DFN.

    """
    data, rows, cols, size = get_discharge_matrix_arrays(
        fractures_struc_array,
        element_struc_array,
        discharge_elements,
        discharge_int,
        z_int,
    )

    # create the csr sparse matrix
    matrix = sp.sparse.csc_matrix((data, (rows, cols)), shape=(size, size))

    return matrix


@nb.njit(parallel=PARALLEL, cache=CACHE)
def get_discharge_matrix_arrays(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharge_int,
    z_int,
):
    """
    Builds the discharge matrix for the DFN and adds it to the DFN.

    """
    size = discharge_elements.size + fractures_struc_array.size

    # Create a sparse matrix
    # create the row, col and data arrays
    rows_np = np.zeros(size * size, dtype=np.int64)
    cols_np = np.zeros(size * size, dtype=np.int64)
    data_np = np.zeros(size * size, dtype=np.float64)
    inds = np.zeros(size * size, dtype=np.bool)

    # Add the discharge for each discharge element
    for j in nb.prange(discharge_elements.size):
        e = discharge_elements[j]
        cnt_par = size * j
        row = j
        if e["_type"] == 0:  # Intersection
            z0 = z_int["z0"][j][:discharge_int]
            z1 = z_int["z1"][j][:discharge_int]
            f0 = fractures_struc_array[e["frac0"]]
            el0 = f0["elements"]
            for k in range(f0["nelements"]):
                ee = element_struc_array[el0[k]]
                if ee["_id"] == e["_id"] or ee["_type"] not in {
                    0,
                    2,
                    3,
                }:  # skip itself and elements not type 0,2,3
                    continue
                # add the discharge term to the matrix for each element in the first fracture
                pos = np.where(discharge_elements["_id"] == ee["_id"])[0][0]
                if ee["_type"] == 0:  # Intersection
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = hpc_fracture.head_from_phi(
                        f0, get_discharge_term(ee, z0, e["frac0"])
                    )
                    inds[cnt_par] = 1
                    cnt_par += 1
                else:
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = hpc_fracture.head_from_phi(
                        f0, get_discharge_term(ee, z0, e["frac0"])
                    )
                    inds[cnt_par] = 1
                    cnt_par += 1
            f1 = fractures_struc_array[e["frac1"]]
            el1 = f1["elements"]
            for k in range(f1["nelements"]):
                ee = element_struc_array[el1[k]]
                if ee["_id"] == e["_id"] or ee["_type"] not in {
                    0,
                    2,
                    3,
                }:  # skip itself and elements not type 0,2,3
                    continue
                # add the discharge term to the matrix for each element in the second fracture
                pos = np.where(discharge_elements["_id"] == ee["_id"])[0][0]
                if ee["_type"] == 0:  # Intersection
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = hpc_fracture.head_from_phi(
                        f1, -get_discharge_term(ee, z1, e["frac1"])
                    )
                    inds[cnt_par] = 1
                    cnt_par += 1
                else:
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = hpc_fracture.head_from_phi(
                        f1, -get_discharge_term(ee, z1, e["frac1"])
                    )
                    inds[cnt_par] = 1
                    cnt_par += 1
            pos_f0 = e["frac0"]
            rows_np[cnt_par] = row
            cols_np[cnt_par] = len(discharge_elements) + pos_f0
            data_np[cnt_par] = hpc_fracture.head_from_phi(f0, 1)
            inds[cnt_par] = 1
            cnt_par += 1
            pos_f1 = e["frac1"]
            rows_np[cnt_par] = row
            cols_np[cnt_par] = len(discharge_elements) + pos_f1
            data_np[cnt_par] = hpc_fracture.head_from_phi(f1, -1)
            inds[cnt_par] = 1
            cnt_par += 1
        else:
            z0 = z_int["z0"][j][:discharge_int]
            f0 = fractures_struc_array[e["frac0"]]
            el0 = f0["elements"]
            for k in range(f0["nelements"]):
                ee = element_struc_array[el0[k]]
                if ee["_id"] == e["_id"] or ee["_type"] not in {
                    0,
                    2,
                    3,
                }:  # skip itself and elements not type 0,2,3
                    continue
                # add the discharge term to the matrix for each element in the fracture
                pos = np.where(discharge_elements["_id"] == ee["_id"])[0][0]
                if ee["_type"] == 0:  # Intersection
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = get_discharge_term(ee, z0, e["frac0"])
                    inds[cnt_par] = 1
                    cnt_par += 1
                else:
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = get_discharge_term(ee, z0, e["frac0"])
                    inds[cnt_par] = 1
                    cnt_par += 1
            pos_f = e["frac0"]
            rows_np[cnt_par] = row
            cols_np[cnt_par] = len(discharge_elements) + pos_f
            data_np[cnt_par] = 1
            inds[cnt_par] = 1
            cnt_par += 1

    # Add the continuity of flow conditions for each fracture
    for j in nb.prange(fractures_struc_array.size):
        f = fractures_struc_array[j]
        row = discharge_elements.size + j
        cnt_par = size * discharge_elements.size + size * j
        # if there are only one (or less) discharge element in the fracture, skip
        # num_discharge_el = sum([e in discharge_elements["_id"] for e in f["elements"][: f["nelements"]]])
        # if num_discharge_el <= 1:
        #    continue
        # fill the matrix for the fractures
        for e in element_struc_array[f["elements"][: f["nelements"]]]:
            if e["_type"] in [0, 2, 3]:  # Intersection, Well, Constant head line
                # add the discharge term to the matrix for each element in the fracture
                pos = np.where(discharge_elements["_id"] == e["_id"])[0][0]
                if e["_type"] == 0:  # Intersection
                    if e["frac0"] == f["_id"]:
                        rows_np[cnt_par] = row
                        cols_np[cnt_par] = pos
                        data_np[cnt_par] = 1
                        inds[cnt_par] = 1
                        cnt_par += 1
                    else:
                        rows_np[cnt_par] = row
                        cols_np[cnt_par] = pos
                        data_np[cnt_par] = -1
                        inds[cnt_par] = 1
                        cnt_par += 1
                else:
                    rows_np[cnt_par] = row
                    cols_np[cnt_par] = pos
                    data_np[cnt_par] = 1
                    inds[cnt_par] = 1
                    cnt_par += 1

    rows = rows_np[inds]
    cols = cols_np[inds]
    data = data_np[inds]
    return data, rows, cols, size


# @nb.njit( parallel=PARALLEL, cache=CACHE)
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

    cnt_bnd = 0

    # Add the head for each discharge element
    for j in range(num_elements * 0):
        e = element_struc_array[j]
        if e["_type"] == 2:  # Well
            bnd_error[j] = 0.0
            continue
        coefs = e["coef"][: e["ncoef"]]
        coef0 = np.max(np.abs(coefs[1:2]))
        coef1 = np.max(np.abs(coefs[-2:]))
        coef_ratio_re = np.abs(np.real(coefs[-1]) / np.real(coefs[2]))
        coef_ratio_im = np.abs(np.imag(coefs[-1]) / np.imag(coefs[2]))
        coef_ratio_re = np.abs(np.real(coefs[-1]) / coef0)
        coef_ratio_im = np.abs(np.imag(coefs[-1]) / coef0)
        coef_ratio = np.nanmax([coef_ratio_re, coef_ratio_im])
        coef_ratio = coef1 / coef0
        if coef_ratio < 0.4:
            coef_ratio = 0.0
        if np.max(np.abs(coefs[1:2])) < max_error:
            coef_ratio = 0.0
        if e["_type"] in [0, 3]:  # Intersection, Constant head line
            frac0 = fracture_struc_array[e["frac0"]]
            z0 = z_int["z0"][j][:discharge_int]
            omega_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega_vec[i] = hpc_fracture.calc_omega(
                    frac0, z0[i], element_struc_array
                )
            omega = np.sum(omega_vec) / discharge_int
            if e["_type"] == 0:  # Intersection
                frac1 = fracture_struc_array[e["frac1"]]
                z1 = z_int["z1"][j][:discharge_int]
                omega1_vec = np.zeros(discharge_int, dtype=np.complex128)
                for i in range(discharge_int):
                    omega1_vec[i] = hpc_fracture.calc_omega(
                        frac1, z1[i], element_struc_array
                    )
                omega1 = np.sum(omega1_vec) / discharge_int
                head1 = np.real(omega1) / frac1["t"]
                head0 = np.real(omega) / frac0["t"]
                bnd_error[j] = np.abs((head1 - head0) / head1)
                if j == 450:
                    logger.debug(f"450: {head1}, {head0}, {bnd_error[450]}")
            else:  # Well or Constant head line
                bnd_error[j] = np.abs((e["phi"] - np.real(omega)) / e["phi"])
        elif e["_type"] == 1:  # Bounding circle
            frac0 = fracture_struc_array[e["frac0"]]
            z0 = z_int["z0"][j][:discharge_int]
            omega_vec = np.zeros(discharge_int, dtype=np.complex128)
            w_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega_vec[i] = hpc_fracture.calc_omega(
                    frac0, z0[i], element_struc_array
                )
                w_vec[i] = hpc_fracture.calc_w(frac0, z0[i], element_struc_array)
            # The angle of w_vec
            phi_min = np.min(np.real(omega_vec))
            phi_max = np.max(np.real(omega_vec))
            ids = frac0["elements"][: frac0["nelements"]]
            phi_max1, phi_min1, z_max, z_min, l_min, l_max = get_max_min_phi(
                element_struc_array,
                fracture_struc_array,
                ids,
                e["frac0"],
                z_int,
                discharge_int,
            )
            if np.abs(phi_max1 - phi_min1) < 1e-2 - 1e30:
                phi_error = 0.0
            else:
                if phi_max < phi_max1:
                    phi_max = phi_max1
                if phi_min > phi_min1:
                    phi_min = phi_min1
                phi_error = np.max(
                    [
                        np.abs((phi_max - phi_max1) / phi_max),
                        np.abs((phi_min - phi_min1) / phi_min),
                    ]
                )
            dpsi = np.imag(omega_vec[1:] - omega_vec[:-1])
            dpsi_pos = work_array[j]["element_pos"][
                : work_array[j]["len_discharge_element"]
            ]
            dpsi_corr = e["dpsi_corr"][dpsi_pos]
            corr_pos = np.floor(dpsi_pos * (discharge_int / e["nint"])).astype(int)
            corr_dpsi_corr = np.zeros(discharge_int)
            for i in range(len(dpsi_pos)):
                corr_dpsi_corr[corr_pos[i]] += dpsi_corr[i]
            dpsi = dpsi - corr_dpsi_corr[:-1]
            omega = np.abs(np.sum(np.real(omega_vec)) / discharge_int)

            rmse = np.sqrt(np.sum((dpsi - omega) ** 2) / discharge_int) * 0

            bnd_error[j] = rmse

            if phi_error > bnd_error[j]:
                bnd_error[j] = phi_error

        if bnd_error[j] < coef_ratio:
            bnd_error[j] = coef_ratio

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
    z_int : np.ndarray[dtype_z_arrays]
        The z arrays for the discharge elements
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
    for j, e in enumerate(element_struc_array[ids]):
        if e["_type"] in [0, 2, 3]:  # Well, Constant head line
            z0 = z_int["z0"][ids[j]][:discharge_int]
            frac0 = fracture_struc_array[e["frac0"]]
            endpoints = e["endpoints0"]
            if e["_type"] == 0:  # Intersection
                if e["frac1"] == frac_id:
                    z0 = z_int["z1"][ids[j]][:discharge_int]
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
def get_discharge_term(element, z, frac):
    if element["_type"] == 0:  # Intersection
        return hpc_intersection.discharge_term(element, z, frac)
    elif element["_type"] == 2:  # Well
        return hpc_well.discharge_term(element, z)
    elif element["_type"] == 3:  # Constant head line
        return hpc_const_head_line.discharge_term(element, z)
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
    if self_["_type"] == 0:  # Intersection
        self_["ncoef"] = n
        self_["nint"] = n * nint_mult
        stop = 2 * np.pi + 2 * np.pi / self_["nint"]
        self_["thetas"] = np.linspace(
            start=np.pi / (2 * self_["nint"]),
            stop=stop - stop / self_["nint"],
            num=self_["nint"],
        )
    elif self_["_type"] == 3:  # Constant Head Line
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
