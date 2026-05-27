"""
Notes
-----
This module contains the HPC solve functions.
"""

import logging
import time

import numpy as np
import numba as nb
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
    element_struc_array : np.ndarray[element_dtype]
        The array of elements

    """
    # Get the constants, this is necessary for Numba parallelization to work
    max_error = constants["MAX_ERROR"]
    max_iterations = constants["MAX_ITERATIONS"]
    damping = float(constants["DAMPING"])

    # get the discharge elements
    logger.info("Compiling HPC code...")

    # Allocate memory for the work array
    num_elements = len(element_struc_array)
    work_array = np.zeros(num_elements, dtype=dtype_work)

    z_int = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int, element_struc_array, discharge_int)
    z_int_error = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int_error, element_struc_array, discharge_int)

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

    error = np.float64(1.0)
    nit = 0
    cnt_error = 0
    error_q = 1.0
    start = time.time()
    sum_timee = 0.0
    sum_timeq = 0.0
    while cnt_error < 2 and nit < max_iterations:
        nit += 1

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

        # After the while loop, before get_bnd_error:
        calc_fracture_constant(
            fracture_struc_array,
            element_struc_array,
            error_struc_array,
            work_array,
            discharge_int,
            z_int,
        )

        error, _id = get_error(error_struc_array)
        error_coef = np.max(error_struc_array["error_coef"])

        # print info
        if nit < 10:
            logger.info(
                f"Iteration: 0{nit}, Error E: {error:.3e}, Coef: {error_coef:.3e}, Q: {error_q:.3e}, Element: {_id}, N of Coef: {element_struc_array[_id]['ncoef']}, Type: {element_struc_array[_id]['_type']}"
            )
        else:
            logger.info(
                f"Iteration: {nit}, Error E: {error:.3e}, Coef: {error_coef:.3e}, Q: {error_q:.3e}, Element: {_id}, N of Coef: {element_struc_array[_id]['ncoef']}, Type: {element_struc_array[_id]['_type']}"
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

    bnd_error = np.zeros((num_elements, 7), dtype=np.float64)

    get_bnd_error(
        num_elements,
        fracture_struc_array,
        element_struc_array,
        error_struc_array,
        work_array,
        discharge_int,
        bnd_error,
        z_int,
    )

    return error_struc_array, work_array


def calc_fracture_constant(
    fracture_struc_array,
    element_struc_array,
    error_struc_array,
    work_array,
    discharge_int,
    z_int,
):
    """
    Calculate the constant for the fractures.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    error_struc_array : np.ndarray[element_dtype]
        Array of elements to compute the error for
    work_array : np.ndarray[dtype_work]
        The work array
    discharge_int : int
        The number of integration points

    Returns
    -------
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures with the constant calculated

    """
    for i in range(len(fracture_struc_array)):
        frac = fracture_struc_array[i]
        nel = frac["nelements"]
        nint = discharge_int
        imag_const = 0.0
        real_const = 0.0
        return

        for e in element_struc_array[frac["elements"][:nel]]:
            el_id = e["_id"]
            if e["_type"] == 1:  # Bounding circle
                z_pos = z_int["z0"][el_id][:nint]

                mf.find_branch_cuts(
                    e,
                    z_pos,
                    fracture_struc_array,
                    element_struc_array,
                    work_array[el_id],
                    nint,
                )

                dpsi_corr = np.zeros(nint)
                for k in range(work_array[el_id]["len_discharge_element"]):
                    ek = element_struc_array[work_array[el_id]["discharge_element"][k]]
                    pos = int(work_array[el_id]["element_pos"][k])
                    dpsi_corr[pos] += ek["q"] * work_array[el_id]["sign_array"][k]

                omega_pts = np.zeros(nint, dtype=np.complex128)
                for ii in range(nint):
                    omega_pts[ii] = hpc_fracture.calc_omega(
                        frac, z_pos[ii], element_struc_array
                    )

                omega_er = np.zeros(nint, dtype=np.complex128)
                for ii in range(nint):
                    omega_er[ii] = hpc_fracture.calc_omega_error(
                        frac, z_pos[ii], error_struc_array
                    )

                psi = np.zeros(nint)
                psi[0] = np.imag(omega_pts[0])
                for ii in range(nint - 1):
                    raw_dpsi = np.imag(omega_pts[ii + 1]) - np.imag(omega_pts[ii])
                    psi[ii + 1] = psi[ii] + (raw_dpsi - dpsi_corr[ii])

                imag_const = np.mean(psi + np.imag(omega_er) * 0)

            elif e["_type"] in [0, 3]:  # Intersection, Constant head line
                z0 = z_int["z0"][el_id][:discharge_int]
                omega_vec = np.zeros(discharge_int, dtype=np.complex128)
                for ii in range(discharge_int):
                    omega_vec[ii] = hpc_fracture.calc_omega(
                        frac, z0[ii], element_struc_array
                    )

                omega_er = np.zeros(discharge_int, dtype=np.complex128)
                for ii in range(discharge_int):
                    omega_er[ii] = hpc_fracture.calc_omega_error(
                        frac, z0[ii], error_struc_array
                    )

                if e["_type"] == 0:  # Intersection
                    frac1 = fracture_struc_array[e["frac1"]]
                    z1 = z_int["z1"][el_id][:discharge_int]
                    omega1_vec = np.zeros(discharge_int, dtype=np.complex128)
                    for ii in range(discharge_int):
                        omega1_vec[ii] = hpc_fracture.calc_omega(
                            frac1, z1[ii], element_struc_array
                        )

                    omega_er1 = np.zeros(discharge_int, dtype=np.complex128)
                    for ii in range(discharge_int):
                        omega_er1[ii] = hpc_fracture.calc_omega_error(
                            frac1, z1[ii], error_struc_array
                        )

                    real_const = np.mean(
                        np.real(omega_vec)
                        - np.real(omega1_vec)
                        + np.real(omega_er) * 0
                        - np.real(omega_er1) * 0
                    )
                else:  # Constant head line
                    real_const = np.mean(
                        np.real(omega_vec) - e["phi"] + np.real(omega_er) * 0
                    )

        frac["error_constant"] = real_const + 1j * imag_const


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

    # Add the head for each discharge element
    for j in range(num_elements):
        e = element_struc_array[j]
        nint = int(e["nint"])
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
                for ii in range(nint):
                    chi = work_array["exp_array_p"][j][ii]
                    z0 = gf.map_chi_to_z_line(chi, e["endpoints0"])
                    omega0 = hpc_fracture.calc_omega(frac0, z0, element_struc_array)
                    omega_error0 = hpc_fracture.calc_omega_error(
                        frac0, z0, error_struc_array, e["_id"]
                    )
                    z1 = gf.map_chi_to_z_line(chi, e["endpoints1"])
                    omega1 = hpc_fracture.calc_omega(frac1, z1, element_struc_array)
                    omega_error1 = hpc_fracture.calc_omega_error(
                        frac1, z1, error_struc_array
                    )
                    omega_er[ii] = omega_error0 - omega_error1
                    dphi[ii] = (np.real(omega0) - np.real(omega1)) + (
                        np.real(omega_error0) - np.real(omega_error1)
                    )
                    dphi_only[ii] = np.real(omega1) - np.real(omega0)

                phi_const = np.mean(dphi_only - omega_er.real)
                omega_er -= phi_const

                import matplotlib.pyplot as plt

                plt.figure()
                plt.title(
                    f"Boundary condition error for element {j} (type {e['_type']})"
                )
                plt.plot(dphi_only, label="BC")
                plt.plot(-omega_er.real, label="E(z)")
                plt.legend()
                plt.show()
            else:  # Well or Constant head line
                omega_er = np.zeros(nint, dtype=np.complex128)
                for ii in range(nint):
                    chi = work_array["exp_array_p"][j][ii]
                    z = gf.map_chi_to_z_line(chi, e["endpoints0"])
                    omega = hpc_fracture.calc_omega(frac0, z, element_struc_array)
                    omega_error = hpc_fracture.calc_omega_error(
                        frac0, z, error_struc_array
                    )
                    omega_er[ii] = omega_error
                    dphi[ii] = e["phi"] - np.real(omega) + np.real(omega_error)
                    dphi_only[ii] = e["phi"] - np.real(omega)

                phi_const = np.mean(dphi_only - omega_er.real)
                omega_er -= phi_const
                import matplotlib.pyplot as plt

                plt.figure()
                plt.title(
                    f"Boundary condition error for element {j} (type {e['_type']})"
                )
                plt.plot(dphi_only, label="BC")
                plt.plot(-omega_er.real, label="E(z)")
                plt.legend()
                plt.show()
        elif e["_type"] == 1:  # Bounding circle
            dpsi_corr = e["dpsi_corr"][: nint - 1]
            dpsi = np.zeros(nint, dtype=np.float64)
            dpsi_only = np.zeros(nint, dtype=np.float64)
            om_error = np.zeros(nint, dtype=np.complex128)
            for ii in range(nint):
                chi = work_array["exp_array_p"][j][ii]
                z = gf.map_chi_to_z_circle(chi, e["radius"], e["center"])
                omega = hpc_fracture.calc_omega(frac0, z, element_struc_array)
                omega_error = hpc_fracture.calc_omega_error(frac0, z, error_struc_array)
                om_error[ii] = omega_error
                work_array["psi"][j][ii] = np.imag(omega)
            delta_psi = work_array["psi"][j][1:nint] - work_array["psi"][j][: nint - 1]
            work_array["dpsi"][j][0] = (
                0.0  # Add this line to set the first value of dpsi to zero
            )
            work_array["dpsi"][j][1:nint] = delta_psi - dpsi_corr

            psi0 = work_array["psi"][j][0]
            for ii in range(nint):
                psi1 = psi0 + work_array["dpsi"][j][ii]
                work_array["psi"][j][ii] = psi1
                psi0 = psi1
                dpsi[ii] = work_array["psi"][j][ii]
                dpsi_only[ii] = dpsi[ii]

            import matplotlib.pyplot as plt

            plt.figure()
            plt.title(f"Boundary condition error for element {j} (type {e['_type']})")
            plt.plot(dpsi_only, label="BC")
            plt.plot(-om_error.imag, label="E(z)")
            plt.plot(-om_error.real, label="Re E(z)")
            plt.legend()
            plt.show()
