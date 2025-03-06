"""
Notes
-----
This module contains the HPC solve functions.
"""
import time
from zipfile import error

import numpy as np
import numba as nb
import scipy as sp
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_intersection, hpc_fracture, hpc_const_head_line, hpc_well, hpc_bounding_circle, NO_PYTHON, PARALLEL
from andfn.element import MAX_NCOEF, MAX_ELEMENTS

dtype_work = np.dtype([
        ('phi', np.float64, MAX_NCOEF * 2),
        ('psi', np.float64, MAX_NCOEF * 2),
        ('coef', np.complex128, MAX_NCOEF),
        ('coef0', np.complex128, MAX_NCOEF),
        ('coef1', np.complex128, MAX_NCOEF),
        ('old_coef', np.complex128, MAX_NCOEF),
        ('dpsi', np.float64, MAX_NCOEF * 2),
        ('error', np.float64),
        ('integral', np.complex128, MAX_NCOEF),
        ('sign_array', np.int64, MAX_ELEMENTS),
        ('discharge_element', np.int64, MAX_ELEMENTS),
        ('element_pos', np.int64, MAX_ELEMENTS),
        ('len_discharge_element', np.int64)
    ])

dtype_z_arrays = np.dtype([
        ('z0', complex, MAX_ELEMENTS),
        ('z1', complex, MAX_ELEMENTS)
    ])

MAX_COEF = 50
MULTIPLIER = 1.2


def solve(fracture_struc_array, element_struc_array, discharge_matrix, discharge_int, max_error, max_iterations):
    """
    Solves the DFN.

    Parameters
    ----------
    fracture_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_matrix : np.ndarray
        The discharge matrix
    discharge_int : int
        The number of integration points
    max_error : np.float64
        The maximum error allowed
    max_iterations : int
        The maximum number of iterations

    Returns
    -------
    element_struc_array : np.ndarray[element_dtype]
        The array of elements

    """

    # get the discharge elements
    print('Compiling HPC code...')
    discharge_elements = get_discharge_elements(element_struc_array)


    # Allocate memory for the work array
    num_elements = len(element_struc_array)
    work_array = np.zeros(num_elements, dtype=dtype_work)
    # head matrix
    size = discharge_elements.size + fracture_struc_array.size
    head_matrix = np.zeros(size)
    bnd_error = np.zeros(num_elements)
    discharges = np.zeros(size)
    discharges_old = np.zeros(size)
    z_int = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int, discharge_elements, discharge_int)
    z_int_error = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int_error, element_struc_array, discharge_int)

    # LU-factorization
    #discharge_matrix = sp.sparse.csc_matrix(discharge_matrix)
    lu_matrix = sp.sparse.linalg.splu(discharge_matrix)

    # Set old error
    for i in nb.prange(num_elements):
        e = element_struc_array[i]
        e['error_old'] = 1e30
        e['error'] = 1e30

    # solve once to get the initial discharges
    t = mf.calc_thetas(element_struc_array[0]['nint'], element_struc_array[0]['type_'])
    #solve_discharge_matrix(fracture_struc_array, element_struc_array, discharge_matrix, discharge_elements,
    #                               discharge_int, head_matrix, discharges, z_int, lu_matrix)
    #cnt = element_solver2(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, 0,
    #                      0)

    error = error_old = np.float64(1.0)
    nit = 0
    cnt_error = 0
    error_q = 1.0
    start = time.time()
    while cnt_error < 2 and nit < max_iterations:
        nit += 1
        # Solve the discharge matrix
        startQ = time.time()
        if error_q > max_error or cnt_error > 0:
            discharges_old[:] = discharges[:]
            solve_discharge_matrix(fracture_struc_array, element_struc_array,discharge_matrix, discharge_elements,
                                   discharge_int, head_matrix, discharges, z_int, lu_matrix)
            error_q = np.max(np.abs(discharges - discharges_old))
        print(f'Solve Q time: {time.time() - startQ}')

        # Solve the elements
        StartE = time.time()
        cnt = element_solver2(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, nit, cnt_error)
        print(f'Solve E time: {time.time() - StartE}')

        error, id_ = get_error(element_struc_array)

        # print progress
        if nit < 10:
            print(f'Iteration: 0{nit}, Max error: {mf.float2str(error)}, Error Q: {mf.float2str(error_q)}, '
                  f'Elements in solve loop: {len(element_struc_array) - cnt}')
        else:
            print(f'Iteration: {nit}, Max error: {mf.float2str(error)}, Error Q: {mf.float2str(error_q)}, '
                  f'Elements in solve loop: {len(element_struc_array) - cnt}')

        cnt_bnd = get_bnd_error(num_elements, fracture_struc_array, element_struc_array, work_array, discharge_int, bnd_error, z_int_error, nit)
        print(f'Max boundary error: {np.max(bnd_error):.4e}, Element id: {np.argmax(bnd_error)} [type: '
              f'{element_struc_array[np.argmax(bnd_error)]["type_"]}, ncoef: {element_struc_array[np.argmax(bnd_error)]["ncoef"]}]')



        if cnt_bnd > 0:
            error = 1.0

        if error < max_error:
            cnt_error += 1
            error = 1.0

    print(f'Solve time: {time.time() - start}')
    for i in range(num_elements):
        e = element_struc_array[i]
        print(f'Element id: {i}, Type: {e["type_"]}, ncoef: {e["ncoef"]}, Error: {e["error"]}, bnd error: {bnd_error[i]}')
    return element_struc_array

@nb.jit(nopython=NO_PYTHON)
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
    id_ : int
        The id of the element with the maximum error
    """
    error = 0.0
    id_ = 0
    for e in element_struc_array:
        if e['error'] > error:
            error = e['error']
            id_ = e['id_']
    return error, id_

@nb.jit(nopython=NO_PYTHON, parallel=PARALLEL, cache=True)
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
    for i in range(len(element_struc_array)):
        if element_struc_array[i]['type_'] in {0, 2, 3}:  # Intersection, Well, Constant head line
            el[i] = 1
    discharge_elements = element_struc_array[el]
    return discharge_elements

@nb.jit(nopython=NO_PYTHON, parallel=True, cache=True)
def element_solver(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, nit, cnt_error):
    cnt = 0

    # Solve the elements
    for i in nb.prange(num_elements):
        e = element_struc_array[i]
        if e['error'] < max_error and nit > 30 and cnt_error == 0:
            cnt += 1
            continue
        if e['type_'] == 0:  # Intersection
            hpc_intersection.solve(e, fracture_struc_array, element_struc_array, work_array[i])
        elif e['type_'] == 1:  # Bounding circle
            hpc_bounding_circle.solve(e, fracture_struc_array, element_struc_array, work_array[i])
        elif e['type_'] == 2:  # Well
            e['error'] = 0.0
            cnt += 1
        elif e['type_'] == 3:  # Constant head line
            hpc_const_head_line.solve(e, fracture_struc_array, element_struc_array, work_array[i])

    # Get the coefficients from the work array
    for i in nb.prange(num_elements):
        e = element_struc_array[i]
        e['coef'][:e['ncoef']] = work_array[i]['coef'][:e['ncoef']]

    return cnt

@nb.jit(nopython=NO_PYTHON, parallel=True, cache=False)
def element_solver2(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, nit, cnt_error):
    error = 1.0
    nit_el = 0
    while error > max_error and nit_el < 5:
        nit_el += 1

        # Solve the elements
        cnt = element_solver(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, nit, cnt_error)

        error, id_ = get_error(element_struc_array)

        # Get the coefficients from the work array
        for i in range(num_elements):
            e = element_struc_array[i]
            if e['error'] > e['error_old'] and e['error'] > max_error and e['ncoef'] < MAX_COEF and nit_el > 100:
                e['ncoef'] = int(e['ncoef'] * MULTIPLIER)
                e['nint'] = e['ncoef'] * 2
                e['thetas'][:e['nint']] = mf.calc_thetas(e['nint'], e['type_'])
                work_array[i]['len_discharge_element'] = 0
                e['coef'][:e['ncoef']] = 0.0
                e['error'] = 1e30



        print(f'Error: {mf.float2str(error)}, Element id: {id_} [type: {element_struc_array[id_]["type_"]}, '
              f'ncoef: {element_struc_array[id_]["ncoef"]}]')

    return cnt


def solve_discharge_matrix(fractures_struc_array, element_struc_array, discharge_matrix, discharge_elements,
                           discharge_int, head_matrix, discharges, z_int, lu_matrix):
    """
    Solves the discharge matrix for the DFN and stores the discharges and constants in the elements and fractures.

    Parameters
    ----------
    fractures_struc_array : np.ndarray[fracture_dtype]
        Array of fractures
    element_struc_array : np.ndarray[element_dtype]
        Array of elements
    discharge_matrix : np.ndarray
        The discharge matrix
    discharge_elements : np.ndarray[element_dtype]
        The discharge elements
    discharge_int : int
        The number of integration points

    Returns
    -------
    fractures_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    """

    # pre solver
    start0 = time.time()
    pre_matrix_solve(fractures_struc_array, element_struc_array, discharge_elements, discharge_int, head_matrix, z_int)
    print(f'Pre solve time: {time.time() - start0}')

    # Solve the discharge matrix
    start0 = time.time()
    #spso = sp.sparse.linalg.spsolve(discharge_matrix, head_matrix)
    discharges[:] = lu_matrix.solve(head_matrix)
    #print(f'Diff: {np.sum(np.abs(spso - discharges))}')
    #discharges[:], info = sp.sparse.linalg.cgs(discharge_matrix, head_matrix)
    print(f'Solve matrix time: {time.time() - start0}')

    # post solver
    start0 = time.time()
    post_matrix_solve(fractures_struc_array, element_struc_array, discharge_elements, discharges)
    print(f'Post solve time: {time.time() - start0}')

@nb.jit(nopython=NO_PYTHON, parallel=PARALLEL, cache=True)
def pre_matrix_solve(fractures_struc_array, element_struc_array, discharge_elements,
                     discharge_int, head_matrix, z_int):
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
        e = element_struc_array[i]
        if e['type_'] in {0, 2, 3}:  # Intersection, Well, Constant head line
            e['q'] = 0.0

    # Set the constants equal to zero
    for i in nb.prange(len(fractures_struc_array)):
        fractures_struc_array[i]['constant'] = 0.0

    # Get the head matrix
    build_head_matrix(fractures_struc_array, element_struc_array, discharge_elements, discharge_int, head_matrix, z_int)


@nb.jit(nopython=NO_PYTHON, parallel=PARALLEL, cache=True)
def post_matrix_solve(fractures_struc_array, element_struc_array, discharge_elements,
                        discharges):
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

    Returns
    -------
    fractures_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    """
    # Set the discharges for each element
    for i in nb.prange(len(discharge_elements)):
        e = discharge_elements[i]
        element_struc_array[e['id_']]['q'] = discharges[i]

    # Set the constants for each fracture
    for i in nb.prange(len(fractures_struc_array)):
        fractures_struc_array[i]['constant'] = discharges[len(discharge_elements) + i]

@nb.jit(nopython=NO_PYTHON, parallel=PARALLEL, cache=True)
def build_head_matrix(fractures_struc_array, element_struc_array, discharge_elements, discharge_int, head_matrix, z_int):
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
        frac0 = fractures_struc_array[e['frac0']]
        z0 = z_int['z0'][j][:discharge_int]
        omega_vec = np.zeros(discharge_int, dtype=np.complex128)
        for i in range(discharge_int):
            omega_vec[i] = hpc_fracture.calc_omega(frac0, z0[i], element_struc_array) / discharge_int
        omega = np.sum(omega_vec)
        if e['type_'] == 0:  # Intersection
            frac1 = fractures_struc_array[e['frac1']]
            z1 = z_int['z1'][j][:discharge_int]
            omega1_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega1_vec[i] = hpc_fracture.calc_omega(frac1, z1[i], element_struc_array) / discharge_int
            omega1 = np.sum(omega1_vec)
            head_matrix[j] = np.real(omega1) / frac1['t'] - np.real(omega) / frac0['t']
        elif e['type_'] in [2, 3]:  # Well or Constant head line
            head_matrix[j] = e['phi'] - np.real(omega)


#@nb.jit(nopython=NO_PYTHON, parallel=PARALLEL, cache=True)
def get_bnd_error(num_elements, fracture_struc_array, element_struc_array, work_array, discharge_int, bnd_error, z_int, nit):
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
    discharge_int : int
        The number of integration points
    bnd_error : np.ndarray[dtype_head_matrix]
        The error matrix for the boundary conditions
    z_int : np.ndarray[dtype_z_arrays]
        The z arrays for the discharge elements

    Returns
    -------
    matrix : np.ndarray
        The head matrix
    """

    cnt_bnd = 0

    # Add the head for each discharge element
    for j in range(num_elements):
        e = element_struc_array[j]
        coefs = e['coef'][:e['ncoef']]
        coef_ratio_re = np.abs(np.real(coefs[-1]) / np.real(coefs[1]))
        coef_ratio_im = np.abs(np.imag(coefs[-1]) / np.imag(coefs[1]))
        coef_ratio = np.max([coef_ratio_re, coef_ratio_im])
        if e['type_'] == 2: # Well
            bnd_error[j] = 0.0
        elif e['type_'] in [0, 3]:  # Intersection, Constant head line
            frac0 = fracture_struc_array[e['frac0']]
            z0 = z_int['z0'][j][:discharge_int]
            omega_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega_vec[i] = hpc_fracture.calc_omega(frac0, z0[i], element_struc_array)
            omega = np.sum(omega_vec) / discharge_int
            if e['type_'] == 0:  # Intersection
                frac1 = fracture_struc_array[e['frac1']]
                z1 = z_int['z1'][j][:discharge_int]
                omega1_vec = np.zeros(discharge_int, dtype=np.complex128)
                for i in range(discharge_int):
                    omega1_vec[i] = hpc_fracture.calc_omega(frac1, z1[i], element_struc_array)
                omega1 = np.sum(omega1_vec) / discharge_int
                bnd_error[j] = np.abs( (np.real(omega1) / frac1['t'] - np.real(omega) / frac0['t'] ) / (np.real(omega1) / frac1['t']) )
            else:  # Well or Constant head line
                bnd_error[j] = np.abs( (e['phi'] - np.real(omega)) / e['phi'] )
        elif e['type_'] == 1: # Bounding circle
            frac0 = fracture_struc_array[e['frac0']]
            z0 = z_int['z0'][j][:discharge_int]
            omega_vec = np.zeros(discharge_int, dtype=np.complex128)
            for i in range(discharge_int):
                omega_vec[i] = hpc_fracture.calc_omega(frac0, z0[i], element_struc_array)
            head_min = np.min(np.real(omega_vec))/frac0['t']
            head_max = np.max(np.real(omega_vec))/frac0['t']
            dpsi = np.imag(omega_vec[1:] - omega_vec[:-1])
            dpsi_pos = work_array[j]['element_pos'][:work_array[j]['len_discharge_element']]
            dpsi_corr = e['dpsi_corr'][dpsi_pos]
            corr_pos = np.floor(dpsi_pos * (discharge_int / e['nint'])).astype(int)
            corr_dpsi_corr = np.zeros(discharge_int)
            for i in range(len(dpsi_pos)):
                corr_dpsi_corr[corr_pos[i]] += dpsi_corr[i]
            dpsi = dpsi - corr_dpsi_corr[:-1]
            psi0 = np.imag(omega_vec[0])
            omega_vec[:] = 0.0
            omega_vec[0] = psi0
            for i in range(discharge_int - 1):
                omega_vec[i+1] = psi0 + dpsi[i]
                psi0 = omega_vec[i+1]
            domega = np.sum(dpsi) / discharge_int
            omega = np.abs( np.sum(np.real(omega_vec)) / discharge_int )
            q_sum = np.sum( np.abs(element_struc_array[work_array[j]['discharge_element'][:work_array[j]['len_discharge_element']]]['q'] ))
            # TODO: Get the endpoints and increase the coefficients in fractures with short elements
            denom = np.max([q_sum, omega])
            rmse = np.sqrt(np.sum((dpsi-omega)**2)/discharge_int)
            psi_ratio =  np.abs(domega/denom)
            bnd_error[j] = rmse

        if bnd_error[j] < coef_ratio:
            bnd_error[j] = coef_ratio

        if bnd_error[j] > 0.01 and e['ncoef'] < MAX_COEF and nit > 1:
            cnt_bnd += 1
            e['ncoef'] = int(e['ncoef'] * MULTIPLIER)
            e['nint'] = e['ncoef'] * 2
            e['thetas'][:e['nint']] = mf.calc_thetas(e['nint'], e['type_'])
            work_array[j]['len_discharge_element'] = 0
            e['coef'][:e['ncoef']] = 0.0
            """
            fr = fracture_struc_array[e['frac0']]
            el = element_struc_array[fr['elements'][:fr['nelements']]]
            for jj in range(fr['nelements']):
                id_is = el[jj]['id_']
                work_array[id_is]['len_discharge_element'] = 0
                if element_struc_array[id_is]['ncoef'] < MAX_COEF:
                    element_struc_array[id_is]['ncoef'] = int(element_struc_array[id_is]['ncoef'] * MULTIPLIER)
                    element_struc_array[id_is]['nint'] = element_struc_array[id_is]['ncoef'] * 2
                    element_struc_array[id_is]['thetas'][:element_struc_array[id_is]['nint']] = mf.calc_thetas(
                        element_struc_array[id_is]['nint'], element_struc_array[id_is]['type_'])
                    element_struc_array[id_is]['coef'][:element_struc_array[id_is]['ncoef']] = 0.0
                    #element_struc_array[id_is]['error'] = 1e30
                if element_struc_array[id_is]['type_'] == 10:
                    fr = fracture_struc_array[element_struc_array[id_is]['frac1']]
                    el2 = element_struc_array[fr['elements'][:fr['nelements']]]
                    for jj2 in range(fr['nelements']):
                        id_is2 = el2[jj2]['id_']
                        if id_is == id_is2:
                            continue
                        work_array[id_is2]['len_discharge_element'] = 0
                        if element_struc_array[id_is2]['ncoef'] < MAX_COEF:
                            element_struc_array[id_is2]['ncoef'] = int(element_struc_array[id_is2]['ncoef'] * MULTIPLIER)
                            element_struc_array[id_is2]['nint'] = element_struc_array[id_is2]['ncoef'] * 2
                            element_struc_array[id_is2]['thetas'][:element_struc_array[id_is2]['nint']] = mf.calc_thetas(
                                element_struc_array[id_is2]['nint'], element_struc_array[id_is]['type_'])
                            #element_struc_array[id_is2]['error'] = 1e30
            """
    return cnt_bnd

@nb.jit(nopython=NO_PYTHON)
def get_z_int_array(z_int, elements, discharge_int):
    # Add the head for each discharge element
    for j in range(elements.size):
        e = elements[j]
        if e['type_'] == 0:  # Intersection
            z_int['z0'][j][:discharge_int] = hpc_intersection.z_array(e, discharge_int, e['frac0'])
            z_int['z1'][j][:discharge_int] = hpc_intersection.z_array(e, discharge_int, e['frac1'])
        elif e['type_'] == 1:  # Bounding circle
            z_int['z0'][j][:discharge_int] = hpc_bounding_circle.z_array(e, discharge_int)
        elif e['type_'] == 2:  # Well
            z_int['z0'][j][:discharge_int] = hpc_well.z_array(e, discharge_int)
        elif e['type_'] == 3:  # Constant head line
            z_int['z0'][j][:discharge_int] = hpc_const_head_line.z_array(e, discharge_int)

@nb.jit(nopython=NO_PYTHON)
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
    if self_['type_'] == 0:  # Intersection
        self_['ncoef'] = n
        self_['nint'] = n * nint_mult
        stop = 2 * np.pi + 2 * np.pi / self_['nint']
        self_['thetas'] = np.linspace(start=np.pi / (2 * self_['nint']), stop=stop - stop/self_['nint'],
                                      num=self_['nint'])
    elif self_['type_'] == 3:  # Constant Head Line
        self_['ncoef'] = n
        self_['nint'] = n * nint_mult
        stop = 2 * np.pi + 2 * np.pi / self_['nint']
        self_['thetas'] = np.linspace(start=np.pi / (2 * self_['nint']), stop=stop - stop / self_['nint'],
                                      num=self_['nint'])
    elif self_['type_'] == 1:  # Bounding Circle
        self_['ncoef'] = n
        self_['nint'] = n * nint_mult
        self_['thetas'][:self_['nint']] = np.linspace(start=0, stop=2 * np.pi - 2 * np.pi/self_['nint'],
                                                      num=self_['nint'])
