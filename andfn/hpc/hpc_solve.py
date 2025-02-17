"""
Notes
-----
This module contains the hpc_solve class.
"""
import time

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
    print('Compiling HPC code...', end='')
    discharge_elements = get_discharge_elements(element_struc_array)


    # Allocate memory for the work array
    num_elements = len(element_struc_array)
    work_array = np.zeros(num_elements, dtype=dtype_work)
    # head matrix
    size = discharge_elements.size + fracture_struc_array.size
    head_matrix = np.zeros(size)
    discharges = np.zeros(size)
    discharges_old = np.zeros(size)
    z_int = np.zeros(num_elements, dtype=dtype_z_arrays)
    get_z_int_array(z_int, discharge_elements, discharge_int)

    # solve first iteration to compile the code
    solve_discharge_matrix(fracture_struc_array, element_struc_array, discharge_matrix, discharge_elements,
                           discharge_int, head_matrix, discharges, z_int)
    element_solver(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, 0, 0, 0)
    print(' done!')

    error = error_old = np.float64(1.0)
    nit = 0
    cnt_error = 0
    error_q = 1.0
    start = time.time()
    while cnt_error < 2 and nit < max_iterations:
        cnt = 0
        nit += 1
        # Solve the discharge matrix
        if error_q > max_error or cnt_error > 0:
            discharges_old[:] = discharges[:]
            solve_discharge_matrix(fracture_struc_array, element_struc_array,discharge_matrix, discharge_elements,
                                   discharge_int, head_matrix, discharges, z_int)
            error_q = np.max(np.abs(discharges - discharges_old))


        # Solve the elements
        element_solver(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, nit, cnt_error, cnt)


        # Check the error
        errors = np.array([e['error'] for e in element_struc_array], dtype=np.float64)
        error = np.max(errors)

        # print progress
        if nit < 10:
            print(
                f'Iteration: 0{nit}, Max error: {mf.float2str(error)}, Error Q: {mf.float2str(error_q)}, Elements in solve loop: {len(element_struc_array) - cnt}')
        else:
            print(f'Iteration: {nit}, Max error: {mf.float2str(error)}, Error Q: {mf.float2str(error_q)}, Elements in solve loop: {len(element_struc_array) - cnt}')

        error_old = error

        if error < max_error:
            cnt_error += 1
            error = 1.0

    print(f'Solve time: {time.time() - start}')
    return element_struc_array

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
def element_solver(num_elements, element_struc_array, fracture_struc_array, work_array, max_error, nit, cnt_error, cnt):

    # Solve the elements
    for i in nb.prange(num_elements):
        e = element_struc_array[i]
        if e['error'] < max_error and nit > 3 and cnt_error == 0:
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


def solve_discharge_matrix(fractures_struc_array, element_struc_array, discharge_matrix, discharge_elements,
                           discharge_int, head_matrix, discharges, z_int):
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
    pre_matirx_solve(fractures_struc_array, element_struc_array, discharge_elements, discharge_int, head_matrix, z_int)

    # Solve the discharge matrix
    discharges[:] = sp.sparse.linalg.spsolve(discharge_matrix, head_matrix)

    # post solver
    post_matrix_solve(fractures_struc_array, element_struc_array, discharge_elements, discharges)

@nb.jit(nopython=NO_PYTHON, parallel=PARALLEL, cache=True)
def pre_matirx_solve(fractures_struc_array, element_struc_array, discharge_elements,
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
        match e['type_']:
            case 0: # Intersection
                frac1 = fractures_struc_array[e['frac1']]
                z1 = z_int['z1'][j][:discharge_int]
                omega1_vec = np.zeros(discharge_int, dtype=np.complex128)
                for i in range(discharge_int):
                    omega1_vec[i] = hpc_fracture.calc_omega(frac1, z1[i], element_struc_array) / discharge_int
                omega1 = np.sum(omega1_vec)
                head_matrix[j] = np.real(omega1) / frac1['t'] - np.real(omega) / frac0['t']
            case 2 | 3: # Well or Constant head line
                head_matrix[j] = e['phi'] - np.real(omega)

@nb.jit(nopython=NO_PYTHON)
def get_z_int_array(z_int, discharge_elements, discharge_int):
    # Add the head for each discharge element
    for j in range(discharge_elements.size):
        e = discharge_elements[j]
        match e['type_']:
            case 0:  # Intersection
                z_int['z0'][j][:discharge_int] = hpc_intersection.z_array(e, discharge_int, e['frac0'])
                z_int['z1'][j][:discharge_int] = hpc_intersection.z_array(e, discharge_int, e['frac1'])
            case 2:  # Well
                z_int['z0'][j][:discharge_int] = hpc_well.z_array(e, discharge_int)
            case 3:  # Constant head line
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
    match self_['type_']:
        case 0:  # Intersection
            self_['ncoef'] = n
            self_['nint'] = n * nint_mult
            stop = 2 * np.pi + 2 * np.pi / self_['nint']
            self_['thetas'] = np.linspace(start=np.pi / (2 * self_['nint']), stop=stop - stop/self_['nint'],
                                      num=self_['nint'])
        case 3:  # Constant Head Line
            self_['ncoef'] = n
            self_['nint'] = n * nint_mult
            stop = 2 * np.pi + 2 * np.pi / self_['nint']
            self_['thetas'] = np.linspace(start=np.pi / (2 * self_['nint']), stop=stop - stop / self_['nint'],
                                          num=self_['nint'])
        case 1:  # Bounding Circle
            self_['ncoef'] = n
            self_['nint'] = n * nint_mult
            self_['thetas'][:self_['nint']] = np.linspace(start=0, stop=2 * np.pi - 2 * np.pi/self_['nint'],
                                                          num=self_['nint'])
