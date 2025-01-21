"""
Notes
-----
This module contains the hpc_solve class.
"""
from datetime import datetime

import numpy as np
import numba as nb
from AnDFN.hpc import hpc_math_functions as mf
from AnDFN.hpc import hpc_intersection, hpc_fracture, hpc_const_head_line, hpc_well, hpc_bounding_circle, NO_PYTHON, FASTMATH


@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH, parallel=True)
def solve(fracture_struc_array, element_struc_array, discharge_matrix, discharge_int, max_error, max_iterations,
          coef_increase=1.5):
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

    print(' done!')
    # get the discharge elements
    el = np.zeros(len(element_struc_array), dtype=np.bool_)
    for i in nb.prange(len(element_struc_array)):
        match element_struc_array[i]['type_']:
            case 0: # Intersection
                el[i] = 1
            case 2: # Well
                el[i] = 1
            case 3: # Constant head line
                el[i] = 1
    discharge_elements = element_struc_array[el]

    error = error_old = np.float64(1.0)
    nit = 0
    cnt_error = 0
    while cnt_error < 2 and nit < max_iterations:
        cnt = 0
        nit += 1
        # Solve the discharge matrix
        fracture_struc_array, element_struc_array = solve_discharge_matrix(fracture_struc_array, element_struc_array,
                                                                           discharge_matrix, discharge_elements, discharge_int)


        # Solve the elements
        for i in nb.prange(len(element_struc_array)):
            e = element_struc_array[i]
            if e['error'] < max_error and nit > 3 and cnt_error==0:
                cnt += 1
                continue
            match e['type_']:
                case 0: # Intersection
                    e['coef'][:e['ncoef']], e['error'] = hpc_intersection.solve(e, fracture_struc_array, element_struc_array)
                case 1: # Bounding circle
                    e['coef'][:e['ncoef']], e['error'] = hpc_bounding_circle.solve(e, fracture_struc_array, element_struc_array)
                case 2: # Well
                    e['error'] = 0.0
                    cnt += 1
                case 3: # Constant head line
                    e['coef'][:e['ncoef']], e['error'] = hpc_const_head_line.solve(e, fracture_struc_array, element_struc_array)

        # Check the error
        errors = np.array([e['error'] for e in element_struc_array], dtype=np.float64)
        error = np.max(errors)

        # print progress
        if nit < 10:
            print(
                f'Iteration: 0{nit}, Max error: {mf.float2str(error)}, Elements in solve loop: {len(element_struc_array) - cnt}')
        else:
            print(f'Iteration: {nit}, Max error: {mf.float2str(error)}, Elements in solve loop: {len(element_struc_array) - cnt}')

        # Check the boundary error
        if 300 < nit < 1000:
            """
            Checks if the elements meet the boundary condition tolerance and increases the number of coefficients 
            if they do not.
            """
            cnt_bc = 0
            tolerance = 1e-2
            for i in nb.prange(len(element_struc_array)):
                e = element_struc_array[i]
                e['error'] = max_error * 1.0001
                bnd_error = 0.0
                match e['type_']:
                    case 0:  # Intersection
                        bnd_error = 0.0
                    case 1:  # Bounding circle
                        bnd_error = hpc_bounding_circle.check_boundary_condition(e, fracture_struc_array, element_struc_array)
                        #print(f'Bounding circle error: {mf.float2str(bnd_error)}')
                    case 2:  # Well
                        bnd_error = 0.0
                    case 3:  # Constant head line
                        bnd_error = 0.0
                if bnd_error > tolerance:
                    set_new_ncoef(e, int(e['ncoef'] * coef_increase), 2)
                    cnt_bc += 1
                    match e['type_']:
                        case 0:  # Intersection
                            e['coef'][:e['ncoef']], e['error'] = hpc_intersection.solve(e, fracture_struc_array,
                                                                                        element_struc_array)
                        case 1:  # Bounding circle
                            e['coef'][:e['ncoef']], e['error'] = hpc_bounding_circle.solve(e, fracture_struc_array,
                                                                                           element_struc_array)
                        case 2:  # Well
                            e['error'] = 0.0
                            cnt += 1
                        case 3:  # Constant head line
                            e['coef'][:e['ncoef']], e['error'] = hpc_const_head_line.solve(e, fracture_struc_array,
                                                                                           element_struc_array)
            print(f'Increased for {cnt_bc} elements')
        # Check if the error is increasing
        if error > error_old and nit > 300 and error > max_error:
            pos = np.argmax(errors)
            set_new_ncoef(element_struc_array[pos], int(element_struc_array[pos]['ncoef'] * coef_increase), 2)
            print(f'Increasing ncoef for element id: {pos}')
            e = element_struc_array[pos]
            match e['type_']:
                case 0:  # Intersection
                    e['coef'][:e['ncoef']], e['error'] = hpc_intersection.solve(e, fracture_struc_array,
                                                                                element_struc_array)
                case 1:  # Bounding circle
                    e['coef'][:e['ncoef']], e['error'] = hpc_bounding_circle.solve(e, fracture_struc_array,
                                                                                   element_struc_array)
                case 2:  # Well
                    e['error'] = 0.0
                    cnt += 1
                case 3:  # Constant head line
                    e['coef'][:e['ncoef']], e['error'] = hpc_const_head_line.solve(e, fracture_struc_array,
                                                                                   element_struc_array)
            cnt_error = 0

        error_old = error

        if error < max_error:
            cnt_error += 1
            error = 1.0

    return element_struc_array

@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def solve_discharge_matrix(fractures_struc_array, element_struc_array, discharge_matrix, discharge_elements,
                           discharge_int):
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

    # Set the discharges equal to zero
    for e in element_struc_array:
        if e['type_'] in {0, 2, 3}:  # Intersection, Well, Constant head line
            e['q'] = 0.0

    # Set the constants equal to zero
    for f in fractures_struc_array:
        f['constant'] = 0.0

    # Get the head matrix
    head_matrix = build_head_matrix(fractures_struc_array, element_struc_array, discharge_elements, discharge_int)

    # Solve the discharge matrix
    discharges = np.linalg.solve(discharge_matrix, head_matrix)

    # Set the discharges for each element
    for i, e in enumerate(discharge_elements):
        element_struc_array[e['id_']]['q'] = discharges[i]

    # Set the constants for each fracture
    for i, f in enumerate(fractures_struc_array):
        fractures_struc_array[i]['constant'] = discharges[len(discharge_elements) + i]

    return fractures_struc_array, element_struc_array

@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def build_head_matrix(fractures_struc_array, element_struc_array, discharge_elements, discharge_int):
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

    Returns
    -------
    matrix : np.ndarray
        The head matrix
    """
    size = discharge_elements.size + fractures_struc_array.size
    matrix = np.zeros(size)

    # Add the head for each discharge element
    row = 0
    for e in discharge_elements:
        frac0 = fractures_struc_array[e['frac0']]
        match e['type_']:
            case 0: # Intersection
                frac1 = fractures_struc_array[e['frac1']]
                z0 = hpc_intersection.z_array(e, discharge_int, e['frac0'])
                z1 = hpc_intersection.z_array(e, discharge_int, e['frac1'])
                omega0 = hpc_fracture.calc_omega(frac0, z0, element_struc_array)
                omega1 = hpc_fracture.calc_omega(frac1, z1, element_struc_array)
                matrix[row] = np.mean(np.real(omega1)) / frac1['t'] - np.mean(np.real(omega0)) / frac0['t']
            case 2: # Well
                z = hpc_well.z_array(e, discharge_int)
                omega = hpc_fracture.calc_omega(frac0, z, element_struc_array)
                matrix[row] = e['phi'] - np.mean(np.real(omega))
            case 3: # Constant head line
                z = hpc_const_head_line.z_array(e, discharge_int)
                omega = hpc_fracture.calc_omega(frac0, z, element_struc_array)
                matrix[row] = e['phi'] - np.mean(np.real(omega))
        row += 1
    return matrix

@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
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