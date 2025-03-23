
import numpy as np
import numba as nb
from . import hpc_math_functions as mf
from . import hpc_geometry_functions as gf
from andfn.hpc import NO_PYTHON, hpc_fracture


@nb.jit(nopython=NO_PYTHON, inline='always')
def get_chi(self_, z):
    """
    Maps the complex z plane to the complex chi plane for the bounding circle.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z : complex

    Returns
    -------
    chi : complex
        The mapped point in the chi plane.
    """
    chi = gf.map_z_circle_to_chi(z, self_['radius'])
    return chi


@nb.jit(nopython=NO_PYTHON, inline='always')
def z_array(self_, n):
    """
    Returns an array of points in the complex z plane.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    n : int
        The number of points to return.

    Returns
    -------
    z : np.ndarray[complex]
        The array of points in the complex z plane.
    """
    theta = mf.calc_thetas(n, self_['type_'])
    chi = np.exp(1j * theta)
    z = gf.map_chi_to_z_circle(chi, self_['radius'])
    return z

@nb.jit(nopython=NO_PYTHON, inline='always')
def calc_omega(self_, z):
    """
    Calculates the omega for the bounding circle.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z : complex
        A point in the complex z plane.

    Returns
    -------
    omega : complex
        The complex potential for the bounding circle.
    """
    chi = get_chi(self_, z)
    omega = mf.taylor_series(chi, self_['coef'][:self_['ncoef']])
    return omega

#@nb.jit(nopython=NO_PYTHON, inline='always')
def calc_w(self_, z):
    """
    Calculates the omega for the bounding circle.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    z : complex
        A point in the complex z plane.

    Returns
    -------
    omega : complex
        The complex potential for the bounding circle.
    """
    chi = get_chi(self_, z)
    w = mf.taylor_series_d1(chi, self_['coef'][:self_['ncoef']])
    return w


@nb.jit(nopython=NO_PYTHON)
def find_branch_cuts(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Find the branch cuts for the fracture.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    work_array : np.ndarray[dtype_work]
        The work array

    Returns
    -------
    dpsi_corr : np.ndarray[np.float64]
        The correction to the potential due to the branch cuts
    """
    # Find the branch cuts
    z_pos = gf.map_chi_to_z_circle(np.exp(1j * self_['thetas'][:self_['nint']]), self_['radius'])
    dpsi_corr = np.zeros(self_['nint'] - 1, dtype=float)

    nel = fracture_struc_array[self_['frac0']]['nelements']
    elements_list = fracture_struc_array[self_['frac0']]['elements'][:nel]
    elements = element_struc_array[elements_list]
    work_array['len_discharge_element'] = 0

    cnt = 0
    for ii in range(self_['nint'] - 1):
        for e in elements:
            if e['type_'] == 0:  # Intersection
                if e['frac0'] == self_['frac0']:
                    chi0 = gf.map_z_line_to_chi(z_pos[ii], e['endpoints0'])
                    chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e['endpoints0'])
                    ln0 = np.imag(np.log(chi0))
                    ln1 = np.imag(np.log(chi1))
                    if np.sign(ln0) != np.sign(ln1) and np.abs(ln0) + np.abs(ln1) > np.pi:
                        dpsi_corr[ii] -= e['q']
                        work_array['element_pos'][cnt] = ii
                        work_array['discharge_element'][cnt] = e['id_']
                        work_array['sign_array'][cnt] = -1
                        work_array['len_discharge_element'] += 1
                        cnt += 1
                else:
                    chi0 = gf.map_z_line_to_chi(z_pos[ii], e['endpoints1'])
                    chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e['endpoints1'])
                    ln0 = np.imag(np.log(chi0))
                    ln1 = np.imag(np.log(chi1))
                    if np.sign(ln0) != np.sign(ln1) and np.abs(ln0) + np.abs(ln1) > np.pi:
                        dpsi_corr[ii] += e['q']
                        work_array['element_pos'][cnt] = ii
                        work_array['discharge_element'][cnt] = e['id_']
                        work_array['sign_array'][cnt] = 1
                        work_array['len_discharge_element'] += 1
                        cnt += 1
            elif e['type_'] == 2:  # Well
                chi0 = gf.map_z_circle_to_chi(z_pos[ii], e['radius'], e['center'])
                chi1 = gf.map_z_circle_to_chi(z_pos[ii + 1], e['radius'], e['center'])
                if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                    dpsi_corr[ii] -= e['q']
                    work_array['element_pos'][cnt] = ii
                    work_array['discharge_element'][cnt] = e['id_']
                    work_array['sign_array'][cnt] = -1
                    work_array['len_discharge_element'] += 1
                    cnt += 1
            elif e['type_'] == 3: # Constant head line
                chi0 = gf.map_z_line_to_chi(z_pos[ii], e['endpoints0'])
                chi1 = gf.map_z_line_to_chi(z_pos[ii + 1], e['endpoints0'])
                if np.sign(np.imag(chi0)) != np.sign(np.imag(chi1)) and np.real(chi0) < 0:
                    dpsi_corr[ii] -= e['q']
                    work_array['element_pos'][cnt] = ii
                    work_array['discharge_element'][cnt] = e['id_']
                    work_array['sign_array'][cnt] = -1
                    work_array['len_discharge_element'] += 1
                    cnt += 1

@nb.jit(nopython=NO_PYTHON)
def get_dpsi_corr(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Get the correction to the stream function due to the branch cuts.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    work_array : np.ndarray[dtype_work]
        The work array

    Returns
    -------
    Edits the self_ array in place.
    """
    if work_array['len_discharge_element'] == 0:
        find_branch_cuts(self_, fracture_struc_array, element_struc_array, work_array)
    # set dpsi_corr to zero
    self_['dpsi_corr'][:self_['nint']-1] = 0.0
    for i in range(work_array['len_discharge_element']):
        e = element_struc_array[work_array['discharge_element'][i]]
        self_['dpsi_corr'][work_array['element_pos'][i]] += e['q'] * work_array['sign_array'][i]


@nb.jit(nopython=NO_PYTHON)
def solve(self_, fracture_struc_array, element_struc_array, work_array):
    """
    Solves the bounding circle element.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    work_array : np.ndarray[dtype_work]
        The work array

    Returns
    -------
    Edits the self_ array and works_array in place.
    """

    get_dpsi_corr(self_, fracture_struc_array, element_struc_array, work_array)
    frac0 = fracture_struc_array[self_['frac0']]
    work_array['old_coef'][:self_['ncoef']] = self_['coef'][:self_['ncoef']]
    mf.cauchy_integral_domega(self_['nint'], self_['ncoef'], self_['thetas'][:self_['nint']], self_['dpsi_corr'][:self_['nint']-1],
                                   frac0, self_['id_'], element_struc_array,
                                   self_['radius'], work_array, work_array['coef'][:self_['ncoef']])
    work_array['coef'][:self_['ncoef']] = -work_array['coef'][:self_['ncoef']]
    #self_['error'] = np.max(np.abs(work_array['coef'][:self_['ncoef']] - work_array['old_coef'][:self_['ncoef']]))
    self_['error_old'] = self_['error']
    self_['error'] = mf.calc_error(work_array['coef'][:self_['ncoef']], work_array['old_coef'][:self_['ncoef']])


@nb.jit(nopython=NO_PYTHON)
def check_boundary_condition(self_, fracture_struc_array, element_struc_array, n=10):
    """
    Check if the bounding circle satisfies the boundary conditions.

    Parameters
    ----------
    self_ : np.ndarray[element_dtype]
        The bounding circle element
    fracture_struc_array : np.ndarray[fracture_dtype]
        The array of fractures
    element_struc_array : np.ndarray[element_dtype]
        The array of elements
    n : int
        The number of points to check the boundary condition

    Returns
    -------
    error : np.float64
        The error in the boundary condition
    """

    # Calculate the stream function on the boundary of the fracture
    chi = np.exp(1j * self_['thetas'])
    z = gf.map_chi_to_z_circle(chi, self_['radius'])
    frac0 = fracture_struc_array[self_['frac0']]
    omega0 = hpc_fracture.calc_omega(frac0, z, element_struc_array)
    psi = np.imag(omega0)
    dpsi = psi[1:] - psi[:-1]
    q = np.sum(np.abs(self_['dpsi_corr'][:self_['nint']-1]))
    mean_dpsi = np.abs(np.max(dpsi) - np.min(dpsi))
    #if mean_dpsi > q/2:
    #    mean_dpsi = np.abs(np.abs(np.max(dpsi) - np.min(dpsi)) - q)
    if q < 1e-10:
        return np.abs(np.max(psi) - np.min(psi))
    return mean_dpsi / q
