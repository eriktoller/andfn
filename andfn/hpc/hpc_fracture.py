"""
Notes
-----
This module contains the HPC fracture functions.
"""

import numpy as np
import numba as nb
from andfn.hpc import hpc_intersection, hpc_const_head_line, hpc_well, hpc_bounding_circle, NO_PYTHON, FASTMATH
from andfn.element import element_dtype, element_index_dtype, fracture_dtype, fracture_index_dtype


@nb.jit(nopython=NO_PYTHON, fastmath=FASTMATH)
def calc_omega(self_, z, element_struc_array, exclude=-1):
    """
    Calculates the omega for the fracture excluding element "exclude".

    Parameters
    ----------
    self_ : np.ndarray fracture_dtype

    z : np.ndarray
        A point in the complex z plane.
    element_struc_array : np.ndarray[element_dtype]
        Array of elements.
    exclude : int
        Label of element to exclude from the omega calculation.

    Returns
    -------
    omega : np.ndarray
        The complex potential for the fracture.
    """
    omega = np.full(z.size, self_['constant'], dtype=np.complex128)

    for e in range(self_['nelements']):
        el = self_['elements'][e]
        if el != exclude:
            element = element_struc_array[el]
            match element['type_']:
                case 0:  # Intersection
                    omega += hpc_intersection.calc_omega(element, z, self_['id_'])
                case 1:  # Bounding circle
                    omega += hpc_bounding_circle.calc_omega(element, z)
                case 2:  # Well
                    omega += hpc_well.calc_omega(element, z, self_['t'])
                case 3:  # Constant head line
                    omega += hpc_const_head_line.calc_omega(element, z)

    return omega
