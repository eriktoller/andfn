"""
Notes
-----
This module contains the solver for the DFN using a consolidated and compiled version.
"""

import numpy as np
import numba as nb
from . import math_functions as mf
from . import geometry_functions as gf

########################################################################################################################
#                      INTERSECTION ELEMENT                                                                            #
########################################################################################################################

def intersection_calc_omega(element_array, z, frac_is):
    # Se if function is in the first or second fracture that the intersection is associated with
    if frac_is == element_array['frac0']:
        chi = gf.map_z_line_to_chi(z, element_array['endpoints0'])
        omega = mf.asym_expansion(chi, element_array['coef'], offset=0) + mf.well_chi(chi, element_array['q'])
    else:
        chi = gf.map_z_line_to_chi(z, element_array['endpoints1'])
        omega = mf.asym_expansion(chi, -element_array['coef'], offset=0) + mf.well_chi(chi, -element_array['q'])
    return omega