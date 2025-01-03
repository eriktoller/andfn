"""Copyright (C), 2024, Erik A. L. Toller.

Erik A. L. Toller, WSP Sverige AB
erik dot toller at wsp dot com

AnDFN is a computer program that calculated the flow in a discrete fracture network (DFN) using the Analytic Element
Method (AEM).
"""

# version number
__name__ = "AnDFN"
__author__ = "Erik A.L. Toller"
__version__ = "0.1"

# Import all classes and functions
from AnDFN.bounding import BoundingCircle
from AnDFN.const_head import ConstantHeadLine
from AnDFN.dfn import DFN
from AnDFN.fracture import Fracture
from AnDFN.intersection import Intersection
from AnDFN.well import Well
import AnDFN.geometry_functions
import AnDFN.math_functions

__all__ = [
    'BoundingCircle',
    'ConstantHeadLine',
    'DFN',
    'Fracture',
    'Intersection',
    'Well'
]

# data type for structured arrays
import numpy as np

dfn_dtype = np.dtype([
        ('label', np.str_),
        ('id', np.int_),
        ('frac0', np.str_),
        ('frac1', np.str_),
        ('endpoints0', np.ndarray),
        ('endpoints1', np.ndarray),
        ('radius', np.float64),
        ('center', np.complex128),
        ('head', np.float64),
        ('ncoef', np.int_),
        ('nint', np.int_),
        ('q', np.float64),
        ('thetas', np.ndarray),
        ('coef', np.ndarray),
        ('old_coef', np.ndarray),
        ('dpsi_corr', np.ndarray),
        ('error', np.float64)
])
