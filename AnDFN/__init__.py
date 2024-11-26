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

elements_dtype = np.dtype([('label', (np.str_, 10)), ('array_field', np.ndarray)])
dfn_dtype = np.dtype([('label', (np.str_, 10)), ('element', elements_dtype)])
