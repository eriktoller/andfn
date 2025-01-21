"""Copyright (C), 2024, Erik A. L. Toller.

Erik A. L. Toller, WSP Sverige AB
erik dot toller at wsp dot com

AnDFN is a computer program that calculated the flow in a discrete fracture network (DFN) using the Analytic Element
Method (AEM).
"""

# version number
__name__ = "andfn"
__author__ = "Erik A.L. Toller"
__version__ = "0.1"

# Import all classes and functions
from andfn.bounding import BoundingCircle
from andfn.const_head import ConstantHeadLine
from andfn.dfn import DFN
from andfn.fracture import Fracture
from andfn.intersection import Intersection
from andfn.well import Well
import andfn.geometry_functions
import andfn.math_functions

__all__ = [
    'BoundingCircle',
    'ConstantHeadLine',
    'DFN',
    'Fracture',
    'Intersection',
    'Well'
]
