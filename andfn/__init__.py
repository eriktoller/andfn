"""
Copyright (C), 2024, Erik A. L. Toller.

AnDFN is a computer program that calculated the flow in a discrete fracture network (DFN) using the Analytic Element
Method (AEM).
"""

# version number
__name__ = "andfn"
__author__ = "Erik A.L. Toller"
__version__ = "0.1.8"

# Import all classes and functions
from andfn.element import Element
from andfn.bounding import BoundingCircle
from andfn.const_head import ConstantHeadLine
from andfn.fracture import Fracture
from andfn.intersection import Intersection
from andfn.well import Well
from andfn.impermeable_object import ImpermeableCircle
from andfn.dfn import DFN

__all__ = [
    "BoundingCircle",
    "ConstantHeadLine",
    "DFN",
    "Fracture",
    "Intersection",
    "Well",
    "ImpermeableCircle",
]
