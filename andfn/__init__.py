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
from andfn.bounding import BoundingCircle
from andfn.const_head import ConstantHeadLine
from andfn.fracture import Fracture
from andfn.intersection import Intersection
from andfn.well import Well
from andfn.impermeable_object import ImpermeableCircle, ImpermeableLine
from andfn.structures import ConstantHeadPrism, ImpermeablePrism
from andfn.dfn import DFN
from andfn.geometry_functions import map_2d_to_3d, map_3d_to_2d, fracture_intersection

__all__ = [
    "BoundingCircle",
    "ConstantHeadLine",
    "DFN",
    "Fracture",
    "Intersection",
    "Well",
    "ImpermeableCircle",
    "ImpermeableLine",
    "ConstantHeadPrism",
    "ImpermeablePrism",
    "map_2d_to_3d",
    "map_3d_to_2d",
    "fracture_intersection",
]
