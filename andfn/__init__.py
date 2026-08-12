"""
Copyright (C), 2024, Erik A. L. Toller.

AnDFN is a computer program that calculated the flow in a discrete fracture network (DFN) using the Analytic Element
Method (AEM).
"""

# version number
__name__ = "andfn"
__author__ = "Erik A.L. Toller"
__version__ = "0.1.17"

# Import all classes and functions
from andfn.bounding import BoundingCircle
from andfn.const_head import ConstantHeadLine
from andfn.dfn import DFN
from andfn.fracture import Fracture
from andfn.geometry_functions import (
    copy_dfn,
    fracture_intersection,
    map_2d_to_3d,
    map_3d_to_2d,
)
from andfn.impermeable_object import ImpermeableCircle, ImpermeableLine
from andfn.intersection import Intersection
from andfn.io import export_fractures
from andfn.regions import RectangularRegion
from andfn.structures import ConstantHeadPrism, ImpermeablePrism
from andfn.well import Well

__all__ = [
    "DFN",
    "BoundingCircle",
    "ConstantHeadLine",
    "ConstantHeadPrism",
    "Fracture",
    "ImpermeableCircle",
    "ImpermeableLine",
    "ImpermeablePrism",
    "Intersection",
    "RectangularRegion",
    "Well",
    "copy_dfn",
    "export_fractures",
    "fracture_intersection",
    "map_2d_to_3d",
    "map_3d_to_2d",
]
