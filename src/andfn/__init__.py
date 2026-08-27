"""
Copyright (C), 2024, Erik A. L. Toller.

AnDFN is a computer program that calculated the flow in a discrete fracture network (DFN) using the Analytic Element
Method (AEM).
"""

# version number
__author__ = "Erik A.L. Toller"
__version__ = "0.1.17"

from importlib import import_module

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
    "set_log_level",
]

_LAZY_EXPORTS = {
    "DFN": ("dfn", "DFN"),
    "BoundingCircle": ("bounding", "BoundingCircle"),
    "ConstantHeadLine": ("const_head", "ConstantHeadLine"),
    "ConstantHeadPrism": ("structures", "ConstantHeadPrism"),
    "Fracture": ("fracture", "Fracture"),
    "ImpermeableCircle": ("impermeable_object", "ImpermeableCircle"),
    "ImpermeableLine": ("impermeable_object", "ImpermeableLine"),
    "ImpermeablePrism": ("structures", "ImpermeablePrism"),
    "Intersection": ("intersection", "Intersection"),
    "RectangularRegion": ("regions", "RectangularRegion"),
    "Well": ("well", "Well"),
    "copy_dfn": ("geometry_functions", "copy_dfn"),
    "export_fractures": ("io", "export_fractures"),
    "fracture_intersection": ("geometry_functions", "fracture_intersection"),
    "map_2d_to_3d": ("geometry_functions", "map_2d_to_3d"),
    "map_3d_to_2d": ("geometry_functions", "map_3d_to_2d"),
    "set_log_level": ("constants", "set_log_level"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
