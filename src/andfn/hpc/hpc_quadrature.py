"""Runtime-sized main-solver quadrature; padded legacy records remain supported."""

import operator

import numba as nb
import numpy as np
from numba.extending import overload


def _dimensions(rows, count):
    for name, value in (("rows", rows), ("count", count)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Quadrature {name} must be an integer")
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"Quadrature {name} must be an integer") from exc
        if (name == "rows" and integer < 0) or (name == "count" and integer <= 0):
            raise ValueError(f"Invalid quadrature {name}")
    rows, count = operator.index(rows), operator.index(count)
    limit = np.iinfo(np.intp).max
    if count > limit - 2 or rows > limit or count > limit // 32:
        raise ValueError("Quadrature dimensions exceed addressable array size")
    if rows > limit // (count * 32):
        raise ValueError("Quadrature payload exceeds addressable array size")
    return rows, count


def allocate_quadrature(rows, count):
    """Allocate two contiguous side/row/point planes, independent of MAX_NCOEF."""
    rows, count = _dimensions(rows, count)
    return np.zeros((2, rows, count), dtype=np.complex128)


def quadrature_row(values, side, row, count):
    if values.ndim == 3:
        return values[side, row, :count]
    return values["z0" if side == 0 else "z1"][row][:count]


@overload(quadrature_row)
def _quadrature_row(values, side, row, count):
    if isinstance(values, nb.types.Array) and values.ndim == 3:

        def dense(values, side, row, count):
            return values[side, row, :count]

        return dense

    def legacy(values, side, row, count):
        if side == 0:
            return values["z0"][row][:count]
        return values["z1"][row][:count]

    return legacy


def validate_quadrature(values, rows, count):
    if values.ndim == 3:
        rows, count = _dimensions(rows, count)
        if (
            values.dtype != np.complex128
            or not values.flags.c_contiguous
            or values.shape != (2, rows, count)
        ):
            raise ValueError("Invalid dense discharge quadrature shape or dtype")
    elif count <= 0 or count > values.dtype["z0"].shape[0]:
        raise ValueError("Invalid discharge integration count")
    elif values.size < rows:
        raise ValueError("Insufficient discharge quadrature rows")
