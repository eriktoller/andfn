"""Compact numeric storage helpers shared by the solver kernels."""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import NDArray

element_metadata_dtype = np.dtype(
    [
        ("_id", np.int_),
        ("_type", np.int_),
        ("frac0", np.int_),
        ("frac1", np.int_),
        ("endpoints0", np.complex128, 2),
        ("endpoints1", np.complex128, 2),
        ("radius", np.float64),
        ("center", np.complex128),
        ("head", np.float64),
        ("phi", np.float64),
        ("ncoef", np.int_),
        ("nint", np.int_),
        ("q", np.float64),
        ("error", np.float64),
        ("error_old", np.float64),
        ("error_old2", np.float64),
        ("error_coef", np.float64),
    ]
)
fracture_metadata_dtype = np.dtype(
    [
        ("_id", np.int_),
        ("t", np.float64),
        ("radius", np.float64),
        ("center", np.float64, 3),
        ("normal", np.float64, 3),
        ("x_vector", np.float64, 3),
        ("y_vector", np.float64, 3),
        ("constant", np.float64),
        ("error_constant", np.float64),
    ]
)


@dataclass
class FixedOrderStorage:
    """Flat coefficient buffers and compact fixed element metadata."""

    elements: NDArray
    coef_data: NDArray
    candidate_coef_data: NDArray
    old_coef_data: NDArray
    coef_start: NDArray
    coef_capacity: NDArray
    element_dtype = element_metadata_dtype

    def validate(self):
        _array(self.elements, self.element_dtype, "elements")
        for name in ("coef_data", "candidate_coef_data", "old_coef_data"):
            _array(getattr(self, name), np.complex128, name)
        _validate_layout(
            self.coef_start,
            self.elements["ncoef"].copy(),
            self.coef_capacity,
            self.coef_data.size,
            "coef",
        )
        if not np.array_equal(self.elements["_id"], np.arange(self.elements.size)):
            raise ValueError("Element IDs must equal array positions")
        buffers = (self.coef_data, self.candidate_coef_data, self.old_coef_data)
        if any(buffer.size != self.coef_data.size for buffer in buffers):
            raise ValueError("Coefficient buffers must have identical sizes")
        if any(
            np.shares_memory(a, b)
            for i, a in enumerate(buffers)
            for b in buffers[i + 1 :]
        ):
            raise ValueError("Coefficient buffers must not alias")
        nint = self.elements["nint"]
        if np.any(nint < 0):
            raise ValueError("Integration lengths must be nonnegative")
        if np.any((self.elements["_type"] != 2) & (nint < 2)):
            raise ValueError(
                "Non-well elements require at least two integration points"
            )

    @property
    def nbytes(self):
        return sum(array.nbytes for array in vars(self).values())


def _array(array, dtype, name, ndim=1):
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype(dtype)
        or array.ndim != ndim
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be a contiguous {ndim}D {np.dtype(dtype)} array")


def _counts(values, size, name):
    values = np.asarray(values)
    if values.shape != (size,) or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain one integer per element")
    if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit nonnegative int64")
    return values.astype(np.int64, copy=True)


def _layout(capacities):
    total = sum(int(value) for value in capacities)
    if total > np.iinfo(np.int64).max:
        raise ValueError("Total capacity exceeds int64 offsets")
    starts = np.zeros(capacities.size, dtype=np.int64)
    if capacities.size > 1:
        np.cumsum(capacities[:-1], out=starts[1:])
    return starts, total


def _validate_layout(starts, lengths, capacities, size, name):
    for suffix, array in (
        ("start", starts),
        ("length", lengths),
        ("capacity", capacities),
    ):
        _array(array, np.int64, f"{name}_{suffix}")
    if starts.shape != lengths.shape or starts.shape != capacities.shape:
        raise ValueError(f"{name} layout arrays must have matching shapes")
    if np.any(lengths < 0) or np.any(capacities < lengths):
        raise ValueError(f"{name} requires 0 <= length <= capacity")
    expected, total = _layout(capacities)
    if not np.array_equal(starts, expected) or total != size:
        raise ValueError(f"{name} segments must partition the complete data array")


def growth_capacities(required, current=None, *, quantum=4, growth_factor=1.25):
    """Retain capacity where possible and grow only beyond logical demand."""
    quantum = index(quantum)
    if quantum <= 0 or not np.isfinite(growth_factor) or growth_factor < 1:
        raise ValueError("quantum must be positive and growth_factor must be >= 1")
    required = np.asarray(required)
    required = _counts(required, required.size, "required")
    current = (
        np.zeros_like(required)
        if current is None
        else _counts(current, required.size, "current")
    )
    capacities = current.copy()
    limit = np.iinfo(np.int64).max
    for i, need in enumerate(required):
        if need <= current[i]:
            continue
        grown = np.ceil(int(current[i]) * growth_factor)
        if not np.isfinite(grown) or grown > limit:
            raise ValueError("Capacity exceeds int64")
        target = max(int(need), int(grown))
        target = ((target + quantum - 1) // quantum) * quantum
        if target > limit:
            raise ValueError("Capacity exceeds int64")
        capacities[i] = target
    _layout(capacities)
    return capacities


@dataclass
class Connectivity:
    """Ordered CSR fracture-to-element connectivity."""

    ids: NDArray
    indptr: NDArray

    @property
    def nbytes(self):
        return self.ids.nbytes + self.indptr.nbytes

    def element_ids(self, fracture):
        fracture = index(fracture)
        if not 0 <= fracture < self.indptr.size - 1:
            raise IndexError("Invalid fracture position")
        start, end = self.indptr[fracture : fracture + 2]
        return self.ids[start:end]

    def validate(self, nfractures, nelements):
        if not isinstance(self.ids, np.ndarray) or self.ids.dtype not in (
            np.dtype(np.int32),
            np.dtype(np.int64),
        ):
            raise ValueError("Connectivity IDs must use int32 or int64")
        _array(self.ids, self.ids.dtype, "connectivity_ids")
        _array(self.indptr, np.int64, "connectivity_indptr")
        if (
            self.indptr.shape != (nfractures + 1,)
            or self.indptr[0] != 0
            or self.indptr[-1] != self.ids.size
            or np.any(self.indptr[1:] < self.indptr[:-1])
        ):
            raise ValueError("Invalid CSR offsets")
        if np.any(self.ids < 0) or np.any(self.ids >= nelements):
            raise ValueError("Connectivity references an invalid element")
