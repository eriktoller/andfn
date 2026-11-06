"""Opt-in numeric storage for the DFN memory-layout migration.

Containers and conversions run in Python. Their arrays, not the containers,
are intended to be passed to future Numba kernels.
"""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import NDArray

from .element import element_dtype_hpc, fracture_dtype_hpc

_ELEMENT_ARRAYS = {"coef", "old_coef", "thetas", "dpsi_corr"}
element_metadata_dtype = np.dtype(
    [
        (name, element_dtype_hpc.fields[name][0])
        for name in element_dtype_hpc.names
        if name not in _ELEMENT_ARRAYS
    ]
)
fracture_metadata_dtype = np.dtype(
    [
        (name, fracture_dtype_hpc.fields[name][0])
        for name in fracture_dtype_hpc.names
        if name not in {"elements", "nelements"}
    ]
)
fixed_order_element_dtype = np.dtype(
    [
        (name, element_dtype_hpc.fields[name][0])
        for name in element_dtype_hpc.names
        if name not in {"coef", "old_coef"}
    ]
)


@dataclass
class FixedOrderStorage:
    """Coefficient-only migration; integration fields remain legacy padded."""

    elements: NDArray
    coef_data: NDArray
    candidate_coef_data: NDArray
    old_coef_data: NDArray
    coef_start: NDArray
    coef_capacity: NDArray
    element_dtype = fixed_order_element_dtype

    def validate(self):
        _array(self.elements, self.element_dtype, "fixed-order elements")
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
            raise ValueError("IDs must equal array positions")
        buffers = (self.coef_data, self.candidate_coef_data, self.old_coef_data)
        if any(buffer.size != self.coef_data.size for buffer in buffers):
            raise ValueError("Coefficient buffers must have identical sizes")
        if any(
            np.shares_memory(a, b)
            for i, a in enumerate(buffers)
            for b in buffers[i + 1 :]
        ):
            raise ValueError("Coefficient buffers must not alias")
        if np.any(self.elements["ncoef"] > element_dtype_hpc["coef"].shape[0]):
            raise ValueError("Fixed-order coefficients exceed retained work capacity")
        nint = self.elements["nint"]
        if np.any(nint < 0) or np.any(nint > element_dtype_hpc["thetas"].shape[0]):
            raise ValueError("Invalid fixed integration lengths")
        if np.any((self.elements["_type"] != 2) & (nint < 2)):
            raise ValueError(
                "Non-well elements require at least two integration points"
            )

    @property
    def nbytes(self):
        return sum(array.nbytes for array in vars(self).values())


def fixed_order_from_hpc(elements, *, quantum=4):
    """Copy coefficient state without allocating a complete padded record."""
    _array(elements, element_dtype_hpc, "legacy elements")
    lengths = _counts(elements["ncoef"], elements.size, "ncoef")
    if np.any(lengths > elements["coef"].shape[1]):
        raise ValueError("ncoef exceeds legacy coefficient storage")
    capacities = growth_capacities(lengths, quantum=quantum)
    starts, size = _layout(capacities)
    metadata = np.empty(elements.size, dtype=fixed_order_element_dtype)
    for name in metadata.dtype.names:
        metadata[name] = elements[name]
    current = np.zeros(size, dtype=np.complex128)
    old = np.zeros_like(current)
    for i, length in enumerate(lengths):
        start = starts[i]
        current[start : start + length] = elements["coef"][i, :length]
        old[start : start + length] = elements["old_coef"][i, :length]
    storage = FixedOrderStorage(
        metadata, current, np.zeros_like(current), old, starts, capacities
    )
    storage.validate()
    return storage


def fixed_order_to_hpc(storage):
    """Explicit compatibility export; never called by the fixed-order solver."""
    storage.validate()
    elements = np.zeros(storage.elements.size, dtype=element_dtype_hpc)
    for name in storage.elements.dtype.names:
        elements[name] = storage.elements[name]
    for i, n in enumerate(storage.elements["ncoef"]):
        start = storage.coef_start[i]
        elements["coef"][i, :n] = storage.coef_data[start : start + n]
        elements["old_coef"][i, :n] = storage.old_coef_data[start : start + n]
    return elements


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
    """Choose capacities without coupling allocation limits to solver limits.

    Existing capacity is retained unless exceeded. This allocation policy does
    not decide whether a numerical order is allowed to grow.
    """
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
    """Ordered, fixed CSR fracture-to-element connectivity."""

    ids: NDArray
    indptr: NDArray

    @property
    def nbytes(self):
        """Owned NumPy payload, not process resident or peak memory."""
        return self.ids.nbytes + self.indptr.nbytes

    def element_ids(self, fracture):
        """Python diagnostic view; kernels receive IDs and offsets directly."""
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


@dataclass
class DFNStorage:
    """Owned flat arrays; coefficient and integration layouts are independent.

    ``old_coef_data`` is a Phase 1 compatibility field, not a requirement of
    the future solver. Candidate coefficients start at zero. Scratch arrays,
    branch caches and exponential caches are not represented in this phase.
    """

    elements: NDArray
    fractures: NDArray
    connectivity: Connectivity
    coef_start: NDArray
    coef_capacity: NDArray
    coef_data: NDArray
    old_coef_data: NDArray
    candidate_coef_data: NDArray
    integration_start: NDArray
    integration_capacity: NDArray
    theta_data: NDArray
    dpsi_corr_data: NDArray

    @property
    def nbytes(self):
        """Owned numeric payload only; excludes Python/allocator overhead."""
        arrays = (
            self.elements,
            self.fractures,
            self.connectivity.ids,
            self.connectivity.indptr,
            self.coef_start,
            self.coef_capacity,
            self.coef_data,
            self.old_coef_data,
            self.candidate_coef_data,
            self.integration_start,
            self.integration_capacity,
            self.theta_data,
            self.dpsi_corr_data,
        )
        return sum(array.nbytes for array in arrays)

    def validate(self):
        _array(self.elements, element_metadata_dtype, "elements")
        _array(self.fractures, fracture_metadata_dtype, "fractures")
        ne = self.elements.size
        for name in ("coef_data", "old_coef_data", "candidate_coef_data"):
            _array(getattr(self, name), np.complex128, name)
            if getattr(self, name).size != self.coef_data.size:
                raise ValueError("Coefficient buffers must share one layout")
        for name in ("theta_data", "dpsi_corr_data"):
            _array(getattr(self, name), np.float64, name)
            if getattr(self, name).size != self.theta_data.size:
                raise ValueError("Integration buffers must share one layout")
        buffers = (
            self.coef_data,
            self.old_coef_data,
            self.candidate_coef_data,
            self.theta_data,
            self.dpsi_corr_data,
        )
        for i, array in enumerate(buffers):
            if any(np.shares_memory(array, other) for other in buffers[i + 1 :]):
                raise ValueError("Data buffers must not alias")
        # Structured scalar fields are strided; layout validation uses copies.
        _validate_layout(
            self.coef_start,
            self.elements["ncoef"].astype(np.int64),
            self.coef_capacity,
            self.coef_data.size,
            "coef",
        )
        _validate_layout(
            self.integration_start,
            self.elements["nint"].astype(np.int64),
            self.integration_capacity,
            self.theta_data.size,
            "integration",
        )
        if self.coef_start.size != ne or self.integration_start.size != ne:
            raise ValueError("Layouts must contain one segment per element")
        if not np.array_equal(self.elements["_id"], np.arange(ne)):
            raise ValueError("Element IDs must equal array positions")
        if not np.array_equal(self.fractures["_id"], np.arange(self.fractures.size)):
            raise ValueError("Fracture IDs must equal array positions")
        for name in ("frac0", "frac1"):
            references = self.elements[name]
            if np.any(references < -1) or np.any(references >= self.fractures.size):
                raise ValueError(f"Invalid {name} fracture reference")
        self.connectivity.validate(self.fractures.size, ne)

    def coefficients(self, element):
        """Return a writable view of logical coefficients, excluding slack."""
        return self._element_view(element, self.coef_data)

    def old_coefficients(self, element):
        """Return the logical compatibility old-coefficient view."""
        return self._element_view(element, self.old_coef_data)

    def candidate_coefficients(self, element):
        """Return the logical candidate-coefficient view."""
        return self._element_view(element, self.candidate_coef_data)

    def theta_values(self, element):
        """Return the logical integration theta view."""
        return self._element_view(element, self.theta_data, integration=True)

    def dpsi_corrections(self, element):
        """Return all nint logical correction entries retained by conversion."""
        return self._element_view(element, self.dpsi_corr_data, integration=True)

    def fracture_element_ids(self, fracture):
        """Return ordered logical connectivity for Python-level diagnostics."""
        return self.connectivity.element_ids(fracture)

    def _element_view(self, element, data, *, integration=False):
        element = index(element)
        if not 0 <= element < self.elements.size:
            raise IndexError("Invalid element position")
        starts = self.integration_start if integration else self.coef_start
        length = self.elements["nint" if integration else "ncoef"][element]
        return data[starts[element] : starts[element] + length]

    def expand(self, ncoef, nint, *, quantum=4, growth_factor=1.25):
        """Batch-expand outside kernels, preserving prefixes and zeroing tails.

        Repacking invalidates previously returned views. Callers must reacquire
        views after expansion. All validation/allocation precedes state changes.
        Shrinking is deliberately not supported.
        """
        self.validate()
        nc = _counts(ncoef, self.elements.size, "ncoef")
        ni = _counts(nint, self.elements.size, "nint")
        if np.any(nc < self.elements["ncoef"]) or np.any(ni < self.elements["nint"]):
            raise ValueError("Expansion cannot shrink logical lengths")
        cc = growth_capacities(
            nc, self.coef_capacity, quantum=quantum, growth_factor=growth_factor
        )
        ic = growth_capacities(
            ni, self.integration_capacity, quantum=quantum, growth_factor=growth_factor
        )
        cs, csize = _layout(cc)
        ins, isize = _layout(ic)
        repack_coef = not np.array_equal(cc, self.coef_capacity)
        repack_integration = not np.array_equal(ic, self.integration_capacity)
        coefficient_arrays = (
            [np.zeros(csize, dtype=np.complex128) for _ in range(3)]
            if repack_coef
            else [self.coef_data, self.old_coef_data, self.candidate_coef_data]
        )
        integration_arrays = (
            [np.zeros(isize, dtype=np.float64) for _ in range(2)]
            if repack_integration
            else [self.theta_data, self.dpsi_corr_data]
        )
        for i in range(self.elements.size):
            old = self.coef_start[i]
            new = cs[i]
            length = self.elements["ncoef"][i]
            for source, target in zip(
                (self.coef_data, self.old_coef_data, self.candidate_coef_data),
                coefficient_arrays,
            ):
                if repack_coef:
                    target[new : new + length] = source[old : old + length]
                target[new + length : new + nc[i]] = 0
            old = self.integration_start[i]
            new = ins[i]
            length = self.elements["nint"][i]
            for source, target in zip(
                (self.theta_data, self.dpsi_corr_data), integration_arrays
            ):
                if repack_integration:
                    target[new : new + length] = source[old : old + length]
                target[new + length : new + ni[i]] = 0
        self.coef_start, self.coef_capacity = cs, cc
        self.integration_start, self.integration_capacity = ins, ic
        self.coef_data, self.old_coef_data, self.candidate_coef_data = (
            coefficient_arrays
        )
        self.theta_data, self.dpsi_corr_data = integration_arrays
        self.elements["ncoef"], self.elements["nint"] = nc, ni


def from_hpc(elements, fractures, *, quantum=4, index_dtype=np.int64):
    """Copy logical legacy HPC values into independently owned flat storage."""
    _array(elements, element_dtype_hpc, "legacy elements")
    _array(fractures, fracture_dtype_hpc, "legacy fractures")
    ne, nf = elements.size, fractures.size
    nc = _counts(elements["ncoef"], ne, "ncoef")
    ni = _counts(elements["nint"], ne, "nint")
    if np.any(nc > elements["coef"].shape[1]):
        raise ValueError("ncoef exceeds legacy coefficient storage")
    if np.any(ni > elements["thetas"].shape[1]):
        raise ValueError("nint exceeds legacy integration storage")
    connectivity = connectivity_from_hpc(fractures, ne, index_dtype=index_dtype)
    cc = growth_capacities(nc, quantum=quantum)
    ic = growth_capacities(ni, quantum=quantum)
    cs, csize = _layout(cc)
    ins, isize = _layout(ic)
    em = np.zeros(ne, dtype=element_metadata_dtype)
    fm = np.zeros(nf, dtype=fracture_metadata_dtype)
    for name in em.dtype.names:
        em[name] = elements[name]
    for name in fm.dtype.names:
        fm[name] = fractures[name]
    storage = DFNStorage(
        em,
        fm,
        connectivity,
        cs,
        cc,
        np.zeros(csize, dtype=np.complex128),
        np.zeros(csize, dtype=np.complex128),
        np.zeros(csize, dtype=np.complex128),
        ins,
        ic,
        np.zeros(isize, dtype=np.float64),
        np.zeros(isize, dtype=np.float64),
    )
    for i in range(ne):
        c, n = cs[i], nc[i]
        storage.coef_data[c : c + n] = elements["coef"][i, :n]
        storage.old_coef_data[c : c + n] = elements["old_coef"][i, :n]
        c, n = ins[i], ni[i]
        storage.theta_data[c : c + n] = elements["thetas"][i, :n]
        storage.dpsi_corr_data[c : c + n] = elements["dpsi_corr"][i, :n]
    storage.validate()
    return storage


def to_hpc(storage):
    """Produce zero-padded arrays accepted by unchanged numerical kernels.

    Only logical values are preserved, not unused legacy padding or candidates.
    """
    storage.validate()
    nc, ni = storage.elements["ncoef"], storage.elements["nint"]
    degree = np.diff(storage.connectivity.indptr)
    if np.any(nc > element_dtype_hpc["coef"].shape[0]):
        raise ValueError("ncoef exceeds legacy coefficient storage")
    if np.any(ni > element_dtype_hpc["thetas"].shape[0]):
        raise ValueError("nint exceeds legacy integration storage")
    if np.any(degree > fracture_dtype_hpc["elements"].shape[0]):
        raise ValueError("Connectivity exceeds legacy fracture storage")
    elements = np.zeros(storage.elements.size, dtype=element_dtype_hpc)
    fractures = np.zeros(storage.fractures.size, dtype=fracture_dtype_hpc)
    for name in storage.elements.dtype.names:
        elements[name] = storage.elements[name]
    for name in storage.fractures.dtype.names:
        fractures[name] = storage.fractures[name]
    for i in range(elements.size):
        c, n = storage.coef_start[i], nc[i]
        elements["coef"][i, :n] = storage.coef_data[c : c + n]
        elements["old_coef"][i, :n] = storage.old_coef_data[c : c + n]
        c, n = storage.integration_start[i], ni[i]
        elements["thetas"][i, :n] = storage.theta_data[c : c + n]
        elements["dpsi_corr"][i, :n] = storage.dpsi_corr_data[c : c + n]
    for i in range(fractures.size):
        start, end = storage.connectivity.indptr[i : i + 2]
        fractures["elements"][i, : degree[i]] = storage.connectivity.ids[start:end]
    fractures["nelements"] = degree
    return elements, fractures


def connectivity_from_hpc(fractures, nelements, *, index_dtype=np.int64):
    """Extract connectivity without copying coefficient or fracture records."""
    from .hpc.hpc_connectivity import FractureArrays

    index_dtype = np.dtype(index_dtype)
    if index_dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise ValueError("index_dtype must be int32 or int64")
    if isinstance(fractures, FractureArrays):
        connectivity = Connectivity(fractures.ids, fractures.indptr)
        connectivity.validate(fractures.size, nelements)
        if connectivity.ids.dtype != index_dtype:
            if np.any(connectivity.ids > np.iinfo(index_dtype).max):
                raise ValueError("Connectivity references an invalid element")
            return Connectivity(
                connectivity.ids.astype(index_dtype), connectivity.indptr
            )
        return connectivity
    _array(fractures, fracture_dtype_hpc, "legacy fractures")
    degree = _counts(fractures["nelements"], fractures.size, "nelements")
    if np.any(degree > fractures["elements"].shape[1]):
        raise ValueError("nelements exceeds legacy connectivity storage")
    starts, total = _layout(degree)
    indptr = np.empty(fractures.size + 1, dtype=np.int64)
    indptr[:-1], indptr[-1] = starts, total
    ids = np.empty(total, dtype=index_dtype)
    for i in range(fractures.size):
        row = fractures["elements"][i, : degree[i]]
        if (
            np.any(row < 0)
            or np.any(row >= nelements)
            or np.any(row > np.iinfo(index_dtype).max)
        ):
            raise ValueError("Connectivity references an invalid element")
        ids[indptr[i] : indptr[i + 1]] = row
    connectivity = Connectivity(ids, indptr)
    connectivity.validate(fractures.size, nelements)
    return connectivity


def fracture_metadata_from_hpc(fractures):
    """Copy fixed metadata only; no legacy connectivity is reconstructed."""
    _array(fractures, fracture_dtype_hpc, "legacy fractures")
    metadata = np.empty(fractures.size, dtype=fracture_metadata_dtype)
    for name in metadata.dtype.names:
        metadata[name] = fractures[name]
    return metadata
