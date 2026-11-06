"""Direct model-object consolidation into the established compact solver state."""

import numpy as np

from andfn.storage import (
    Connectivity,
    _layout,
    element_metadata_dtype,
    fracture_metadata_dtype,
    growth_capacities,
)

from .hpc_adaptive import AdaptiveSolver, AdaptiveState, AdaptiveStorage, validate_state
from .hpc_connectivity import FractureArrays
from .hpc_work import allocate_work, work_record


def consolidate(elements, fractures):
    """Preserve object/adjacency ordering without any padded execution bridge."""
    metadata = np.zeros(len(elements), dtype=element_metadata_dtype)
    for i, element in enumerate(elements):
        if element._id != i:
            raise ValueError("Element IDs must equal array positions")
        for name in metadata.dtype.names:
            if name in ("frac0", "frac1"):
                fracture = getattr(element, name, None)
                metadata[name][i] = -1 if fracture is None else fracture._id
            elif name in ("endpoints0", "endpoints1"):
                metadata[name][i] = getattr(element, name, 0.0)
            else:
                metadata[name][i] = getattr(element, name, 0)
    if np.any((metadata["_type"] != 2) & (metadata["ncoef"] < 2)):
        raise ValueError("Adaptive non-well elements require at least two coefficients")
    capacity = growth_capacities(metadata["ncoef"])
    starts, size = _layout(capacity)
    current = np.zeros(size, dtype=complex)
    for i, element in enumerate(elements):
        n = metadata["ncoef"][i]
        if n:
            if len(element.coef) < n:
                raise ValueError("Insufficient logical element coefficients")
            current[starts[i] : starts[i] + n] = element.coef[:n]
    storage = AdaptiveStorage(
        metadata,
        current,
        np.zeros_like(current),
        np.zeros_like(current),
        starts,
        capacity,
    )
    work = allocate_work(metadata, starts, capacity, scratch_groups=7)
    for i, element in enumerate(elements):
        n = metadata["nint"][i]
        for name in ("thetas", "dpsi_corr"):
            source = getattr(element, name, ())
            length = min(len(source), n)
            work_record(work, i)[name][:length] = source[:length]
    owner = AdaptiveSolver(AdaptiveState(storage, work))
    validate_state(owner.state)
    fixed = np.empty(len(fractures), dtype=fracture_metadata_dtype)
    degree = np.array(
        [len(fracture.elements) for fracture in fractures], dtype=np.int64
    )
    offsets, total = _layout(degree)
    indptr = np.empty(len(fractures) + 1, dtype=np.int64)
    indptr[:-1], indptr[-1] = offsets, total
    ids = np.empty(total, dtype=np.int64)
    for i, fracture in enumerate(fractures):
        if fracture._id != i:
            raise ValueError("Fracture IDs must equal array positions")
        for name in fixed.dtype.names:
            fixed[name][i] = getattr(fracture, name, 0)
        ids[indptr[i] : indptr[i + 1]] = [element._id for element in fracture.elements]
    Connectivity(ids, indptr).validate(len(fractures), len(elements))
    return owner, FractureArrays(fixed, ids, indptr)


def update_objects(owner, fractures, element_objects, fracture_objects):
    """Copy only logical results when explicit object unconsolidation is requested."""
    storage = owner.state.storage
    for i, obj in enumerate(element_objects):
        row = storage.elements[i]
        for name in ("ncoef", "nint", "q", "phi", "error"):
            if hasattr(obj, name):
                setattr(obj, name, row[name].item())
        n, start = row["ncoef"], storage.coef_start[i]
        if hasattr(obj, "coef"):
            obj.coef = storage.coef_data[start : start + n].copy()
        record = work_record(owner.state.work, i)
        for name in ("thetas", "dpsi_corr"):
            if hasattr(obj, name):
                setattr(obj, name, record[name][: row["nint"]].copy())
    for i, obj in enumerate(fracture_objects):
        obj.constant = float(fractures["constant"][i])
