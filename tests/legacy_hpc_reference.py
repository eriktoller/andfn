"""Test-only padded records for low-level numerical regression fixtures."""

import numpy as np

MAX_NCOEF = 200
MAX_ELEMENTS = 500
MAX_DISCHARGE_INT = 1000

element_dtype_hpc = np.dtype(
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
        ("thetas", np.float64, MAX_NCOEF * 2),
        ("coef", np.complex128, MAX_NCOEF),
        ("old_coef", np.complex128, MAX_NCOEF),
        ("dpsi_corr", np.float64, MAX_NCOEF * 2),
        ("error", np.float64),
        ("error_old", np.float64),
        ("error_old2", np.float64),
        ("error_coef", np.float64),
    ]
)
fracture_dtype_hpc = np.dtype(
    [
        ("_id", np.int_),
        ("t", np.float64),
        ("radius", np.float64),
        ("center", np.float64, 3),
        ("normal", np.float64, 3),
        ("x_vector", np.float64, 3),
        ("y_vector", np.float64, 3),
        ("elements", np.int_, MAX_ELEMENTS),
        ("nelements", np.int_),
        ("constant", np.float64),
        ("error_constant", np.float64),
    ]
)
dtype_work = np.dtype(
    [
        ("phi", np.float64, MAX_NCOEF * 2),
        ("psi", np.float64, MAX_NCOEF * 2),
        ("coef", np.complex128, MAX_NCOEF),
        ("coef0", np.complex128, MAX_NCOEF),
        ("coef1", np.complex128, MAX_NCOEF),
        ("coef_error", np.complex128, MAX_NCOEF),
        ("a0_error", np.complex128),
        ("old_coef", np.complex128, MAX_NCOEF),
        ("dpsi", np.float64, MAX_NCOEF * 2),
        ("error", np.float64),
        ("integral", np.complex128, MAX_NCOEF * 2),
        ("sign_array", np.int64, MAX_ELEMENTS),
        ("discharge_element", np.int64, MAX_ELEMENTS),
        ("element_pos", np.int64, MAX_ELEMENTS),
        ("len_discharge_element", np.int64),
        ("exp_array_m", np.complex128, MAX_NCOEF * 2),
        ("exp_array_p", np.complex128, MAX_NCOEF * 2),
        ("z_integral", np.complex128, MAX_NCOEF * 2),
        ("set_zero", np.bool_),
    ]
)
dtype_z_arrays = np.dtype(
    [("z0", np.complex128, MAX_NCOEF * 2), ("z1", np.complex128, MAX_NCOEF * 2)]
)


def fixed_order_from_hpc(elements, *, quantum=4):
    from andfn.hpc.hpc_adaptive import AdaptiveStorage
    from andfn.storage import (
        _layout,
        element_metadata_dtype,
        growth_capacities,
    )

    ncoef = elements["ncoef"].astype(np.int64)
    capacities = growth_capacities(ncoef, quantum=quantum)
    starts, size = _layout(capacities)
    metadata = np.zeros(elements.size, dtype=element_metadata_dtype)
    for name in metadata.dtype.names:
        metadata[name] = elements[name]
    current = np.zeros(size, dtype=np.complex128)
    old = np.zeros_like(current)
    for i, count in enumerate(ncoef):
        start = starts[i]
        current[start : start + count] = elements["coef"][i, :count]
        old[start : start + count] = elements["old_coef"][i, :count]
    storage = AdaptiveStorage(
        metadata, current, np.zeros_like(current), old, starts, capacities
    )
    storage.validate()
    return storage


def adaptive_from_hpc(elements, *, scratch_groups=7):
    from andfn.hpc.hpc_adaptive import AdaptiveSolver, AdaptiveState
    from andfn.hpc.hpc_work import allocate_work, work_record

    storage = fixed_order_from_hpc(elements)
    work = allocate_work(
        storage.elements,
        storage.coef_start,
        storage.coef_capacity,
        scratch_groups=scratch_groups,
    )
    for i, element in enumerate(elements):
        record = work_record(work, i)
        for name in ("thetas", "dpsi_corr"):
            record[name][: element["nint"]] = element[name][: element["nint"]]
    owner = AdaptiveSolver(AdaptiveState(storage, work))
    owner.state.storage.validate()
    return owner


def to_hpc(state):
    storage = getattr(state, "storage", state)
    elements = np.zeros(storage.elements.size, dtype=element_dtype_hpc)
    for name in storage.elements.dtype.names:
        elements[name] = storage.elements[name]
    for i, count in enumerate(storage.elements["ncoef"]):
        start = storage.coef_start[i]
        elements["coef"][i, :count] = storage.coef_data[start : start + count]
        elements["old_coef"][i, :count] = storage.old_coef_data[start : start + count]
    if hasattr(state, "work"):
        from andfn.hpc.hpc_work import work_record

        for i, element in enumerate(elements):
            for name in ("thetas", "dpsi_corr"):
                element[name][: element["nint"]] = work_record(state.work, i)[name][
                    : element["nint"]
                ]
    return elements


def connectivity_from_hpc(fractures, nelements, *, index_dtype=np.int64):
    from andfn.storage import Connectivity, _layout

    if index_dtype not in (np.int32, np.int64):
        raise ValueError("index_dtype must be int32 or int64")
    if hasattr(fractures, "ids") and hasattr(fractures, "indptr"):
        limits = np.iinfo(index_dtype)
        if fractures.ids.size and (
            fractures.ids.min() < limits.min or fractures.ids.max() > limits.max
        ):
            raise ValueError("Connectivity IDs do not fit index_dtype")
        ids = (
            fractures.ids
            if fractures.ids.dtype == np.dtype(index_dtype)
            else fractures.ids.astype(index_dtype)
        )
        result = Connectivity(ids, fractures.indptr)
        result.validate(fractures.size, nelements)
        return result
    degree = fractures["nelements"].astype(np.int64)
    starts, total = _layout(degree)
    indptr = np.empty(fractures.size + 1, dtype=np.int64)
    indptr[:-1], indptr[-1] = starts, total
    ids = np.empty(total, dtype=index_dtype)
    for i, count in enumerate(degree):
        values = fractures["elements"][i, :count]
        limits = np.iinfo(index_dtype)
        if values.size and (values.min() < limits.min or values.max() > limits.max):
            raise ValueError("Connectivity IDs do not fit index_dtype")
        ids[indptr[i] : indptr[i + 1]] = values
    result = Connectivity(ids, indptr)
    result.validate(fractures.size, nelements)
    return result


def fracture_metadata_from_hpc(fractures):
    from andfn.storage import fracture_metadata_dtype

    metadata = np.empty(fractures.size, dtype=fracture_metadata_dtype)
    for name in metadata.dtype.names:
        metadata[name] = fractures[name]
    return metadata


def fixed_order_to_hpc(storage):
    return to_hpc(storage)


def import_work(work, legacy, elements):
    from andfn.hpc.hpc_work import (
        BRANCH_FIELDS,
        COEFFICIENT_FIELDS,
        FIELDS,
        SCALAR_FIELDS,
        scratch_fields,
        work_record,
    )

    for i, element in enumerate(elements):
        target = work_record(work, i)
        for name in FIELDS:
            if name == "branch_valid":
                target[name] = legacy["len_discharge_element"][i] != 0
                continue
            if name in scratch_fields(work):
                if np.any(legacy[name][i] != 0):
                    raise ValueError(
                        "Nonzero scratch import requires element-owned diagnostic mode"
                    )
                continue
            if name in ("thetas", "dpsi_corr"):
                target[name][: element["nint"]] = element[name][: element["nint"]]
            elif name in SCALAR_FIELDS:
                target[name] = legacy[name][i]
            else:
                length = (
                    element["ncoef"]
                    if name in COEFFICIENT_FIELDS + ("integral",)
                    else (
                        legacy["len_discharge_element"][i]
                        if name in BRANCH_FIELDS
                        else element["nint"]
                    )
                )
                if length > target[name].size:
                    raise ValueError(
                        "Imported live branch events exceed prepared capacity"
                    )
                target[name][:length] = legacy[name][i, :length]


def export_work(work, dtype, *, include_scratch=True):
    from andfn.hpc.hpc_work import (
        BRANCH_FIELDS,
        SCALAR_FIELDS,
        scratch_fields,
        work_record,
    )

    if include_scratch and scratch_fields(work):
        raise ValueError(
            "Worker scratch has no element history; use include_scratch=False or element-owned diagnostic mode"
        )
    result = np.zeros(work.set_zero.size, dtype=dtype)
    for i in range(result.size):
        record = work_record(work, i)
        for name in dtype.names:
            if name == "coef" or name in scratch_fields(work):
                continue
            if name in SCALAR_FIELDS:
                result[name][i] = record[name]
            else:
                value = record[name]
                if name in BRANCH_FIELDS:
                    value = value[: record["len_discharge_element"]]
                if value.size > result[name].shape[1]:
                    raise ValueError(
                        "Compact events exceed legacy diagnostic export capacity"
                    )
                result[name][i, : value.size] = value
    return result
