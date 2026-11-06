"""Sized compact main-solver work; legacy structured work is unchanged."""

import operator
from collections import namedtuple

import numba as nb
import numpy as np
from numba.extending import overload

from andfn.storage import _layout, _validate_layout, growth_capacities

COEFFICIENT_FIELDS = ("coef0", "coef1", "old_coef", "coef_error")
INTEGRATION_FIELDS = (
    "thetas",
    "dpsi_corr",
    "exp_array_m",
    "exp_array_p",
    "z_integral",
    "phi",
    "psi",
    "dpsi",
)
CACHE_FIELDS = INTEGRATION_FIELDS[:5]
SCRATCH_GROUPS = (("coef0", "coef1"), ("integral",), ("phi", "psi", "dpsi"))
SCRATCH_FIELDS = sum(SCRATCH_GROUPS, ())
BRANCH_FIELDS = ("sign_array", "discharge_element", "element_pos")
SCALAR_FIELDS = (
    "len_discharge_element",
    "a0_error",
    "error",
    "set_zero",
    "branch_valid",
)
ARRAY_FIELDS = COEFFICIENT_FIELDS + ("integral",) + INTEGRATION_FIELDS + BRANCH_FIELDS
FIELDS = ARRAY_FIELDS + SCALAR_FIELDS
_Record = namedtuple("_Record", FIELDS)


class WorkRecord(_Record):
    __slots__ = ()

    def __getitem__(self, key):
        if isinstance(key, str):
            value = getattr(self, key)
            return value[0] if key in SCALAR_FIELDS else value
        return tuple.__getitem__(self, key)

    def __setitem__(self, key, value):
        if key not in SCALAR_FIELDS:
            raise TypeError("Only scalar work entries support assignment")
        getattr(self, key)[0] = value


BranchRecord = namedtuple("BranchRecord", BRANCH_FIELDS + ("len_discharge_element",))


WorkArrays = namedtuple(
    "WorkArrays",
    FIELDS
    + (
        "coef_start",
        "coef_capacity",
        "integration_start",
        "integration_capacity",
        "scratch_layout",
        "branch_start",
        "branch_capacity",
    ),
)


@overload(operator.getitem)
def _record_getitem(record, key):
    if (
        isinstance(record, (nb.types.NamedTuple, nb.types.NamedUniTuple))
        and record.instance_class in (WorkRecord, BranchRecord)
        and isinstance(key, nb.types.StringLiteral)
    ):
        name = key.literal_value
        fields = FIELDS if record.instance_class is WorkRecord else BranchRecord._fields
        index = fields.index(name)
        if name in SCALAR_FIELDS:

            def scalar(record, key):
                return record[index][0]

            return scalar

        def array(record, key):
            return record[index]

        return array


@overload(operator.setitem)
def _record_setitem(record, key, value):
    if (
        isinstance(record, (nb.types.NamedTuple, nb.types.NamedUniTuple))
        and record.instance_class in (WorkRecord, BranchRecord)
        and isinstance(key, nb.types.StringLiteral)
        and key.literal_value in SCALAR_FIELDS
    ):
        fields = FIELDS if record.instance_class is WorkRecord else BranchRecord._fields
        index = fields.index(key.literal_value)

        def scalar(record, key, value):
            record[index][0] = value

        return scalar


def work_record(work, i):
    if not isinstance(work, WorkArrays):
        return work[i]
    c, nc = work.coef_start[i], work.coef_capacity[i]
    s, ni = work.integration_start[i], work.integration_capacity[i]
    b, bn = work.branch_start[i], work.branch_capacity[i]
    mask, _slots, cs, ns = work.scratch_layout
    worker = nb.get_thread_id()
    c0, n0 = (worker * cs, cs) if mask & 1 else (c, nc)
    ci, ni0 = (worker * cs, cs) if mask & 2 else (c, nc)
    si, sn = (worker * ns, ns) if mask & 4 else (s, ni)
    return WorkRecord(
        work.coef0[c0 : c0 + n0],
        work.coef1[c0 : c0 + n0],
        work.old_coef[c : c + nc],
        work.coef_error[c : c + nc],
        work.integral[ci : ci + ni0],
        *(getattr(work, name)[s : s + ni] for name in CACHE_FIELDS),
        *(getattr(work, name)[si : si + sn] for name in ("phi", "psi", "dpsi")),
        *(getattr(work, name)[b : b + bn] for name in BRANCH_FIELDS),
        *(getattr(work, name)[i : i + 1] for name in SCALAR_FIELDS),
    )


@overload(work_record, inline="always")
def _work_record(work, i):
    if isinstance(work, nb.types.NamedTuple) and work.instance_class is WorkArrays:

        def compact(work, i):
            c, nc = work.coef_start[i], work.coef_capacity[i]
            s, ni = work.integration_start[i], work.integration_capacity[i]
            b, bn = work.branch_start[i], work.branch_capacity[i]
            mask, _slots, cs, ns = work.scratch_layout
            worker = nb.get_thread_id()
            c0, n0 = (worker * cs, cs) if mask & 1 else (c, nc)
            ci, ni0 = (worker * cs, cs) if mask & 2 else (c, nc)
            si, sn = (worker * ns, ns) if mask & 4 else (s, ni)
            return WorkRecord(
                work.coef0[c0 : c0 + n0],
                work.coef1[c0 : c0 + n0],
                work.old_coef[c : c + nc],
                work.coef_error[c : c + nc],
                work.integral[ci : ci + ni0],
                work.thetas[s : s + ni],
                work.dpsi_corr[s : s + ni],
                work.exp_array_m[s : s + ni],
                work.exp_array_p[s : s + ni],
                work.z_integral[s : s + ni],
                work.phi[si : si + sn],
                work.psi[si : si + sn],
                work.dpsi[si : si + sn],
                work.sign_array[b : b + bn],
                work.discharge_element[b : b + bn],
                work.element_pos[b : b + bn],
                work.len_discharge_element[i : i + 1],
                work.a0_error[i : i + 1],
                work.error[i : i + 1],
                work.set_zero[i : i + 1],
                work.branch_valid[i : i + 1],
            )

        return compact

    def legacy(work, i):
        return work[i]

    return legacy


def branch_topology_valid(work):
    if isinstance(work, WorkRecord):
        return work["branch_valid"] or work["len_discharge_element"] != 0
    return work["len_discharge_element"] != 0


def invalidate_branch(work):
    work["len_discharge_element"] = 0
    if isinstance(work, WorkRecord):
        work["branch_valid"] = False


@overload(invalidate_branch, inline="always")
def _invalidate_branch(work):
    if isinstance(work, nb.types.NamedTuple) and work.instance_class is WorkRecord:

        def compact(work):
            work["len_discharge_element"] = 0
            work["branch_valid"] = False

        return compact

    def legacy(work):
        work["len_discharge_element"] = 0

    return legacy


@overload(branch_topology_valid, inline="always")
def _branch_topology_valid(work):
    if isinstance(work, nb.types.NamedTuple) and work.instance_class is WorkRecord:

        def compact(work):
            return work["branch_valid"] or work["len_discharge_element"] != 0

        return compact

    def legacy(work):
        return work["len_discharge_element"] != 0

    return legacy


def integration_view(element, work, name):
    if isinstance(work, WorkRecord):
        return work[name]
    return element[name]


@overload(integration_view, inline="always")
def _integration_view(element, work, name):
    if isinstance(name, nb.types.StringLiteral):
        index = FIELDS.index(name.literal_value)
        if isinstance(work, nb.types.NamedTuple) and work.instance_class is WorkRecord:

            def compact(element, work, name):
                return work[index]

            return compact
        key = name.literal_value

        def legacy(element, work, name):
            return element[key]

        return legacy


def scratch_fields(work):
    return tuple(
        name
        for group, names in enumerate(SCRATCH_GROUPS)
        if work.scratch_layout[0] & (1 << group)
        for name in names
    )


def resize_scratch(work, coef_capacity, integration_capacity, *, groups=None):
    """Prepare worker storage outside parallel regions; never copy scratch."""
    mask, slots, cs, ns = work.scratch_layout
    old_mask = mask
    if groups is not None:
        if groups not in range(8):
            raise ValueError("Scratch group mask must be between zero and seven")
        mask, slots = groups, nb.config.NUMBA_NUM_THREADS
    new_cs = max(cs, int(coef_capacity.max(initial=0)))
    new_ns = max(ns, int(integration_capacity.max(initial=0)))
    if groups is None and (new_cs, new_ns) == (cs, ns):
        return {}
    values = {"scratch_layout": np.array([mask, slots, new_cs, new_ns], dtype=np.int64)}
    for group, names in enumerate(SCRATCH_GROUPS):
        if mask & (1 << group):
            for name in names:
                width = new_ns if group == 2 else new_cs
                values[name] = np.zeros(slots * width, dtype=getattr(work, name).dtype)
        elif old_mask & (1 << group):
            size = (
                int(integration_capacity.sum())
                if group == 2
                else int(coef_capacity.sum())
            )
            for name in names:
                values[name] = np.zeros(size, dtype=getattr(work, name).dtype)
    return values


def allocate_work(elements, coef_start, coef_capacity, *, scratch_groups=0):
    if scratch_groups not in range(8):
        raise ValueError("Scratch group mask must be between zero and seven")
    capacities = growth_capacities(elements["nint"])
    starts, size = _layout(capacities)
    ncoef = int(coef_capacity.sum())
    slots = nb.config.NUMBA_NUM_THREADS
    cs, ns = int(coef_capacity.max(initial=0)), int(capacities.max(initial=0))
    selected = {
        name
        for group, names in enumerate(SCRATCH_GROUPS)
        if scratch_groups & (1 << group)
        for name in names
    }
    values = {}
    for name in COEFFICIENT_FIELDS + ("integral",):
        values[name] = np.zeros(
            slots * cs if name in selected else ncoef, dtype=complex
        )
    for name in INTEGRATION_FIELDS:
        dtype = (
            complex if name in ("exp_array_m", "exp_array_p", "z_integral") else float
        )
        values[name] = np.zeros(slots * ns if name in selected else size, dtype=dtype)
    branch_capacity = np.where(np.isin(elements["_type"], (1, 4, 5)), 4, 0).astype(
        np.int64
    )
    branch_start, branch_size = _layout(branch_capacity)
    for name in BRANCH_FIELDS:
        values[name] = np.zeros(branch_size, dtype=np.int64)
    for name in SCALAR_FIELDS:
        dtype = (
            np.int64
            if name == "len_discharge_element"
            else (
                bool
                if name in ("set_zero", "branch_valid")
                else complex
                if name == "a0_error"
                else float
            )
        )
        values[name] = np.zeros(elements.size, dtype=dtype)
    work = WorkArrays(
        **values,
        coef_start=coef_start,
        coef_capacity=coef_capacity,
        integration_start=starts,
        integration_capacity=capacities,
        scratch_layout=np.array([scratch_groups, slots, cs, ns], dtype=np.int64),
        branch_start=branch_start,
        branch_capacity=branch_capacity,
    )
    return work


def payload(work):
    return sum(a.nbytes for a in {id(a): a for a in work}.values())


def state_arrays(state):
    return {id(a): a for a in (*vars(state.storage).values(), *state.work)}


def state_payload(state):
    return sum(a.nbytes for a in state_arrays(state).values())


def validate_work(work, ncoef, nint):
    if work.branch_valid.dtype != np.bool_ or work.branch_valid.shape != ncoef.shape:
        raise ValueError("Invalid branch validity metadata")
    if work.scratch_layout.dtype != np.int64 or work.scratch_layout.shape != (4,):
        raise ValueError("Invalid scratch ownership metadata")
    _validate_layout(
        work.coef_start,
        np.ascontiguousarray(ncoef),
        work.coef_capacity,
        work.old_coef.size,
        "work coef",
    )
    _validate_layout(
        work.integration_start,
        np.ascontiguousarray(nint),
        work.integration_capacity,
        work.thetas.size,
        "work integration",
    )
    for name in COEFFICIENT_FIELDS + ("integral",):
        if name in scratch_fields(work):
            continue
        a = getattr(work, name)
        if (
            a.dtype != np.complex128
            or a.shape != work.old_coef.shape
            or not a.flags.c_contiguous
        ):
            raise ValueError(f"Invalid compact work array: {name}")
    for name in INTEGRATION_FIELDS:
        if name in scratch_fields(work):
            continue
        a = getattr(work, name)
        dtype = (
            np.complex128
            if name in ("exp_array_m", "exp_array_p", "z_integral")
            else np.float64
        )
        if a.dtype != dtype or a.shape != work.thetas.shape or not a.flags.c_contiguous:
            raise ValueError(f"Invalid compact work array: {name}")
    mask, slots, cs, ns = work.scratch_layout
    if mask < 0 or mask > 7 or slots < nb.config.NUMBA_NUM_THREADS:
        raise ValueError("Invalid scratch ownership layout")
    if mask and (
        cs < work.coef_capacity.max(initial=0)
        or ns < work.integration_capacity.max(initial=0)
    ):
        raise ValueError("Insufficient worker scratch extent")
    for name in scratch_fields(work):
        a = getattr(work, name)
        width = ns if name in SCRATCH_GROUPS[2] else cs
        dtype = np.float64 if name in SCRATCH_GROUPS[2] else np.complex128
        if a.shape != (slots * width,) or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"Invalid worker scratch: {name}")
    for name in BRANCH_FIELDS:
        a = getattr(work, name)
        if (
            a.dtype != np.int64
            or a.shape != work.sign_array.shape
            or not a.flags.c_contiguous
        ):
            raise ValueError(f"Invalid compact branch array: {name}")
    _validate_layout(
        work.branch_start,
        work.len_discharge_element,
        work.branch_capacity,
        work.sign_array.size,
        "branch events",
    )


def import_work(work, legacy, elements):
    """Explicit test/compatibility initialization, never used in iterations."""
    for i, e in enumerate(elements):
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
                target[name][: e["nint"]] = e[name][: e["nint"]]
            elif name in SCALAR_FIELDS:
                target[name] = legacy[name][i]
            else:
                n = (
                    e["ncoef"]
                    if name in COEFFICIENT_FIELDS + ("integral",)
                    else (
                        legacy["len_discharge_element"][i]
                        if name in BRANCH_FIELDS
                        else e["nint"]
                    )
                )
                if n > target[name].size:
                    raise ValueError(
                        "Imported live branch events exceed prepared capacity"
                    )
                target[name][:n] = legacy[name][i, :n]


def export_work(work, dtype, *, include_scratch=True):
    """Explicit padded diagnostic export; not a solver execution bridge."""
    if include_scratch and scratch_fields(work):
        raise ValueError(
            "Worker scratch has no element history; use include_scratch=False or element-owned diagnostic mode"
        )
    result = np.zeros(work.set_zero.size, dtype=dtype)
    for i in range(result.size):
        record = work_record(work, i)
        for name in dtype.names:
            if name == "coef":
                continue
            if name in scratch_fields(work):
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
