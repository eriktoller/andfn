"""Sized compact main-solver work; legacy structured work is unchanged."""

import operator
from collections import namedtuple

import numba as nb
import numpy as np
from numba.extending import overload

from andfn.element import MAX_ELEMENTS
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
BRANCH_FIELDS = ("sign_array", "discharge_element", "element_pos")
SCALAR_FIELDS = ("len_discharge_element", "a0_error", "error", "set_zero")
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


WorkArrays = namedtuple(
    "WorkArrays",
    FIELDS
    + (
        "coef_start",
        "coef_capacity",
        "integration_start",
        "integration_capacity",
    ),
)


@overload(operator.getitem)
def _record_getitem(record, key):
    if (
        isinstance(record, nb.types.NamedTuple)
        and record.instance_class is WorkRecord
        and isinstance(key, nb.types.StringLiteral)
    ):
        name = key.literal_value
        index = FIELDS.index(name)
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
        isinstance(record, nb.types.NamedTuple)
        and record.instance_class is WorkRecord
        and isinstance(key, nb.types.StringLiteral)
        and key.literal_value in SCALAR_FIELDS
    ):
        index = FIELDS.index(key.literal_value)

        def scalar(record, key, value):
            record[index][0] = value

        return scalar


def work_record(work, i):
    if not isinstance(work, WorkArrays):
        return work[i]
    c, nc = work.coef_start[i], work.coef_capacity[i]
    s, ni = work.integration_start[i], work.integration_capacity[i]
    b = i * MAX_ELEMENTS
    return WorkRecord(
        *(getattr(work, name)[c : c + nc] for name in COEFFICIENT_FIELDS),
        work.integral[c : c + nc],
        *(getattr(work, name)[s : s + ni] for name in INTEGRATION_FIELDS),
        *(getattr(work, name)[b : b + MAX_ELEMENTS] for name in BRANCH_FIELDS),
        *(getattr(work, name)[i : i + 1] for name in SCALAR_FIELDS),
    )


@overload(work_record, inline="always")
def _work_record(work, i):
    if isinstance(work, nb.types.NamedTuple) and work.instance_class is WorkArrays:

        def compact(work, i):
            c, nc = work.coef_start[i], work.coef_capacity[i]
            s, ni = work.integration_start[i], work.integration_capacity[i]
            b = i * MAX_ELEMENTS
            return WorkRecord(
                work.coef0[c : c + nc],
                work.coef1[c : c + nc],
                work.old_coef[c : c + nc],
                work.coef_error[c : c + nc],
                work.integral[c : c + nc],
                work.thetas[s : s + ni],
                work.dpsi_corr[s : s + ni],
                work.exp_array_m[s : s + ni],
                work.exp_array_p[s : s + ni],
                work.z_integral[s : s + ni],
                work.phi[s : s + ni],
                work.psi[s : s + ni],
                work.dpsi[s : s + ni],
                work.sign_array[b : b + MAX_ELEMENTS],
                work.discharge_element[b : b + MAX_ELEMENTS],
                work.element_pos[b : b + MAX_ELEMENTS],
                work.len_discharge_element[i : i + 1],
                work.a0_error[i : i + 1],
                work.error[i : i + 1],
                work.set_zero[i : i + 1],
            )

        return compact

    def legacy(work, i):
        return work[i]

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


def allocate_work(elements, coef_start, coef_capacity):
    capacities = growth_capacities(elements["nint"])
    starts, size = _layout(capacities)
    ncoef = int(coef_capacity.sum())
    values = {}
    for name in COEFFICIENT_FIELDS + ("integral",):
        values[name] = np.zeros(ncoef, dtype=complex)
    for name in INTEGRATION_FIELDS:
        dtype = (
            complex if name in ("exp_array_m", "exp_array_p", "z_integral") else float
        )
        values[name] = np.zeros(size, dtype=dtype)
    for name in BRANCH_FIELDS:
        values[name] = np.zeros(elements.size * MAX_ELEMENTS, dtype=np.int64)
    for name in SCALAR_FIELDS:
        dtype = (
            np.int64
            if name == "len_discharge_element"
            else (
                bool if name == "set_zero" else complex if name == "a0_error" else float
            )
        )
        values[name] = np.zeros(elements.size, dtype=dtype)
    return WorkArrays(
        **values,
        coef_start=coef_start,
        coef_capacity=coef_capacity,
        integration_start=starts,
        integration_capacity=capacities,
    )


def payload(work):
    return sum(a.nbytes for a in {id(a): a for a in work}.values())


def state_arrays(state):
    return {id(a): a for a in (*vars(state.storage).values(), *state.work)}


def state_payload(state):
    return sum(a.nbytes for a in state_arrays(state).values())


def validate_work(work, ncoef, nint):
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
        a = getattr(work, name)
        if (
            a.dtype != np.complex128
            or a.shape != work.old_coef.shape
            or not a.flags.c_contiguous
        ):
            raise ValueError(f"Invalid compact work array: {name}")
    for name in INTEGRATION_FIELDS:
        a = getattr(work, name)
        dtype = (
            np.complex128
            if name in ("exp_array_m", "exp_array_p", "z_integral")
            else np.float64
        )
        if a.dtype != dtype or a.shape != work.thetas.shape or not a.flags.c_contiguous:
            raise ValueError(f"Invalid compact work array: {name}")
    for name in BRANCH_FIELDS:
        a = getattr(work, name)
        if a.dtype != np.int64 or a.shape != (ncoef.size * MAX_ELEMENTS,):
            raise ValueError(f"Invalid compact branch array: {name}")
    if np.any(work.len_discharge_element < 0) or np.any(
        work.len_discharge_element > MAX_ELEMENTS
    ):
        raise ValueError("Invalid compact branch-event count")


def import_work(work, legacy, elements):
    """Explicit test/compatibility initialization, never used in iterations."""
    for i, e in enumerate(elements):
        target = work_record(work, i)
        for name in FIELDS:
            if name in ("thetas", "dpsi_corr"):
                target[name][: e["nint"]] = e[name][: e["nint"]]
            elif name in SCALAR_FIELDS:
                target[name] = legacy[name][i]
            else:
                n = (
                    e["ncoef"]
                    if name in COEFFICIENT_FIELDS + ("integral",)
                    else (MAX_ELEMENTS if name in BRANCH_FIELDS else e["nint"])
                )
                target[name][:n] = legacy[name][i, :n]


def export_work(work, dtype):
    """Explicit padded diagnostic export; not a solver execution bridge."""
    result = np.zeros(work.set_zero.size, dtype=dtype)
    for i in range(result.size):
        record = work_record(work, i)
        for name in dtype.names:
            if name == "coef":
                continue
            if name in SCALAR_FIELDS:
                result[name][i] = record[name]
            else:
                value = record[name]
                result[name][i, : value.size] = value
    return result
