"""Error-specific scalar state and numeric views of the established compact layout."""

import operator
from collections import namedtuple
from dataclasses import dataclass

import numba as nb
import numpy as np
from numba.extending import overload

from .hpc_adaptive import AdaptiveState, AdaptiveStorage
from .hpc_branch import prepare_direct, publish_direct
from .hpc_connectivity import ErrorFractures
from .hpc_work import (
    ErrorWorkRecord,
    allocate_work,
    work_record,
)

SCALARS = ("q", "error", "error_old", "error_old2", "error_coef")
scalar_dtype = np.dtype([(name, np.float64) for name in SCALARS])
_ElementRecord = namedtuple(
    "ElementRecord",
    (
        "metadata",
        "scalars",
        "index",
        "coef_data",
        "coef_start",
        "integration_start",
        "integration_capacity",
        "thetas",
        "dpsi_corr",
    ),
)
_ElementArrays = namedtuple(
    "ElementArrays",
    (
        "metadata",
        "scalars",
        "coef_data",
        "coef_start",
        "integration_start",
        "integration_capacity",
        "thetas",
        "dpsi_corr",
    ),
)


class ElementRecord(_ElementRecord):
    __slots__ = ()

    def __getitem__(self, key):
        i = self.index
        if key == "coef":
            c, n = self.coef_start[i], self.metadata["ncoef"][i]
            return self.coef_data[c : c + n]
        if key in ("thetas", "dpsi_corr"):
            s, n = self.integration_start[i], self.integration_capacity[i]
            return getattr(self, key)[s : s + n]
        return (self.scalars if key in SCALARS else self.metadata)[key][i]

    def __setitem__(self, key, value):
        if key not in SCALARS:
            raise TypeError("Fixed error geometry and orders are read-only")
        self.scalars[key][self.index] = value


class ElementArrays(_ElementArrays):
    __slots__ = ()

    @property
    def size(self):
        return self.metadata.size

    def __getitem__(self, key):
        if isinstance(key, str):
            return (self.scalars if key in SCALARS else self.metadata)[key]
        return ElementRecord(
            self.metadata,
            self.scalars,
            key,
            self.coef_data,
            self.coef_start,
            self.integration_start,
            self.integration_capacity,
            self.thetas,
            self.dpsi_corr,
        )


@overload(operator.getitem)
def _getitem(value, key):
    if (
        isinstance(value, nb.types.NamedTuple)
        and value.instance_class is ElementRecord
        and isinstance(key, nb.types.StringLiteral)
    ):
        name = key.literal_value
        if name == "coef":

            def array(value, key):
                i = value.index
                c, n = value.coef_start[i], value.metadata[i]["ncoef"]
                return value.coef_data[c : c + n]

            return array
        if name in ("thetas", "dpsi_corr"):
            field = ElementRecord._fields.index(name)

            def cache(value, key):
                i = value.index
                s, n = value.integration_start[i], value.integration_capacity[i]
                return value[field][s : s + n]

            return cache
        if name in SCALARS:

            def scalar(value, key):
                return value.scalars[value.index][name]

            return scalar

        def fixed(value, key):
            return value.metadata[value.index][name]

        return fixed
    if isinstance(value, nb.types.NamedTuple) and value.instance_class is ElementArrays:
        if isinstance(key, nb.types.StringLiteral):
            name = key.literal_value
            if name in SCALARS:

                def mutable(value, key):
                    return value.scalars[name]

                return mutable

            def column(value, key):
                return value.metadata[name]

            return column
        if not isinstance(key, nb.types.Integer):
            return None

        def row(value, key):
            return ElementRecord(
                value.metadata,
                value.scalars,
                key,
                value.coef_data,
                value.coef_start,
                value.integration_start,
                value.integration_capacity,
                value.thetas,
                value.dpsi_corr,
            )

        return row


@overload(operator.setitem)
def _setitem(value, key, item):
    if (
        isinstance(value, nb.types.NamedTuple)
        and value.instance_class is ElementRecord
        and isinstance(key, nb.types.StringLiteral)
        and key.literal_value in SCALARS
    ):
        name = key.literal_value

        def scalar(value, key, item):
            value.scalars[value.index][name] = item

        return scalar


def error_work_record(work, i, candidates=None, starts=None):
    if candidates is None:
        return work[i]
    record = work_record(work, i)
    c, n = starts[i], work.coef_capacity[i]
    return ErrorWorkRecord(*record, candidates[c : c + n])


@overload(error_work_record)
def _error_work_record(work, i, candidates=None, starts=None):
    if candidates is None or isinstance(
        candidates, (nb.types.NoneType, nb.types.Omitted)
    ):

        def legacy(work, i, candidates=None, starts=None):
            return work[i]

        return legacy

    def compact(work, i, candidates=None, starts=None):
        record = work_record(work, i)
        c, n = starts[i], work.coef_capacity[i]
        return ErrorWorkRecord(
            record.coef0,
            record.coef1,
            record.old_coef,
            record.coef_error,
            record.integral,
            record.thetas,
            record.dpsi_corr,
            record.exp_array_m,
            record.exp_array_p,
            record.z_integral,
            record.phi,
            record.psi,
            record.dpsi,
            record.sign_array,
            record.discharge_element,
            record.element_pos,
            record.len_discharge_element,
            record.a0_error,
            record.error,
            record.set_zero,
            record.branch_valid,
            candidates[c : c + n],
        )

    return compact


def selected_element(elements, selection, i):
    return elements[selection[i]] if selection.dtype.names is None else selection[i]


@overload(selected_element, inline="always")
def _selected_element(elements, selection, i):
    if isinstance(selection.dtype, nb.types.Integer):

        def compact(elements, selection, i):
            return elements[selection[i]]

        return compact

    def legacy(elements, selection, i):
        return selection[i]

    return legacy


@dataclass
class ErrorState:
    state: AdaptiveState
    elements: ElementArrays
    normal: ElementArrays
    fractures: ErrorFractures
    branch_stats: dict


def prepare(owner, fractures):
    """Share solved geometry/layouts; isolate all mutable error numerical state."""
    source = owner.state
    source.storage.validate()
    for array in (
        source.storage.coef_data,
        source.storage.candidate_coef_data,
        source.storage.old_coef_data,
        *source.work,
    ):
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError("Nonfinite compact normal state")
    s = source.storage
    current = np.zeros_like(s.coef_data)
    candidates = np.zeros_like(current)
    work = allocate_work(
        s.elements,
        s.coef_start,
        s.coef_capacity,
        scratch_groups=7,
        integration_source=source.work,
        coefficient_metrics=False,
    )
    history = work.old_coef
    storage = AdaptiveStorage(
        s.elements, current, candidates, history, s.coef_start, s.coef_capacity
    )
    scalars = np.empty(s.elements.size, dtype=scalar_dtype)
    for name in SCALARS:
        scalars[name] = s.elements[name]
    normal = ElementArrays(
        s.elements,
        s.elements,
        s.coef_data,
        s.coef_start,
        source.work.integration_start,
        source.work.integration_capacity,
        source.work.thetas,
        source.work.dpsi_corr,
    )
    private_fractures = ErrorFractures(
        fractures.metadata,
        fractures.ids,
        fractures.indptr,
        fractures["error_constant"].copy(),
    )
    ids = np.flatnonzero(s.elements["_type"] == 1)
    prepared, stats = prepare_direct(
        work, s.elements, private_fractures, ids, s.elements["nint"]
    )
    work = prepared.work
    publish_direct(
        work, prepared.ids, prepared.starts, prepared.counts, prepared.values
    )
    error_elements = ElementArrays(
        s.elements,
        scalars,
        current,
        s.coef_start,
        work.integration_start,
        work.integration_capacity,
        work.thetas,
        work.dpsi_corr,
    )
    return ErrorState(
        AdaptiveState(storage, work), error_elements, normal, private_fractures, stats
    )
