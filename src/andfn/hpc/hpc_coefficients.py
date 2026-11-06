"""Compile-time coefficient views; no containers or coefficient copies."""

import numba as nb
from numba.extending import overload

from .hpc_work import work_record


def coefficient_view(element, data=None, starts=None):
    if data is None:
        return element["coef"][: element["ncoef"]]
    start = starts[element["_id"]]
    return data[start : start + element["ncoef"]]


@overload(coefficient_view, inline="always")
def _coefficient_view(element, data=None, starts=None):
    if data is None or isinstance(data, (nb.types.NoneType, nb.types.Omitted)):

        def legacy(element, data=None, starts=None):
            return element["coef"][: element["ncoef"]]

        return legacy

    def compact(element, data=None, starts=None):
        start = starts[element["_id"]]
        return data[start : start + element["ncoef"]]

    return compact


def candidate_view(element, work, data=None, starts=None):
    if data is None:
        return work["coef"][: element["ncoef"]]
    start = starts[element["_id"]]
    return data[start : start + element["ncoef"]]


@overload(candidate_view, inline="always")
def _candidate_view(element, work, data=None, starts=None):
    if data is None or isinstance(data, (nb.types.NoneType, nb.types.Omitted)):

        def legacy(element, work, data=None, starts=None):
            return work["coef"][: element["ncoef"]]

        return legacy

    def compact(element, work, data=None, starts=None):
        start = starts[element["_id"]]
        return data[start : start + element["ncoef"]]

    return compact


def candidate_at(element, work, i, data=None, starts=None):
    if data is not None:
        return candidate_view(element, None, data, starts)
    return candidate_view(element, work_record(work, i))


@overload(candidate_at, inline="always")
def _candidate_at(element, work, i, data=None, starts=None):
    if data is None or isinstance(data, (nb.types.NoneType, nb.types.Omitted)):

        def legacy(element, work, i, data=None, starts=None):
            return candidate_view(element, work_record(work, i))

        return legacy

    def compact(element, work, i, data=None, starts=None):
        return candidate_view(element, None, data, starts)

    return compact
