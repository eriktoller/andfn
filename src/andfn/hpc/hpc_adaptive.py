"""Batch adaptation for the opt-in flat-coefficient main solver.

Integration grids remain in padded coefficient-free records. Work remains
legacy padded. A single state reference publishes both records and caches after
construction/validation; no active layout is changed by parallel assessment.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numba as nb
import numpy as np

from andfn.element import element_dtype_hpc
from andfn.storage import (
    FixedOrderStorage,
    _counts,
    _layout,
    fixed_order_from_hpc,
    growth_capacities,
)

from . import CACHE, PARALLEL
from . import hpc_math_functions as mf
from . import hpc_solve as hs
from .hpc_coefficients import coefficient_view

_MAX_REQUEST_INCREMENT = (
    np.iinfo(np.int64).max // 2 - element_dtype_hpc["coef"].shape[0]
)


@dataclass(frozen=True)
class AdaptiveState:
    storage: FixedOrderStorage
    work: np.ndarray


@dataclass
class AdaptiveSolver:
    state: AdaptiveState
    batches: list = field(default_factory=list)
    transition_callback: object = None
    assessments: list = field(default_factory=list)


def adaptive_from_hpc(elements):
    """Import logical state; reject initialized padding lost by flat conversion."""
    for e in elements:
        n = e["ncoef"]
        if np.any(e["coef"][n:] != 0) or np.any(e["old_coef"][n:] != 0):
            raise ValueError("Adaptive import requires zero unused coefficient padding")
    storage = fixed_order_from_hpc(elements)
    # Legacy ratio checks require at least coef[1] for every non-well.
    if np.any((elements["_type"] != 2) & (elements["ncoef"] < 2)):
        raise ValueError("Adaptive non-well elements require at least two coefficients")
    return AdaptiveSolver(
        AdaptiveState(storage, np.zeros(elements.size, dtype=hs.dtype_work))
    )


@nb.njit(parallel=PARALLEL, cache=CACHE)
def assess_growth(
    elements, coef_data, coef_start, nit, max_error, ratio_limit, maximum, increase
):
    """Read-only assessment; retain the legacy old-order maximum check."""
    if increase <= 0 or increase > _MAX_REQUEST_INCREMENT:
        raise ValueError("COEF_INCREASE would overflow adaptive request arithmetic")
    requested = elements["ncoef"].copy()
    integrations = elements["nint"].copy()
    required = np.zeros(elements.size, dtype=np.bool_)
    for i in nb.prange(elements.size):
        e = elements[i]
        if nit < 2 or e["_type"] == 2:
            continue
        coefs = coefficient_view(e, coef_data, coef_start)
        coef0 = np.mean(np.abs(coefs[1:3])) + 1e-30
        coef1 = np.max(np.abs(coefs[-3:]))
        if coef0 == 0.0:
            ratio = 0.0
        else:
            ratio = coef1 / coef0 * 0
        if np.max(np.abs(coefs[1:2])) < max_error / 1e10:
            ratio = 0.0
        if (
            e["error"] > e["error_old"]
            and e["error"] > e["error_old2"]
            and e["error"] > max_error
            and nit > 5
        ):
            ratio = ratio_limit + 1.0
        if (
            np.abs(np.sum(coefs[-3:]) / (np.sum(coefs[:-3]) + 1e-30)) > ratio_limit
            and e["error"] > max_error
        ):
            ratio = ratio_limit + 1.0
        if ratio > ratio_limit and e["ncoef"] < maximum and nit > 1:
            requested[i] = e["ncoef"] + increase
            integrations[i] = requested[i] * 2
            required[i] = True
    return requested, integrations, required


def validate_requests(state, requested, integrations, required, maximum, increase):
    state.storage.validate()
    elements = state.storage.elements
    nc = _counts(requested, elements.size, "requested_ncoef")
    ni = _counts(integrations, elements.size, "requested_nint")
    if (
        not isinstance(required, np.ndarray)
        or required.dtype != np.bool_
        or required.shape != (elements.size,)
    ):
        raise ValueError("Invalid growth mask")
    if increase <= 0 or increase > np.iinfo(np.int64).max // 2:
        raise ValueError("COEF_INCREASE must be positive and safely representable")
    if np.any(nc < elements["ncoef"]):
        raise ValueError("Requested ncoef cannot shrink")
    expected = elements["ncoef"].astype(object) + int(increase)
    if np.any(required & ((elements["_type"] == 2) | (elements["ncoef"] >= maximum))):
        raise ValueError("Ineligible adaptive growth request")
    if any(int(nc[i]) != int(expected[i]) for i in np.flatnonzero(required)):
        raise ValueError("Requested ncoef must equal old order plus COEF_INCREASE")
    if np.any(nc[~required] != elements["ncoef"][~required]):
        raise ValueError("Unexpected coefficient change without growth")
    # Explicitly approved exception: the legacy rule can shrink logical nint.
    if any(int(ni[i]) != 2 * int(nc[i]) for i in np.flatnonzero(required)) or np.any(
        ni[~required] != elements["nint"][~required]
    ):
        raise ValueError("Requested nint must follow the element's legacy growth rule")
    if np.any(nc > element_dtype_hpc["coef"].shape[0]) or np.any(
        ni > element_dtype_hpc["thetas"].shape[0]
    ):
        raise ValueError("Growth exceeds retained integration/work storage")
    if state.work.dtype != hs.dtype_work or state.work.shape != (elements.size,):
        raise ValueError("Invalid retained adaptive work storage")
    return nc, ni


@nb.njit(cache=CACHE)
def rebuild_growth_caches(elements, work, required):
    for i in range(elements.size):
        if not required[i]:
            continue
        e = elements[i]
        mf.calc_thetas(e["nint"], e["_type"], e["thetas"][: e["nint"]])
        # Grid changes invalidate event topology, not fracture CSR. Existing
        # circle/line solves rebuild topology and q-dependent corrections.
        work[i]["len_discharge_element"] = 0
        mf.fill_exp_array(e["nint"], e["thetas"], work[i]["exp_array_m"], -1)
        mf.fill_exp_array(e["nint"], e["thetas"], work[i]["exp_array_p"], 1)
        mf.fill_z_integral(e, work[i])


def build_replacement(
    state,
    requested,
    integrations,
    required,
    maximum,
    increase,
    *,
    rebuild=rebuild_growth_caches,
):
    """Construct privately; failures leave every published array untouched."""
    nc, ni = validate_requests(
        state, requested, integrations, required, maximum, increase
    )
    if not np.any(required):
        return state, None
    start_time = perf_counter()
    old = state.storage
    capacities = growth_capacities(nc, old.coef_capacity)
    starts, size = _layout(capacities)
    repack = not np.array_equal(capacities, old.coef_capacity)
    buffers = [np.zeros(size, dtype=np.complex128) for _ in range(3)]
    copied = 0
    for i, length in enumerate(old.elements["ncoef"]):
        before, after = old.coef_start[i], starts[i]
        for source, target in zip(
            (old.coef_data, old.candidate_coef_data, old.old_coef_data), buffers
        ):
            target[after : after + length] = source[before : before + length]
            copied += int(length) * source.itemsize
    elements = old.elements.copy()
    work = state.work.copy()
    for i in np.flatnonzero(required):
        previous = elements["nint"][i]
        # Padded integration bridge already has maximum physical capacity.
        # Newly exposed entries are initialized, but existing values (including
        # entries left behind by a logical shrink) retain legacy behavior.
        for name in ("thetas", "dpsi_corr"):
            elements[name][i, previous : ni[i]] = 0
    elements["ncoef"], elements["nint"] = nc, ni
    storage = FixedOrderStorage(
        elements, buffers[0], buffers[1], buffers[2], starts, capacities
    )
    staging_seconds = perf_counter() - start_time
    cache_start = perf_counter()
    rebuild(elements, work, required)
    cache_seconds = perf_counter() - cache_start
    storage.validate()
    for i in range(elements.size):
        start, length = starts[i], nc[i]
        if any(
            not np.all(np.isfinite(buffer[start : start + length]))
            for buffer in buffers
        ):
            raise ValueError("Nonfinite replacement coefficients")
        if required[i]:
            for name in ("thetas", "dpsi_corr"):
                if not np.all(np.isfinite(elements[name][i, : ni[i]])):
                    raise ValueError("Nonfinite replacement integration values")
            for name in ("exp_array_m", "exp_array_p"):
                if not np.all(np.isfinite(work[name][i, : ni[i]])):
                    raise ValueError("Nonfinite replacement integration cache")
            if work["len_discharge_element"][i] != 0:
                raise ValueError(
                    "Rebuilt integration grid must invalidate branch events"
                )
    stats = {
        "expanded_elements": int(np.count_nonzero(required)),
        "coefficient_repacks": int(repack),
        "integration_repacks": 0,
        "coefficient_bytes_copied": copied,
        "integration_bytes_copied": int(
            old.elements["thetas"].nbytes + old.elements["dpsi_corr"].nbytes
        ),
        "work_bytes_copied": state.work.nbytes,
        "metadata_bytes_copied": int(
            old.elements.nbytes
            - old.elements["thetas"].nbytes
            - old.elements["dpsi_corr"].nbytes
        ),
        "total_bytes_copied": int(copied + old.elements.nbytes + state.work.nbytes),
        "transaction_staging_seconds": staging_seconds,
        "cache_rebuild_seconds": cache_seconds,
        "old_payload": old.nbytes + state.work.nbytes,
        "replacement_payload": storage.nbytes + work.nbytes,
    }
    return AdaptiveState(storage, work), stats


@nb.njit(cache=CACHE)
def recalculate_candidates(
    elements,
    fractures,
    work,
    required,
    max_error,
    nit,
    cnt_error,
    damping,
    coef_data,
    coef_start,
    candidates,
):
    hs.calculate_candidates(
        elements.size,
        elements,
        fractures,
        work,
        max_error,
        nit,
        cnt_error,
        damping,
        coef_data,
        coef_start,
        candidates,
        required,
    )


def adaptive_step(
    owner, fractures, max_error, nit, cnt_error, ratio_limit, maximum, increase, damping
):
    def observe(label, requests=None):
        if owner.transition_callback is not None:
            owner.transition_callback(label, owner.state, requests)

    observe("before_first_candidate")
    state = owner.state
    s = state.storage
    if not 1.0 > max_error:
        return
    arguments = (
        s.elements.size,
        s.elements,
        fractures,
        state.work,
        max_error,
        nit,
        cnt_error,
        damping,
        s.coef_data,
        s.coef_start,
        s.candidate_coef_data,
    )
    if owner.transition_callback is None:
        hs.element_solver(*arguments)
    else:
        hs.calculate_candidates(*arguments)
        observe("after_first_candidate")
        hs.commit_coefficients(
            s.elements,
            state.work,
            damping,
            s.coef_data,
            s.coef_start,
            s.candidate_coef_data,
        )
    del arguments
    observe("after_first_commit")
    if nit < 2:
        return
    detection_start = perf_counter()
    requests = assess_growth(
        s.elements,
        s.coef_data,
        s.coef_start,
        nit,
        max_error,
        ratio_limit,
        maximum,
        increase,
    )
    detection_seconds = perf_counter() - detection_start
    observe("after_assessment", requests)
    if np.any(requests[2]):
        transaction_start = perf_counter()
        replacement, stats = build_replacement(state, *requests, maximum, increase)
        stats["transaction_seconds"] = perf_counter() - transaction_start
    else:
        # Typed assessment produced unchanged lengths. The active state was
        # validated at import/publication; do not rescan it on every no-op.
        replacement, stats = state, None
    owner.assessments.append(
        {
            "iteration": nit,
            "detection_seconds": detection_seconds,
            "expanded_elements": int(np.count_nonzero(requests[2])),
        }
    )
    # Single Python publication after every failure-prone construction step.
    owner.state = replacement
    observe("after_growth_publication", requests)
    s = owner.state.storage
    recalc_start = perf_counter()
    if stats is not None:
        recalculate_candidates(
            s.elements,
            fractures,
            owner.state.work,
            requests[2],
            max_error,
            nit,
            cnt_error,
            damping,
            s.coef_data,
            s.coef_start,
            s.candidate_coef_data,
        )
        stats["detection_seconds"] = detection_seconds
        stats["recalculation_seconds"] = perf_counter() - recalc_start
        owner.batches.append(stats)
    observe("after_adaptive_candidate", requests)
    owner.state.work["set_zero"][:] = False
    hs.commit_coefficients(
        s.elements,
        owner.state.work,
        damping,
        s.coef_data,
        s.coef_start,
        s.candidate_coef_data,
    )
    observe("after_second_commit", requests)


def solve_adaptive(
    owner, fractures, discharge_int, constants, *, iteration_callback=None
):
    """Opt-in compact adaptive main solve; retains the legacy adaptive path."""
    owner.state.storage.validate()
    return hs.solve(
        fractures,
        owner.state.storage.elements,
        discharge_int,
        constants,
        None,
        coef_data=owner.state.storage.coef_data,
        coef_start=owner.state.storage.coef_start,
        candidate_coef_data=owner.state.storage.candidate_coef_data,
        iteration_callback=iteration_callback,
        _adaptive_owner=owner,
    )
