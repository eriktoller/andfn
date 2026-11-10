"""Adaptive solving over compact coefficient and worker-owned cache storage.

The compact owner has independent coefficient, cache, scratch and branch
buffers. Construction never mutates active state; publication occurs only at
the Python barrier, after all allocations and cache rebuilding succeed.
"""

from collections import namedtuple
from dataclasses import dataclass, field
from time import perf_counter

import numba as nb
import numpy as np

from andfn.storage import (
    FixedOrderStorage,
    _counts,
    _layout,
    element_metadata_dtype,
    growth_capacities,
)

from . import CACHE, PARALLEL
from . import hpc_math_functions as mf
from . import hpc_solve as hs
from .hpc_branch import prepare_direct as prepare_branches
from .hpc_branch import publish_direct
from .hpc_coefficients import coefficient_view
from .hpc_work import (
    CACHE_FIELDS,
    COEFFICIENT_FIELDS,
    INTEGRATION_FIELDS,
    WorkArrays,
    integration_view,
    invalidate_branch,
    resize_scratch,
    scratch_fields,
    state_arrays,
    state_payload,
    validate_work,
    work_record,
)


@dataclass(frozen=True)
class AdaptiveState:
    storage: FixedOrderStorage
    work: WorkArrays


class AdaptiveStorage(FixedOrderStorage):
    element_dtype = element_metadata_dtype


@dataclass(frozen=True)
class GrowthTransaction:
    state: AdaptiveState
    previous: AdaptiveState
    ids: np.ndarray
    caches: WorkArrays
    requested_ncoef: np.ndarray
    requested_nint: np.ndarray


CachePatch = namedtuple("CachePatch", CACHE_FIELDS)
CacheStage = namedtuple(
    "CacheStage",
    CACHE_FIELDS
    + ("integration_start", "integration_capacity", "len_discharge_element"),
)


@dataclass(frozen=True)
class LogicalGrowth:
    state: AdaptiveState
    ids: np.ndarray
    ncoef: np.ndarray
    nint: np.ndarray
    starts: np.ndarray
    caches: tuple


@dataclass
class AdaptiveSolver:
    state: AdaptiveState
    batches: list = field(default_factory=list)
    transition_callback: object = None
    assessments: list = field(default_factory=list)
    debug_validation: bool = False
    before_publication: object = None
    branch_initialized: bool = False
    branch_batches: list = field(default_factory=list)


def validate_state(state, *, finite=False):
    """Public structural validation; complete finite scans are opt-in."""
    state.storage.validate()
    e = state.storage.elements
    validate_work(state.work, e["ncoef"], e["nint"])
    if any(not a.flags.writeable for a in state_arrays(state).values()):
        raise ValueError("Compact published buffers must be writable")
    for i, count in enumerate(state.work.len_discharge_element):
        a = state.work.branch_start[i]
        ids = state.work.discharge_element[a : a + count]
        positions = state.work.element_pos[a : a + count]
        signs = state.work.sign_array[a : a + count]
        if (
            np.any(ids < 0)
            or np.any(ids >= e.size)
            or np.any(positions < 0)
            or np.any(positions >= e["nint"][i] - 1)
            or np.any((signs != -1) & (signs != 1))
        ):
            raise ValueError("Invalid live branch event")
    if finite:
        for name in e.dtype.names:
            if e[name].dtype.kind in "fc" and not np.all(np.isfinite(e[name])):
                raise ValueError("Nonfinite element metadata")
        for i, element in enumerate(e):
            c, n = state.storage.coef_start[i], element["ncoef"]
            for a in (
                state.storage.coef_data,
                state.storage.candidate_coef_data,
                state.storage.old_coef_data,
            ):
                if not np.all(np.isfinite(a[c : c + n])):
                    raise ValueError("Nonfinite coefficient prefix")
            for name in COEFFICIENT_FIELDS + ("integral",):
                if name in scratch_fields(state.work):
                    continue
                if not np.all(np.isfinite(getattr(state.work, name)[c : c + n])):
                    raise ValueError("Nonfinite coefficient work prefix")
            for name in INTEGRATION_FIELDS:
                if name in scratch_fields(state.work):
                    continue
                start = state.work.integration_start[i]
                if not np.all(
                    np.isfinite(
                        getattr(state.work, name)[start : start + element["nint"]]
                    )
                ):
                    raise ValueError("Nonfinite cache prefix")
        for name in ("a0_error", "error"):
            if not np.all(np.isfinite(getattr(state.work, name))):
                raise ValueError("Nonfinite scalar work")
        for name in scratch_fields(state.work):
            if not np.all(np.isfinite(getattr(state.work, name))):
                raise ValueError("Nonfinite worker scratch")


@nb.njit(parallel=PARALLEL, cache=CACHE)
def assess_growth(
    elements, coef_data, coef_start, nit, max_error, ratio_limit, maximum, increase
):
    """Read-only assessment; retain the legacy old-order maximum check."""
    if increase <= 0 or increase > np.iinfo(np.int64).max // 2:
        raise ValueError("COEF_INCREASE would overflow adaptive request arithmetic")
    requested = elements["ncoef"].copy()
    integrations = elements["nint"].copy()
    required = np.zeros(elements.size, dtype=np.bool_)
    overflow = np.zeros(elements.size, dtype=np.bool_)
    for i in nb.prange(elements.size):
        e = elements[i]
        if nit >= 2 and e["_type"] != 2:
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
                if e["ncoef"] > np.iinfo(np.int64).max // 2 - increase:
                    overflow[i] = True
                else:
                    requested[i] = e["ncoef"] + increase
                    integrations[i] = requested[i] * 2
                    required[i] = True
    if np.any(overflow):
        raise ValueError("Adaptive request arithmetic would overflow int64")
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
    if not isinstance(state.work, WorkArrays):
        raise TypeError("Invalid compact adaptive work storage")
    validate_work(state.work, elements["ncoef"], elements["nint"])
    return nc, ni


@nb.njit(cache=CACHE)
def rebuild_cache_segments(e, thetas, exp_m, exp_p, z_integral):
    n = e["nint"]
    mf.calc_thetas(n, e["_type"], thetas[:n])
    mf.fill_exp_array(n, thetas, exp_m, -1)
    mf.fill_exp_array(n, thetas, exp_p, 1)
    mf.fill_z_values(e, exp_p, z_integral)


@nb.njit(cache=CACHE)
def rebuild_cache_values(e, record):
    thetas = integration_view(e, record, "thetas")
    rebuild_cache_segments(
        e, thetas, record["exp_array_m"], record["exp_array_p"], record["z_integral"]
    )


@nb.njit(cache=CACHE)
def rebuild_growth_caches(elements, work, required):
    for i in range(elements.size):
        if not required[i]:
            continue
        e = elements[i]
        if hasattr(work, "integration_start") and not hasattr(work, "coef_start"):
            a, n = work.integration_start[i], e["nint"]
            rebuild_cache_segments(
                e,
                work.thetas[a : a + n],
                work.exp_array_m[a : a + n],
                work.exp_array_p[a : a + n],
                work.z_integral[a : a + n],
            )
            work.len_discharge_element[i] = 0
            continue
        record = work_record(work, i)
        # Grid changes invalidate event topology, not fracture CSR. Existing
        # circle/line solves rebuild topology and q-dependent corrections.
        invalidate_branch(record)
        rebuild_cache_values(e, record)


@nb.njit(cache=CACHE)
def validate_affected(
    elements, work, starts, capacities, ids, nc, ni, maximum, increase
):
    """Only affected metadata/layout endpoints; never scan active values."""
    if increase <= 0 or increase > np.iinfo(np.int64).max // 2:
        raise ValueError("Invalid coefficient increment")
    coefficient_repack, integration_repack = False, False
    for i in ids:
        if i < 0 or i >= elements.size or elements[i]["_id"] != i:
            raise ValueError("Invalid affected element ID")
        e = elements[i]
        if e["_type"] == 2 or e["ncoef"] >= maximum:
            raise ValueError("Ineligible adaptive growth request")
        if nc[i] != e["ncoef"] + increase or nc[i] < e["ncoef"]:
            raise ValueError("Invalid requested coefficient length")
        if ni[i] != 2 * nc[i] or ni[i] < 0:
            raise ValueError("Invalid requested integration length")
        if nc[i] > np.iinfo(np.int64).max // 2:
            raise ValueError("Requested integration extent would overflow int64")
        if starts[i] < 0 or starts[i] + capacities[i] > work.old_coef.size:
            raise ValueError("Invalid affected coefficient segment")
        if work.integration_start[i] < 0 or (
            work.integration_start[i] + work.integration_capacity[i] > work.thetas.size
        ):
            raise ValueError("Invalid affected integration segment")
        coefficient_repack |= nc[i] > capacities[i]
        integration_repack |= ni[i] > work.integration_capacity[i]
    return coefficient_repack, integration_repack


@nb.njit(cache=CACHE)
def patch_layout(ids, ni):
    starts = np.empty(ids.size, dtype=np.int64)
    total = 0
    for j in range(ids.size):
        length = ni[ids[j]]
        if length < 0 or total > np.iinfo(np.int64).max - length:
            raise ValueError("Invalid affected cache extent")
        starts[j] = total
        total += length
    return starts, total


@nb.njit(cache=CACHE)
def prepare_cache_patch(elements, work, ids, nc, ni, starts, caches, affected):
    preserved = 0
    for j in range(ids.size):
        i = ids[j]
        affected[j] = elements[i]
        affected[j]["ncoef"], affected[j]["nint"] = nc[i], ni[i]
        a, n = starts[j], ni[i]
        previous = min(elements[i]["nint"], n)
        preserved += previous * 24
        before = work.integration_start[i]
        for k in range(previous):
            caches.dpsi_corr[a + k] = work.dpsi_corr[before + k]
            caches.z_integral[a + k] = work.z_integral[before + k]
        rebuild_cache_segments(
            affected[j],
            caches.thetas[a : a + n],
            caches.exp_array_m[a : a + n],
            caches.exp_array_p[a : a + n],
            caches.z_integral[a : a + n],
        )
    return preserved


@nb.njit(cache=CACHE)
def validate_patch(caches):
    for k in range(caches.thetas.size):
        if not (
            np.isfinite(caches.thetas[k])
            and np.isfinite(caches.dpsi_corr[k])
            and np.isfinite(caches.exp_array_m[k])
            and np.isfinite(caches.exp_array_p[k])
            and np.isfinite(caches.z_integral[k])
        ):
            raise ValueError("Nonfinite replacement integration cache")


@nb.njit(cache=CACHE)
def publish_logical(
    elements, current, candidate, old, coef_start, work, ids, nc, ni, starts, caches
):
    for j in nb.prange(ids.size):
        i = ids[j]
        c = coef_start[i]
        for k in range(elements[i]["ncoef"], nc[i]):
            current[c + k] = 0
            candidate[c + k] = 0
            old[c + k] = 0
            work.old_coef[c + k] = 0
            work.coef_error[c + k] = 0
        a, b = work.integration_start[i], starts[j]
        for k in range(ni[i]):
            work.thetas[a + k] = caches.thetas[b + k]
            work.dpsi_corr[a + k] = caches.dpsi_corr[b + k]
            work.exp_array_m[a + k] = caches.exp_array_m[b + k]
            work.exp_array_p[a + k] = caches.exp_array_p[b + k]
            work.z_integral[a + k] = caches.z_integral[b + k]
        work.len_discharge_element[i] = 0
        work.branch_valid[i] = False
    for j in nb.prange(ids.size):
        i = ids[j]
        elements[i]["ncoef"], elements[i]["nint"] = nc[i], ni[i]


def prepare_logical(state, ids, nc, ni, rebuild):
    s, w = state.storage, state.work
    destinations = (
        s.elements,
        s.coef_data,
        s.candidate_coef_data,
        s.old_coef_data,
        w.old_coef,
        w.coef_error,
        w.len_discharge_element,
        w.branch_valid,
        w.thetas,
        w.dpsi_corr,
        w.exp_array_m,
        w.exp_array_p,
        w.z_integral,
    )
    if any(not a.flags.writeable for a in destinations):
        raise ValueError("Affected publication buffers must be writable")
    extraction = perf_counter()
    starts, size = patch_layout(ids, ni)
    affected = np.empty(ids.size, dtype=state.storage.elements.dtype)
    caches = CachePatch(
        np.zeros(size),
        np.zeros(size),
        np.zeros(size, dtype=complex),
        np.zeros(size, dtype=complex),
        np.zeros(size, dtype=complex),
    )
    extracted = perf_counter() - extraction
    cache_start = perf_counter()
    preserved = prepare_cache_patch(
        state.storage.elements, state.work, ids, nc, ni, starts, caches, affected
    )
    callback_arrays = ()
    if rebuild is not rebuild_growth_caches:
        lengths = ni[ids]
        event_counts = np.zeros(ids.size, dtype=np.int64)
        mask = np.ones(ids.size, dtype=bool)
        callback_arrays = (lengths, event_counts, mask)
        rebuild(affected, CacheStage(*caches, starts, lengths, event_counts), mask)
        if np.any(event_counts != 0):
            raise ValueError("Rebuilt integration grid must invalidate branch events")
    cache_seconds = perf_counter() - cache_start
    validation = perf_counter()
    validate_patch(caches)
    finite_seconds = perf_counter() - validation
    cache_bytes = sum(a.nbytes for a in caches)
    temporary = sum(
        a.nbytes for a in (*caches, ids, starts, affected, *callback_arrays)
    )
    persistent = state_payload(state)
    stats = {
        "path": "within_capacity",
        "expanded_elements": ids.size,
        "coefficient_repacks": 0,
        "integration_repacks": 0,
        "coefficient_bytes_copied": 0,
        "coefficient_repack_bytes": 0,
        "coefficient_relocation_bytes": 0,
        "integration_bytes_copied": preserved + cache_bytes,
        "cache_rebuild_bytes": cache_bytes,
        "cache_publication_bytes": cache_bytes,
        "work_bytes_copied": 0,
        "branch_event_bytes_copied": 0,
        "metadata_bytes_copied": affected.nbytes,
        "total_bytes_copied": preserved + cache_bytes + affected.nbytes,
        "transaction_temporary_payload": temporary,
        "temporary_array_count": 8 + len(callback_arrays),
        "complete_state_scans": 0,
        "affected_element_scans": 3,
        "transaction_staging_seconds": extracted,
        "request_extraction_seconds": extracted,
        "finite_validation_seconds": finite_seconds,
        "cache_rebuild_seconds": cache_seconds,
        "old_payload": persistent,
        "replacement_payload": persistent,
        "peak_simultaneous_numpy_payload": persistent + temporary,
    }
    return LogicalGrowth(state, ids, nc, ni, starts, caches), stats


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
    """Dispatch no-growth, logical patches and independent physical repacks."""
    if (
        any(not isinstance(a, np.ndarray) for a in (required, requested, integrations))
        or required.dtype != np.bool_
        or required.shape != (state.storage.elements.size,)
        or requested.shape != required.shape
        or integrations.shape != required.shape
        or requested.dtype != np.int64
        or integrations.dtype != np.int64
    ):
        raise ValueError("Invalid growth request arrays")
    extraction_started = perf_counter()
    ids = np.flatnonzero(required)
    extraction_seconds = perf_counter() - extraction_started
    if not ids.size:
        return state, None
    started = perf_counter()
    c, integration = validate_affected(
        state.storage.elements,
        state.work,
        state.storage.coef_start,
        state.storage.coef_capacity,
        ids,
        requested,
        integrations,
        maximum,
        increase,
    )
    validation_seconds = perf_counter() - started
    if not c and not integration:
        transaction, stats = prepare_logical(
            state, ids, requested, integrations, rebuild
        )
        stats["validation_seconds"] = validation_seconds
        stats["request_extraction_seconds"] += extraction_seconds
        # Assessment owns these three arrays until the second commit finishes.
        requests_payload = sum(a.nbytes for a in (requested, integrations, required))
        stats["builder_temporary_array_count"] = stats["temporary_array_count"]
        stats["builder_temporary_payload"] = stats["transaction_temporary_payload"]
        stats["temporary_array_count"] += 3
        stats["transaction_temporary_payload"] += requests_payload
        stats["peak_simultaneous_numpy_payload"] += requests_payload
        return transaction, stats
    transaction, stats = build_repack(
        state, requested, integrations, required, maximum, increase, rebuild=rebuild
    )
    stats["path"] = (
        "both_repack"
        if c and integration
        else "coefficient_repack"
        if c
        else "integration_repack"
    )
    stats["complete_state_scans"] = 2
    return transaction, stats


def build_repack(
    state,
    requested,
    integrations,
    required,
    maximum,
    increase,
    *,
    rebuild=rebuild_growth_caches,
):
    """Prepare a transaction; shared buffers are written only by publication."""
    nc, ni = validate_requests(
        state, requested, integrations, required, maximum, increase
    )
    if not np.any(required):
        return state, None
    start_time = perf_counter()
    old = state.storage
    capacities = growth_capacities(nc, old.coef_capacity)
    repack = not np.array_equal(capacities, old.coef_capacity)
    if not repack:
        capacities = old.coef_capacity
    starts, size = (
        _layout(capacities) if repack else (old.coef_start, old.coef_data.size)
    )
    buffers = [old.coef_data, old.candidate_coef_data, old.old_coef_data]
    copied = 0
    values = {}
    if repack:
        buffers = [np.zeros(size, dtype=np.complex128) for _ in range(3)]
        for name in COEFFICIENT_FIELDS + ("integral",):
            if name in scratch_fields(state.work):
                continue
            values[name] = np.zeros(size, dtype=complex)
        for i, length in enumerate(old.elements["ncoef"]):
            before, after = old.coef_start[i], starts[i]
            sources = (
                old.coef_data,
                old.candidate_coef_data,
                old.old_coef_data,
                state.work.old_coef,
                state.work.coef_error,
            )
            targets = (*buffers, values["old_coef"], values["coef_error"])
            for source, target in zip(sources, targets):
                target[after : after + length] = source[before : before + length]
                copied += int(length) * source.itemsize
    integration_capacity = growth_capacities(ni, state.work.integration_capacity)
    integration_repack = not np.array_equal(
        integration_capacity, state.work.integration_capacity
    )
    if not integration_repack:
        integration_capacity = state.work.integration_capacity
    integration_start, integration_size = (
        _layout(integration_capacity)
        if integration_repack
        else (state.work.integration_start, state.work.thetas.size)
    )
    integration_copied = 0
    if integration_repack:
        for name in INTEGRATION_FIELDS:
            if name in scratch_fields(state.work):
                continue
            values[name] = np.zeros(
                integration_size, dtype=getattr(state.work, name).dtype
            )
        for i, length in enumerate(old.elements["nint"]):
            before, after = state.work.integration_start[i], integration_start[i]
            for name in CACHE_FIELDS:
                # Preserve the active prefix, including a legacy logical shrink.
                length_copy = min(int(length), int(integration_capacity[i]))
                values[name][after : after + length_copy] = getattr(state.work, name)[
                    before : before + length_copy
                ]
                integration_copied += length_copy * values[name].itemsize
    values.update(resize_scratch(state.work, capacities, integration_capacity))
    elements = old.elements
    work = state.work._replace(
        **values,
        coef_start=starts,
        coef_capacity=capacities,
        integration_start=integration_start,
        integration_capacity=integration_capacity,
    )
    storage = AdaptiveStorage(
        elements, buffers[0], buffers[1], buffers[2], starts, capacities
    )
    ids = np.flatnonzero(required)
    affected = elements[ids]
    affected["ncoef"], affected["nint"] = nc[ids], ni[ids]
    cache_capacity = ni[ids]
    cache_start, cache_size = _layout(cache_capacity)
    staged = {
        name: np.zeros(cache_size, dtype=getattr(work, name).dtype)
        for name in CACHE_FIELDS
    }
    staged_copy = 0
    for j, i in enumerate(ids):
        previous = old.elements["nint"][i]
        count = min(int(previous), int(ni[i]))
        for name in ("dpsi_corr", "z_integral"):
            staged[name][cache_start[j] : cache_start[j] + count] = getattr(
                state.work, name
            )[state.work.integration_start[i] : state.work.integration_start[i] + count]
            staged_copy += count * staged[name].itemsize
    caches = work._replace(
        **staged,
        integration_start=cache_start,
        integration_capacity=cache_capacity,
        len_discharge_element=np.zeros(ids.size, dtype=np.int64),
        branch_valid=np.zeros(ids.size, dtype=np.bool_),
    )
    staging_seconds = perf_counter() - start_time
    cache_start = perf_counter()
    rebuild(affected, caches, np.ones(ids.size, dtype=bool))
    cache_seconds = perf_counter() - cache_start
    if np.any(caches.len_discharge_element != 0):
        raise ValueError("Rebuilt integration grid must invalidate branch events")
    storage.validate()
    validate_work(work, nc, ni)
    for name in CACHE_FIELDS:
        if not np.all(np.isfinite(getattr(caches, name))):
            raise ValueError("Nonfinite replacement integration cache")
    for i in ids:
        start, length = starts[i], old.elements["ncoef"][i]
        if any(not np.all(np.isfinite(b[start : start + length])) for b in buffers):
            raise ValueError("Nonfinite replacement coefficients")
    cache_bytes = sum(getattr(caches, name).nbytes for name in CACHE_FIELDS)
    replacement_state = AdaptiveState(storage, work)
    old_payload = state_payload(state)
    replacement_payload = state_payload(replacement_state)
    old_arrays = state_arrays(state)
    transaction_arrays = {
        **state_arrays(replacement_state),
        **{
            id(a): a
            for a in (*caches, affected, requested, integrations, required, nc, ni, ids)
        },
    }
    allocated = sum(
        a.nbytes for key, a in transaction_arrays.items() if key not in old_arrays
    )
    stats = {
        "expanded_elements": int(np.count_nonzero(required)),
        "coefficient_repacks": int(repack),
        "integration_repacks": int(integration_repack),
        "coefficient_bytes_copied": copied,
        "coefficient_repack_bytes": copied,
        "coefficient_relocation_bytes": copied,
        "integration_bytes_copied": integration_copied + staged_copy + cache_bytes,
        "cache_rebuild_bytes": cache_bytes,
        "cache_publication_bytes": cache_bytes,
        "work_bytes_copied": 0,
        "branch_event_bytes_copied": 0,
        "metadata_bytes_copied": affected.nbytes,
        "total_bytes_copied": copied
        + integration_copied
        + staged_copy
        + cache_bytes
        + affected.nbytes,
        "transaction_temporary_payload": allocated,
        "temporary_array_count": sum(
            key not in old_arrays for key in transaction_arrays
        ),
        "peak_simultaneous_numpy_payload": old_payload + allocated,
        "transaction_staging_seconds": staging_seconds,
        "cache_rebuild_seconds": cache_seconds,
        "old_payload": old_payload,
        "replacement_payload": replacement_payload,
    }
    return GrowthTransaction(replacement_state, state, ids, caches, nc, ni), stats


def publish(transaction):
    """Only prevalidated, nonallocating array writes occur at this barrier."""
    if isinstance(transaction, AdaptiveState):
        return transaction
    if isinstance(transaction, LogicalGrowth):
        s, w = transaction.state.storage, transaction.state.work
        publish_logical(
            s.elements,
            s.coef_data,
            s.candidate_coef_data,
            s.old_coef_data,
            s.coef_start,
            w,
            transaction.ids,
            transaction.ncoef,
            transaction.nint,
            transaction.starts,
            transaction.caches,
        )
        return transaction.state
    state, old = transaction.state, transaction.previous
    s, work = state.storage, state.work
    for j, i in enumerate(transaction.ids):
        start = s.coef_start[i]
        previous, length = (
            old.storage.elements["ncoef"][i],
            transaction.requested_ncoef[i],
        )
        for buffer in (
            s.coef_data,
            s.candidate_coef_data,
            s.old_coef_data,
            work.old_coef,
            work.coef_error,
        ):
            buffer[start + previous : start + length] = 0
        a, b = work.integration_start[i], transaction.caches.integration_start[j]
        n = transaction.requested_nint[i]
        for name in CACHE_FIELDS:
            getattr(work, name)[a : a + n] = getattr(transaction.caches, name)[
                b : b + n
            ]
        work.len_discharge_element[i] = 0
        work.branch_valid[i] = False
    for i in transaction.ids:
        s.elements["ncoef"][i] = transaction.requested_ncoef[i]
        s.elements["nint"][i] = transaction.requested_nint[i]
    return state


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
    if not owner.branch_initialized:
        branch_preparation, branch_stats = prepare_branches(
            state.work,
            s.elements,
            fractures,
            np.flatnonzero(~state.work.branch_valid),
            s.elements["nint"],
        )
        work = branch_preparation.work
        publish_direct(
            work,
            branch_preparation.ids,
            branch_preparation.starts,
            branch_preparation.counts,
            branch_preparation.values,
        )
        del branch_preparation
        if work is not state.work:
            owner.state = AdaptiveState(s, work)
            state = owner.state
        owner.branch_batches.append(branch_stats)
        owner.branch_initialized = True
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
            None,
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
        branch_preparation, branch_stats = prepare_branches(
            replacement.state.work,
            s.elements,
            fractures,
            replacement.ids,
            requests[1],
        )
        branch_work = branch_preparation.work
        stats.update(branch_stats)
        stats["transaction_temporary_payload"] += branch_stats[
            "branch_count_temporary_numpy_payload"
        ]
        stats["peak_simultaneous_numpy_payload"] += branch_stats[
            "branch_count_temporary_numpy_payload"
        ]
        stats["temporary_array_count"] += branch_stats["branch_count_temporary_arrays"]
        stats["total_bytes_copied"] += (
            branch_stats["branch_event_bytes_copied"]
            + branch_stats["branch_publication_bytes"]
        )
        if branch_work is not replacement.state.work:
            branch_state = AdaptiveState(replacement.state.storage, branch_work)
        else:
            branch_state = None
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
    if stats is None and owner.transition_callback is None:
        # No publication is needed, but the original second commit remains.
        owner.state.work.set_zero[:] = False
        hs.commit_coefficients(
            s.elements,
            None,
            damping,
            s.coef_data,
            s.coef_start,
            s.candidate_coef_data,
        )
        return
    # Single Python publication after every failure-prone construction step.
    if owner.debug_validation:
        validate_state(owner.state, finite=True)
    if owner.before_publication is not None:
        owner.before_publication(owner.state)
    publication_start = perf_counter()
    owner.state = publish(replacement)
    if stats is not None and branch_state is not None:
        owner.state = branch_state
    if stats is not None:
        stats["publication_seconds"] = perf_counter() - publication_start
    observe("after_growth_publication", requests)
    s = owner.state.storage
    recalc_start = perf_counter()
    if stats is not None:
        publish_direct(
            owner.state.work,
            branch_preparation.ids,
            branch_preparation.starts,
            branch_preparation.counts,
            branch_preparation.values,
        )
        del branch_preparation
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
    owner.state.work.set_zero[:] = False
    commit_start = perf_counter()
    hs.commit_coefficients(
        s.elements,
        None,
        damping,
        s.coef_data,
        s.coef_start,
        s.candidate_coef_data,
    )
    if stats is not None:
        stats["commit_seconds"] = perf_counter() - commit_start
    observe("after_second_commit", requests)


def solve_adaptive(
    owner, fractures, discharge_int, constants, *, iteration_callback=None
):
    """Run the adaptive solver over compact state."""
    owner.state.storage.validate()
    return hs.solve(
        fractures,
        discharge_int,
        constants,
        adaptive_owner=owner,
        iteration_callback=iteration_callback,
    )
