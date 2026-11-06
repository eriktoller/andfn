"""Transactional direct branch construction and retained exact-count reference."""

from dataclasses import dataclass
from time import perf_counter

import numba as nb
import numpy as np

from andfn.storage import _layout, growth_capacities

from . import CACHE, PARALLEL
from . import hpc_geometry_functions as gf
from . import hpc_math_functions as mf
from .hpc_quadrature import quadrature_row
from .hpc_work import BRANCH_FIELDS, BranchRecord, WorkArrays

_count_dtype = np.dtype(
    [
        ("sign_array", np.int64, 1),
        ("discharge_element", np.int64, 1),
        ("element_pos", np.int64, 1),
        ("len_discharge_element", np.int64),
    ]
)


@nb.njit(parallel=PARALLEL, cache=CACHE)
def count_events(elements, fractures, ids, nint):
    counts = np.zeros(ids.size, dtype=np.int64)
    for j in nb.prange(ids.size):
        i = ids[j]
        e = elements[i]
        if e["_type"] not in (1, 4, 5):
            continue
        n = nint[i]
        theta = np.empty(n)
        exp_p = np.empty(n, dtype=np.complex128)
        mf.calc_thetas(n, e["_type"], theta)
        mf.fill_exp_array(n, theta, exp_p, 1)
        if e["_type"] in (1, 4):
            points = gf.map_chi_to_z_circle(exp_p, e["radius"], e["center"])
        else:
            points = gf.map_chi_to_z_line(exp_p, e["endpoints0"])
        record = np.empty(1, dtype=_count_dtype)
        mf.find_branch_cuts(e, points, fractures, elements, record[0], n)
        counts[j] = record[0]["len_discharge_element"]
    return counts


def prepare_branches(work, elements, fractures, ids, nint):
    """Return private replacement buffers only when exact counts exceed capacity."""
    counts = count_events(elements, fractures, ids, nint)
    stats = {
        "branch_count_temporary_numpy_payload": int(ids.size) * 17,
        "branch_count_temporary_arrays": 3,
    }
    if not np.any(counts > work.branch_capacity[ids]):
        return work, {**stats, "branch_repacks": 0, "branch_event_bytes_copied": 0}
    capacities = work.branch_capacity.copy()
    capacities[ids] = growth_capacities(counts, capacities[ids])
    starts, size = _layout(capacities)
    values = {name: np.zeros(size, dtype=np.int64) for name in BRANCH_FIELDS}
    copied = 0
    for i, count in enumerate(work.len_discharge_element):
        a, b = work.branch_start[i], starts[i]
        for name in BRANCH_FIELDS:
            values[name][b : b + count] = getattr(work, name)[a : a + count]
            copied += int(count) * 8
    replacement = work._replace(
        **values, branch_start=starts, branch_capacity=capacities
    )
    return replacement, {
        **stats,
        "branch_repacks": 1,
        "branch_event_bytes_copied": copied,
        "branch_replacement_bytes": sum(a.nbytes for a in values.values())
        + starts.nbytes
        + capacities.nbytes,
    }


@dataclass(frozen=True)
class BranchPreparation:
    work: WorkArrays
    ids: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    values: tuple


@nb.njit(parallel=PARALLEL, cache=CACHE)
def construct_events(
    elements,
    fractures,
    ids,
    selected,
    nint,
    starts,
    capacities,
    counts,
    signs,
    event_ids,
    positions,
    quadrature=None,
):
    """Count beyond capacity, but write only private, exclusive segments."""
    for k in nb.prange(selected.size):
        j = selected[k]
        i = ids[j]
        e = elements[i]
        counts[j] = 0
        if e["_type"] not in (1, 4, 5):
            continue
        n = nint[i]
        if quadrature is not None:
            points = quadrature_row(quadrature, 0, i, n)
        else:
            theta = np.empty(n)
            exp_p = np.empty(n, dtype=np.complex128)
            mf.calc_thetas(n, e["_type"], theta)
            mf.fill_exp_array(n, theta, exp_p, 1)
            if e["_type"] in (1, 4):
                points = gf.map_chi_to_z_circle(exp_p, e["radius"], e["center"])
            else:
                points = gf.map_chi_to_z_line(exp_p, e["endpoints0"])
        a, b = starts[j], starts[j] + capacities[j]
        record = BranchRecord(
            signs[a:b], event_ids[a:b], positions[a:b], counts[j : j + 1]
        )
        mf.find_branch_cuts(e, points, fractures, elements, record, n)


@nb.njit(cache=CACHE)
def validate_events(elements, ids, nint, starts, capacities, counts, values):
    for j in range(ids.size):
        count = counts[j]
        if count < 0 or count > capacities[j]:
            raise ValueError("Invalid prepared branch length")
        for k in range(count):
            a = starts[j] + k
            sign, element_id, position = values[0][a], values[1][a], values[2][a]
            if (
                (sign != -1 and sign != 1)
                or element_id < 0
                or element_id >= elements.size
            ):
                raise ValueError("Invalid prepared branch sign or element ID")
            if (
                not np.isfinite(position)
                or position < 0
                or position >= nint[ids[j]] - 1
            ):
                raise ValueError("Invalid prepared branch position")


@nb.njit(cache=CACHE)
def copy_segments(ids, source_starts, destination_starts, counts, source, destination):
    for j in range(ids.size):
        i = ids[j]
        for k in range(counts[j]):
            a, b = source_starts[j] + k, destination_starts[i] + k
            for field in range(3):
                destination[field][b] = source[field][a]


def allocate_events(size):
    return tuple(np.empty(size, dtype=np.int64) for _ in BRANCH_FIELDS)


def prepare_direct(work, elements, fractures, ids, nint, *, quadrature=None):
    """Prepare affected topology without changing any published buffer."""
    if not isinstance(ids, np.ndarray) or ids.dtype != np.int64 or ids.ndim != 1:
        raise ValueError("Branch IDs must be a one-dimensional int64 array")
    if np.any(ids < 0) or np.any(ids >= elements.size) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Branch IDs must be increasing, unique and in range")
    if (
        not isinstance(nint, np.ndarray)
        or nint.dtype != np.int64
        or nint.shape != (elements.size,)
        or np.any(nint[ids] < 1)
    ):
        raise ValueError("Invalid branch integration lengths")
    destinations = tuple(getattr(work, name) for name in BRANCH_FIELDS) + (
        work.len_discharge_element,
        work.branch_valid,
    )
    if any(not a.flags.writeable for a in destinations):
        raise ValueError("Branch publication buffers must be writable")
    capacities = work.branch_capacity[ids]
    starts, size = _layout(capacities)
    values = allocate_events(size)
    counts = np.zeros(ids.size, dtype=np.int64)
    selected = np.arange(ids.size, dtype=np.int64)
    extra = () if quadrature is None else (quadrature,)
    started = perf_counter()
    construct_events(
        elements,
        fractures,
        ids,
        selected,
        nint,
        starts,
        capacities,
        counts,
        *values,
        *extra,
    )
    direct_seconds = perf_counter() - started
    overflow = np.flatnonzero(counts > capacities)
    stats = {
        "branch_repacks": 0,
        "branch_event_bytes_copied": 0,
        "branch_direct_seconds": direct_seconds,
        "branch_repack_seconds": 0.0,
        "branch_reconstruction_seconds": 0.0,
        "branch_overflow_elements": overflow.size,
        "branch_staging_bytes": sum(a.nbytes for a in values),
        "branch_overflow_metadata_bytes": counts.nbytes + overflow.nbytes,
    }
    temporary = sum(
        a.nbytes for a in (capacities, starts, counts, selected, overflow)
    ) + sum(a.nbytes for a in values)
    arrays = 8
    replacement = work
    if overflow.size:
        started = perf_counter()
        new_capacities = capacities.copy()
        new_capacities[overflow] = growth_capacities(
            counts[overflow], capacities[overflow]
        )
        new_starts, new_size = _layout(new_capacities)
        new_values = allocate_events(new_size)
        fitting = np.flatnonzero(counts <= capacities)
        copy_segments(
            fitting, starts[fitting], new_starts, counts[fitting], values, new_values
        )
        temporary += (
            new_capacities.nbytes
            + new_starts.nbytes
            + sum(a.nbytes for a in new_values)
            + fitting.nbytes
        )
        values, starts, capacities = new_values, new_starts, new_capacities
        global_capacities = work.branch_capacity.copy()
        global_capacities[ids] = capacities
        global_starts, global_size = _layout(global_capacities)
        buffers = allocate_events(global_size)
        # Preserve only unaffected live prefixes. Affected results are still private.
        keep = np.ones(elements.size, dtype=np.bool_)
        keep[ids] = False
        unaffected = np.flatnonzero(keep)
        copy_segments(
            unaffected,
            work.branch_start[unaffected],
            global_starts,
            work.len_discharge_element[unaffected],
            tuple(getattr(work, name) for name in BRANCH_FIELDS),
            buffers,
        )
        replacement = work._replace(
            **dict(zip(BRANCH_FIELDS, buffers)),
            branch_start=global_starts,
            branch_capacity=global_capacities,
        )
        replacement_bytes = (
            sum(a.nbytes for a in buffers)
            + global_starts.nbytes
            + global_capacities.nbytes
        )
        temporary += replacement_bytes + keep.nbytes + unaffected.nbytes
        arrays += 13
        stats.update(
            branch_repacks=1,
            branch_replacement_bytes=replacement_bytes,
            branch_event_bytes_copied=(
                int(work.len_discharge_element[unaffected].sum())
                + int(counts[fitting].sum())
            )
            * 24,
            branch_repack_seconds=perf_counter() - started,
        )
        started = perf_counter()
        construct_events(
            elements,
            fractures,
            ids,
            overflow,
            nint,
            starts,
            capacities,
            counts,
            *values,
            *extra,
        )
        stats["branch_reconstruction_seconds"] = perf_counter() - started
    validate_events(elements, ids, nint, starts, capacities, counts, values)
    stats.update(
        branch_count_temporary_numpy_payload=temporary,
        branch_count_temporary_arrays=arrays,
        branch_publication_bytes=int(counts.sum()) * 24,
    )
    return BranchPreparation(replacement, ids, starts, counts, values), stats


@nb.njit(cache=CACHE)
def publish_direct(work, ids, starts, counts, values):
    """Nonallocating writes after all construction and validation succeeds."""
    copy_segments(
        ids,
        starts,
        work.branch_start,
        counts,
        values,
        (work.sign_array, work.discharge_element, work.element_pos),
    )
    for j in range(ids.size):
        i = ids[j]
        work.len_discharge_element[i] = counts[j]
        work.branch_valid[i] = True
