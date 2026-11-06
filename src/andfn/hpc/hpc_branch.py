"""Exact branch-event counting and batch capacity preparation at host barriers."""

import numba as nb
import numpy as np

from andfn.storage import _layout, growth_capacities

from . import CACHE, PARALLEL
from . import hpc_geometry_functions as gf
from . import hpc_math_functions as mf
from .hpc_work import BRANCH_FIELDS

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
