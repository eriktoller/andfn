"""Phase 4D branch stages and unchanged adaptive fixtures; emits JSON."""

import argparse
import contextlib
import io
import json
import logging
from time import perf_counter

import numba as nb
import numpy as np
from benchmark_adaptive import prepare, run_case

from andfn.hpc import hpc_branch as branch
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_work as cw
from andfn.storage import _layout, growth_capacities


@nb.njit(parallel=True)
def reference_fill(elements, fractures, work):
    for i in nb.prange(elements.size):
        if elements[i]["_type"] in (1, 4, 5):
            record = cw.work_record(work, i)
            cw.invalidate_branch(record)
            mf.get_dpsi_corr(elements[i], fractures, elements, record)


def summary(values):
    return {
        "median_ms": float(np.median(values)),
        "p10_p90_ms": np.percentile(values, [10, 90]).tolist(),
    }


def branch_case(count):
    fractures, _, _, owner = prepare(count // 8)
    state = owner.state
    elements, work = state.storage.elements, state.work
    ids, nint = np.arange(count, dtype=np.int64), elements["nint"]
    # Warm all variants before collecting alternating samples.
    branch.count_events(elements, fractures, ids, nint)
    reference_fill(elements, fractures, work)
    result, _ = branch.prepare_direct(work, elements, fractures, ids, nint)
    branch.publish_direct(work, ids, result.starts, result.counts, result.values)
    reduced = work.branch_capacity.copy()
    # Sparse forced overflow: one in every 128 rows, always an event producer.
    reduced[::128] = 0
    starts, size = _layout(reduced)
    overflow_work = work._replace(
        branch_start=starts,
        branch_capacity=reduced,
        **dict(zip(cw.BRANCH_FIELDS, branch.allocate_events(size))),
        len_discharge_element=np.zeros(count, dtype=np.int64),
        branch_valid=np.zeros(count, dtype=np.bool_),
    )
    branch.prepare_direct(overflow_work, elements, fractures, ids, nint)
    times = {key: [] for key in ("count", "fill", "direct_total", "overflow_total")}
    stages = {
        "direct_crossings": [],
        "overflow_detection": [],
        "repack": [],
        "reconstruct": [],
    }
    for sample in range(8):
        for mode in (0, 1, 2, 3) if sample % 2 == 0 else (3, 2, 1, 0):
            if mode == 3:
                overflow_work.len_discharge_element[:] = 0
                overflow_work.branch_valid[:] = False
            start = perf_counter()
            if mode == 0:
                branch.count_events(elements, fractures, ids, nint)
            elif mode == 1:
                reference_fill(elements, fractures, work)
            else:
                prepared, stats = branch.prepare_direct(
                    work if mode == 2 else overflow_work, elements, fractures, ids, nint
                )
                branch.publish_direct(
                    prepared.work,
                    ids,
                    prepared.starts,
                    prepared.counts,
                    prepared.values,
                )
                if sample:
                    if mode == 2:
                        stages["direct_crossings"].append(
                            1000 * stats["branch_direct_seconds"]
                        )
                    else:
                        for label, key in (
                            ("overflow_detection", "branch_direct_seconds"),
                            ("repack", "branch_repack_seconds"),
                            ("reconstruct", "branch_reconstruction_seconds"),
                        ):
                            stages[label].append(1000 * stats[key])
                        overflow_stats = stats
            if sample:
                times[tuple(times)[mode]].append(1000 * (perf_counter() - start))
    logical = int(result.counts.sum()) * 24
    capacity = work.sign_array.nbytes * 3
    staging = sum(a.nbytes for a in result.values)
    return {
        "timings": {key: summary(value) for key, value in times.items()},
        "stages": {key: summary(value) for key, value in stages.items()},
        "logical_branch_bytes": logical,
        "allocated_branch_bytes": capacity,
        "utilization": logical / capacity,
        "persistent_branch_metadata_bytes": count * 25,
        "sufficient_staging_bytes": staging,
        "sufficient_overflow_metadata_bytes": result.counts.nbytes,
        "sufficient_branch_peak_bytes": capacity + count * 25 + staging + count * 32,
        "overflow": overflow_stats,
        "overflow_percentage": 100 * overflow_stats["branch_overflow_elements"] / count,
    }


def policy_probe():
    fractures, _, _, owner = prepare(1)
    elements = owner.state.storage.elements.copy()
    ids = np.arange(elements.size, dtype=np.int64)
    snapshots = []
    for _ in range(6):
        snapshots.append(
            branch.count_events(elements, fractures, ids, elements["nint"])
        )
        grow = (elements["_type"] != 2) & (elements["ncoef"] < 40)
        elements["ncoef"][grow] += 5
        elements["nint"][grow] = 2 * elements["ncoef"][grow]
    all_counts = np.concatenate(snapshots)
    policy = {}
    for name in (
        "exact_initial",
        "previous_plus_quantum",
        "previous_25_percent",
        "type_minimum",
    ):
        previous = snapshots[0]
        if name == "exact_initial":
            capacity = previous.copy()
        elif name == "previous_plus_quantum":
            capacity = np.where(np.isin(elements["_type"], (1, 4, 5)), previous + 4, 0)
        elif name == "previous_25_percent":
            capacity = growth_capacities(np.ceil(previous * 1.25).astype(np.int64))
        else:
            capacity = np.where(np.isin(elements["_type"], (1, 4, 5)), 4, 0).astype(
                np.int64
            )
        initial = capacity.copy()
        overflow_events, repacks, copied = 0, 0, 0
        # Exact construction pays a preliminary pass; the other policies do not.
        for current in snapshots[1:]:
            over = current > capacity
            if np.any(over):
                overflow_events += int(over.sum())
                repacks += 1
                copied += int(previous[~over].sum()) * 24
                capacity = growth_capacities(current, capacity)
            previous = current
        policy[name] = {
            "initial_capacity_bytes": int(initial.sum()) * 24,
            "final_capacity_bytes": int(capacity.sum()) * 24,
            "initial_utilization": float(snapshots[0].sum() / initial.sum()),
            "final_utilization": float(previous.sum() / capacity.sum()),
            "overflow_element_events": overflow_events,
            "overflow_fraction_of_growth_attempts": overflow_events
            / (5 * elements.size),
            "repacks": repacks,
            "unaffected_copy_bytes": copied,
        }
    return {
        "counts_by_grid": [a.tolist() for a in snapshots],
        "mean_median_p90_p95_p99_max": [
            float(all_counts.mean()),
            *np.percentile(all_counts, [50, 90, 95, 99]).tolist(),
            int(all_counts.max()),
        ],
        "policies": policy,
        "storage_sizes": {
            n: {
                "type_minimum_capacity_bytes": n * 48,
                "metadata_bytes": n * 25,
                "logical_bytes_seed": n * 18,
                "overflow_fraction_for_tiled_fixture": 0.0,
            }
            for n in (128, 1024, 10000, 100000)
        },
    }


def main(adaptive=False):
    logging.getLogger("andfn").setLevel(logging.ERROR)
    normal = nb.get_num_threads()
    report = {"policy": policy_probe(), "threads": {}}
    try:
        for threads in sorted({n for n in (1, 2, 4, 8, 16, normal) if n <= normal}):
            nb.set_num_threads(threads)
            cases = {n: branch_case(n) for n in (128, 1024, 10000)}
            if adaptive:
                # Imported helper is unchanged, including all seven paired samples.
                with contextlib.redirect_stdout(io.StringIO()):
                    cases["adaptive"] = {
                        "no": run_case(16, 1e100, 2, 0),
                        "spare": run_case(16, -1, 1, 10, spare=True),
                        "forced": run_case(16, -1, 5, 40),
                        "1024": run_case(128, -1, 5, 40),
                        "24_spare": run_case(16, -1, 1, 10, iterations=24, spare=True),
                    }
            report["threads"][threads] = cases
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", action="store_true")
    main(parser.parse_args().adaptive)
