"""Warm adaptive comparison: uv run python tests\\benchmark_adaptive.py.

Use --peak-rss to run an isolated forced-repack process and report the OS
process-lifetime peak working set/RSS, separately from measured NumPy payload.
"""

import argparse
import cProfile
import ctypes
import io
import json
import logging
import os
import pstats
import subprocess
import sys
from statistics import median
from time import perf_counter

import numba as nb
import numpy as np
from benchmark_fixed_order import differences, fixture
from scipy.sparse.linalg import splu
from test_hpc_adaptive import run_adaptive
from test_hpc_coefficients import legacy_logical, logical, work_for

from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_solve as hs
from andfn.hpc import hpc_work as cw
from andfn.storage import connectivity_from_hpc


def prepare(copies, spare=False):
    fractures, elements = fixture(copies)
    if spare:
        # 12 is already covered by the import's quantum-4 capacity of 12.
        elements["ncoef"][elements["_type"] != 2] = 9
        elements["nint"][elements["_type"] != 2] = 18
    work = work_for(elements)
    owner = adaptive.adaptive_from_hpc(elements)
    cw.import_work(owner.state.work, work, elements)
    return fractures, elements, work, owner


def run_case(
    copies,
    ratio,
    increase,
    maximum,
    iterations=6,
    spare=False,
    *,
    compact_quadrature=False,
):
    _f0, e0, _w0, owner0 = prepare(copies, spare)
    initial_capacity = int(owner0.state.storage.coef_capacity.sum())
    elapsed = [[], []]
    iteration_times = [[[] for _ in range(iterations)] for _ in range(2)]
    samples = None
    for sample in range(7):
        for compact in (False, True) if sample % 2 == 0 else (True, False):
            fractures, elements, work, owner = prepare(copies, spare)
            active = owner.state.storage.elements if compact else elements
            connectivity = connectivity_from_hpc(fractures, elements.size)
            indices = hs.get_discharge_indices(active)
            z = (
                hs.allocate_quadrature(indices.size, 8)
                if compact and compact_quadrature
                else np.zeros(elements.size, dtype=hs.dtype_z_arrays)
            )
            hs.get_z_int_array(z, active, 8, indices)
            lu = splu(
                hs.build_discharge_matrix_csr(
                    fractures,
                    active,
                    connectivity.ids,
                    connectivity.indptr,
                    indices,
                    8,
                    z,
                )
            )
            size = fractures.size + indices.size
            rhs, discharge, old_discharge = (np.zeros(size) for _ in range(3))
            started = perf_counter()
            for nit in range(1, iterations + 1):
                iteration_start = perf_counter()
                s = owner.state.storage if compact else None
                active = s.elements if compact else elements
                hs.solve_discharge_matrix(
                    fractures,
                    active,
                    active,
                    8,
                    rhs,
                    discharge,
                    old_discharge,
                    z,
                    lu,
                    connectivity.ids,
                    connectivity.indptr,
                    indices,
                    s.coef_data if compact else None,
                    s.coef_start if compact else None,
                )
                if compact:
                    adaptive.adaptive_step(
                        owner, fractures, 1e-12, nit, 0, ratio, maximum, increase, 0.5
                    )
                else:
                    hs.element_solver2(
                        elements.size,
                        elements,
                        fractures,
                        work,
                        1e-12,
                        nit,
                        0,
                        ratio,
                        maximum,
                        increase,
                        0.5,
                    )
                hs.get_error(owner.state.storage.elements if compact else elements)
                iteration_times[int(compact)][nit - 1].append(
                    perf_counter() - iteration_start
                )
            elapsed[int(compact)].append(perf_counter() - started)
            if compact:
                samples = owner
            else:
                last_legacy = elements
    state = samples.state
    s = state.storage
    a, b = map(median, elapsed)
    batches = samples.batches
    metrics = {
        "elements": e0.size,
        "legacy_ms": 1000 * a,
        "compact_ms": 1000 * b,
        "runtime_ratio": b / a,
        "legacy_p10_p90_ms": (1000 * np.percentile(elapsed[0], [10, 90])).tolist(),
        "compact_p10_p90_ms": (1000 * np.percentile(elapsed[1], [10, 90])).tolist(),
        "legacy_iteration_ms": [1000 * median(x) for x in iteration_times[0]],
        "compact_iteration_ms": [1000 * median(x) for x in iteration_times[1]],
        "growth_batches": len(batches),
        "expanded_element_events": sum(x["expanded_elements"] for x in batches),
        "initial_capacity_sum": initial_capacity,
        "final_capacity_sum": int(s.coef_capacity.sum()),
        "maximum_logical_ncoef": int(s.elements["ncoef"].max()),
        "capacity_utilization": float(
            s.elements["ncoef"].sum() / s.coef_capacity.sum()
        ),
        "coefficient_repacks": sum(x["coefficient_repacks"] for x in batches),
        "integration_repacks": sum(x["integration_repacks"] for x in batches),
        "coefficient_bytes_copied": sum(x["coefficient_bytes_copied"] for x in batches),
        "integration_bytes_copied": sum(x["integration_bytes_copied"] for x in batches),
        "work_bytes_copied": sum(x["work_bytes_copied"] for x in batches),
        "metadata_bytes_copied": sum(x["metadata_bytes_copied"] for x in batches),
        "total_bytes_copied": sum(x["total_bytes_copied"] for x in batches),
        "transaction_ms": 1000 * sum(x["transaction_seconds"] for x in batches),
        "publication_ms": 1000 * sum(x["publication_seconds"] for x in batches),
        "commit_ms_growth_batches": 1000 * sum(x["commit_seconds"] for x in batches),
        "coefficient_repack_bytes": sum(x["coefficient_repack_bytes"] for x in batches),
        "coefficient_relocation_bytes": sum(
            x["coefficient_relocation_bytes"] for x in batches
        ),
        "cache_rebuild_bytes": sum(x["cache_rebuild_bytes"] for x in batches),
        "branch_event_bytes_copied": sum(
            x["branch_event_bytes_copied"] for x in batches
        ),
        "staging_ms": 1000 * sum(x["transaction_staging_seconds"] for x in batches),
        "cache_rebuild_ms": 1000 * sum(x["cache_rebuild_seconds"] for x in batches),
        "recalculation_ms": 1000 * sum(x["recalculation_seconds"] for x in batches),
        "detection_ms_growth_batches": 1000
        * sum(x["detection_seconds"] for x in batches),
        "detection_ms_all_assessments": 1000
        * sum(x["detection_seconds"] for x in samples.assessments),
        "maximum_old_new_payload": max(
            (x["peak_simultaneous_numpy_payload"] for x in batches), default=0
        ),
        "peak_simultaneous_numpy_payload": max(
            (x["peak_simultaneous_numpy_payload"] for x in batches),
            default=cw.state_payload(state),
        ),
        "transaction_temporary_payload": max(
            (x["transaction_temporary_payload"] for x in batches), default=0
        ),
        "transaction_live_payload_arrays_peak": max(
            (x["temporary_array_count"] for x in batches), default=0
        ),
        "transaction_live_payload_arrays_total": sum(
            x["temporary_array_count"] for x in batches
        ),
        "complete_state_validation_passes": sum(
            x["complete_state_scans"] for x in batches
        ),
        "payload": {
            "legacy_current": e0["coef"].nbytes,
            "legacy_old": e0["old_coef"].nbytes,
            "compact_current": s.coef_data.nbytes,
            "compact_candidate": s.candidate_coef_data.nbytes,
            "compact_compatibility_old": s.old_coef_data.nbytes,
            "retained_integration": sum(
                getattr(state.work, name).nbytes for name in cw.CACHE_FIELDS
            ),
            "retained_work": cw.payload(state.work)
            - sum(getattr(state.work, name).nbytes for name in cw.CACHE_FIELDS),
            "legacy_compatibility": e0.nbytes,
            "total_compact_state_and_work": cw.state_payload(state),
        },
        "final_coefficient_difference": differences(
            legacy_logical(last_legacy), logical(s, s.coef_data)
        ),
    }
    metrics["remaining_solver_work_ms"] = metrics["compact_ms"] - sum(
        metrics[k]
        for k in (
            "transaction_ms",
            "publication_ms",
            "commit_ms_growth_batches",
            "recalculation_ms",
            "detection_ms_all_assessments",
        )
    )
    return metrics


def peak_rss():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def forced_peak():
    # Warm cache construction before recording the baseline lifetime high-water.
    _, _, _, warm = prepare(1)
    e = warm.state.storage.elements
    nc, ni = e["ncoef"].copy(), e["nint"].copy()
    mask = e["_type"] != 2
    nc[mask] += 5
    ni[mask] = 2 * nc[mask]
    adaptive.build_replacement(warm.state, nc, ni, mask, 100, 5)
    _, _, _, owner = prepare(512)
    baseline = peak_rss()
    e = owner.state.storage.elements
    nc, ni = e["ncoef"].copy(), e["nint"].copy()
    mask = e["_type"] != 2
    nc[mask] += 5
    ni[mask] = 2 * nc[mask]
    transaction, stats = adaptive.build_replacement(owner.state, nc, ni, mask, 100, 5)
    replacement = adaptive.publish(transaction)
    # Keep both states live through the OS measurement.
    print(
        json.dumps(
            {
                "elements": e.size,
                "baseline_lifetime_peak_rss": baseline,
                "after_repack_lifetime_peak_rss": peak_rss(),
                "old_new_numpy_payload": stats["peak_simultaneous_numpy_payload"],
                "replacement_numpy_payload": cw.state_payload(replacement),
            }
        )
    )


def profile_residual():
    """Separate diagnostic runs; do not alter the comparable timed cases."""
    f, _e, _, owner = prepare(16, True)
    adaptive.adaptive_step(owner, f, 1e-12, 1, 0, -1.0, 10, 1, 0.5)
    adaptive.adaptive_step(owner, f, 1e-12, 2, 0, -1.0, 10, 1, 0.5)
    steady = cProfile.Profile()
    steady.enable()
    for nit in range(3, 203):
        adaptive.adaptive_step(owner, f, 1e-12, nit, 0, -1.0, 10, 1, 0.5)
    steady.disable()
    _, _, _, original = prepare(16, True)
    s = original.state.storage
    nc, ni = s.elements["ncoef"].copy(), s.elements["nint"].copy()
    required = s.elements["_type"] != 2
    nc[required] += 1
    ni[required] = 2 * nc[required]
    construction = cProfile.Profile()
    construction.enable()
    for _ in range(100):
        adaptive.build_replacement(original.state, nc, ni, required, 10, 1)
    construction.disable()
    traces = {}
    for name, profile in (
        ("steady_adaptive_step", steady),
        ("spare_construction", construction),
    ):
        text = io.StringIO()
        pstats.Stats(profile, stream=text).strip_dirs().sort_stats(
            "tottime"
        ).print_stats(16)
        traces[name] = text.getvalue()
    f, _, _, observed = prepare(16, True)
    transitions = {}
    commits = []

    def observe(label, state, requests):
        now = perf_counter()
        if label == "after_first_commit":
            commits.append(now - transitions["after_first_candidate"])
        elif label == "after_second_commit":
            commits.append(now - transitions["after_adaptive_candidate"])
        transitions[label] = now

    observed.transition_callback = observe
    for nit in range(1, 7):
        adaptive.adaptive_step(observed, f, 1e-12, nit, 0, -1.0, 10, 1, 0.5)
    f, _, _, observed = prepare(16, True)
    transitions.clear()
    commits.clear()
    observed.transition_callback = observe
    start = perf_counter()
    for nit in range(1, 7):
        adaptive.adaptive_step(observed, f, 1e-12, nit, 0, -1.0, 10, 1, 0.5)
    total = perf_counter() - start
    stages = {
        "detection": sum(x["detection_seconds"] for x in observed.assessments),
        "transaction_construction": sum(
            x["transaction_seconds"] - x["cache_rebuild_seconds"]
            for x in observed.batches
        ),
        "publication": sum(x["publication_seconds"] for x in observed.batches),
        "cache_rebuild": sum(x["cache_rebuild_seconds"] for x in observed.batches),
        "candidate_recalculation": sum(
            x["recalculation_seconds"] for x in observed.batches
        ),
        "commit": sum(commits),
    }
    stages["remaining_element_work"] = total - sum(stages.values())
    return {
        "24_iteration_spare_capacity": run_case(
            16, -1, 1, 10, iterations=24, spare=True
        ),
        "six_element_steps_instrumented_ms": {k: 1000 * v for k, v in stages.items()},
        "construction_substages_ms": {
            key: 1000 * sum(x.get(key, 0) for x in observed.batches)
            for key in (
                "request_extraction_seconds",
                "validation_seconds",
                "finite_validation_seconds",
                "cache_rebuild_seconds",
            )
        },
        "profiles": traces,
    }


def main(include_peak=False, include_profile=False):
    logging.getLogger("andfn").setLevel(logging.ERROR)
    normal = nb.get_num_threads()
    nb.set_num_threads(1)
    start = perf_counter()
    # Warm both paths with an actual growth batch before timing.
    f, e, w, owner = prepare(1)
    hs.element_solver2(e.size, e, f, w, 1e-12, 2, 0, -1.0, 40, 2, 0.5)
    adaptive.adaptive_step(owner, f, 1e-12, 2, 0, -1.0, 40, 2, 0.5)
    f, _, _, owner = prepare(1, spare=True)
    adaptive.adaptive_step(owner, f, 1e-12, 2, 0, -1.0, 10, 1, 0.5)
    run_case(1, -1, 2, 40, iterations=2)
    compilation = perf_counter() - start
    report = {
        "compile_setup_seconds": compilation,
        "normal_threads": normal,
        "cases": {},
    }
    try:
        for threads in sorted({1, 2, normal}):
            nb.set_num_threads(threads)
            report["cases"][threads] = {
                "128_no_growth": run_case(16, 1e100, 2, 0),
                "128_forced_repack": run_case(16, -1, 5, 40),
                "128_spare_capacity": run_case(16, -1, 1, 10, spare=True),
                "1024_multiple_expansions": run_case(128, -1, 5, 40),
                "24_iteration_spare_capacity": run_case(
                    16, -1, 1, 10, iterations=24, spare=True
                ),
            }
        nb.set_num_threads(1)
        a, _ = run_adaptive(False)
        b, owner = run_adaptive(True)
        report["iteration_differences"] = [
            {
                "iteration": left[0],
                "trajectory_identical": bool(
                    np.array_equal(left[2], right[2])
                    and np.array_equal(left[3], right[3])
                ),
                "fields": [differences(x, y) for x, y in zip(left[4:], right[4:])],
            }
            for left, right in zip(a, b)
        ]
        if include_peak:
            result = subprocess.run(
                [sys.executable, __file__, "--peak-child"],
                check=True,
                capture_output=True,
                text=True,
            )
            report["process_peak"] = json.loads(result.stdout)
        if include_profile:
            nb.set_num_threads(normal)
            report["residual_profile"] = profile_residual()
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-rss", action="store_true")
    parser.add_argument("--peak-child", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.peak_child:
        forced_peak()
    else:
        main(args.peak_rss, args.profile)
