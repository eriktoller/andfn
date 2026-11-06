"""Phase 4C scratch ownership, candidate scaling and storage probes."""

import argparse
import json
from statistics import median
from time import perf_counter

import numba as nb
import numpy as np


@nb.njit
def scratch_element(i, nc, ni, coef0, coef1, integral, phi, psi, dpsi):
    # Integer-valued heterogeneous writes expose cross-element aliasing exactly.
    for k in range(nc):
        coef0[k] = i + k
        coef1[k] = i - k
        integral[k] = coef0[k] + coef1[k]
    for k in range(ni):
        phi[k] = i + k
        psi[k] = i - k
        dpsi[k] = phi[k] - psi[k]
    result = 0.0
    for k in range(nc):
        result += integral[k].real
    for k in range(ni):
        result += phi[k] + psi[k] + dpsi[k]
    return result


@nb.njit(parallel=True)
def scratch_probe(ncoef, nint, storage, output, mode):
    slots = storage.shape[1]
    if mode == 2:
        for chunk in nb.prange(slots):
            begin = chunk * ncoef.size // slots
            end = (chunk + 1) * ncoef.size // slots
            for i in range(begin, end):
                output[i] = scratch_element(
                    i,
                    ncoef[i],
                    nint[i],
                    storage[0, chunk],
                    storage[1, chunk],
                    storage[2, chunk],
                    storage[3, chunk],
                    storage[4, chunk],
                    storage[5, chunk],
                )
    else:
        for i in nb.prange(ncoef.size):
            slot = np.int64(i) if mode == 0 else np.int64(nb.get_thread_id())
            output[i] = scratch_element(
                i,
                ncoef[i],
                nint[i],
                storage[0, slot],
                storage[1, slot],
                storage[2, slot],
                storage[3, slot],
                storage[4, slot],
                storage[5, slot],
            )


def scratch_case(count, threads, repetitions=9):
    ncoef = np.resize(np.array([8, 12, 16, 24, 32, 48, 64], dtype=np.int64), count)
    nint = 2 * ncoef + np.arange(count) % 3
    width = int(nint.max())
    output = np.empty(count, dtype=np.complex128)
    expected = 2 * np.arange(count) * (ncoef + nint) + nint * (nint - 1)
    result = {}
    for mode, name in enumerate(("element", "worker", "chunk")):
        slots = count if mode == 0 else threads
        storage = np.full((6, slots, width + 1), -987654.0, dtype=np.complex128)
        scratch_probe(ncoef, nint, storage, output, mode)
        values = []
        for _ in range(repetitions):
            start = perf_counter()
            scratch_probe(ncoef, nint, storage, output, mode)
            values.append(perf_counter() - start)
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(storage[:, :, -1], -987654.0)
        result[name] = {
            "median_ms": 1000 * median(values),
            "p10_p90_ms": (1000 * np.percentile(values, [10, 90])).tolist(),
            "allocated_bytes": storage.nbytes,
            "isolated": True,
        }
    assert scratch_probe.nopython_signatures
    return result


def main():
    normal = nb.get_num_threads()
    report = {}
    try:
        for threads in sorted({n for n in (1, 2, 4, 8, 16, normal) if n <= normal}):
            nb.set_num_threads(threads)
            report[threads] = {
                count: scratch_case(count, threads) for count in (128, 1024, 10000)
            }
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


def solver_probe(counts=(128, 1024, 10000), groups=None):
    from benchmark_adaptive import prepare

    from andfn.hpc import hpc_adaptive as adaptive
    from andfn.hpc import hpc_solve as hs
    from andfn.hpc import hpc_work as cw
    from andfn.hpc.hpc_branch import prepare_branches

    normal = nb.get_num_threads()
    report = {}
    try:
        for count in counts:
            f, e, legacy, owner = prepare(count // 8)
            if groups is not None:
                work = owner.state.work
                owner.state = adaptive.AdaptiveState(
                    owner.state.storage,
                    work._replace(
                        **cw.resize_scratch(
                            work,
                            work.coef_capacity,
                            work.integration_capacity,
                            groups=groups,
                        )
                    ),
                )
            state = owner.state
            diagnostic = adaptive.adaptive_from_hpc(e, scratch_groups=0).state
            cw.import_work(diagnostic.work, legacy, e)
            for label, active in (("worker", state), ("element", diagnostic)):
                prepared, _ = prepare_branches(
                    active.work,
                    active.storage.elements,
                    f,
                    np.arange(e.size, dtype=np.int64),
                    active.storage.elements["nint"],
                )
                if label == "worker":
                    state = adaptive.AdaptiveState(active.storage, prepared)
                else:
                    diagnostic = adaptive.AdaptiveState(active.storage, prepared)
            cases = {}
            for threads in sorted({n for n in (1, 2, 4, 8, 16, normal) if n <= normal}):
                nb.set_num_threads(threads)
                elapsed = [[], [], []]
                for sample in range(8):
                    for mode in (0, 1, 2) if sample % 2 == 0 else (2, 1, 0):
                        compact = mode != 0
                        active_state = diagnostic if mode == 1 else state
                        s = active_state.storage
                        args = (
                            s.elements if compact else e,
                            f,
                            active_state.work if compact else legacy,
                            1e-12,
                            1,
                            0,
                            0.5,
                            s.coef_data if compact else None,
                            s.coef_start if compact else None,
                            s.candidate_coef_data if compact else None,
                        )
                        started = perf_counter()
                        hs.calculate_candidates(e.size, *args)
                        seconds = perf_counter() - started
                        if sample:
                            elapsed[mode].append(seconds)
                cases[threads] = {
                    name: {
                        "median_ms": median(values) * 1000,
                        "p10_p90_ms": (1000 * np.percentile(values, [10, 90])).tolist(),
                    }
                    for name, values in zip(
                        ("legacy", "element_compact", "compact"), elapsed
                    )
                }
            report[count] = {
                "threads": cases,
                "legacy_work_bytes": legacy.nbytes,
                "compact_payload": cw.state_payload(state),
                "event_counts": state.work.len_discharge_element.tolist(),
                "event_capacity_sum": int(state.work.branch_capacity.sum()),
                "harness_persistent_numpy_payload": sum(
                    a.nbytes
                    for a in {
                        **cw.state_arrays(state),
                        **cw.state_arrays(diagnostic),
                        id(e): e,
                        id(legacy): legacy,
                        id(f): f,
                    }.values()
                ),
            }
            for name in ("legacy", "element_compact", "compact"):
                one = cases[1][name]["median_ms"]
                for values in cases.values():
                    values[name]["speedup"] = one / values[name]["median_ms"]
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


def memory_probe():
    from benchmark_adaptive import prepare

    from andfn.hpc import hpc_adaptive as adaptive
    from andfn.hpc import hpc_solve as hs
    from andfn.hpc import hpc_work as cw
    from andfn.storage import _layout, growth_capacities

    f, e, _legacy, owner = prepare(1)
    adaptive.adaptive_step(owner, f, 1e-12, 1, 0, 1e100, 0, 2, 0.5)
    base = owner.state
    distribution = base.work.len_discharge_element
    duplicate_ids = 0
    duplicate_events = 0
    for i, count in enumerate(distribution):
        record = cw.work_record(base.work, i)
        ids = record["discharge_element"][:count]
        positions = record["element_pos"][:count]
        duplicate_ids += int(count - np.unique(ids).size)
        duplicate_events += int(count - len(set(zip(positions.tolist(), ids.tolist()))))
    report = {
        "event_distribution": {
            "counts": distribution.tolist(),
            "maximum": int(distribution.max()),
            "mean_all_elements": float(distribution.mean()),
            "histogram": {
                str(k): int((distribution == k).sum()) for k in np.unique(distribution)
            },
            "repeated_element_ids": duplicate_ids,
            "repeated_position_id_pairs": duplicate_events,
        },
        "cases": {},
    }
    for count in (128, 1024, 10000, 100000):
        copies = count // 8
        elements = np.tile(base.storage.elements, copies)
        elements["_id"] = np.arange(count)
        cap = np.tile(base.storage.coef_capacity, copies)
        starts, size = _layout(cap)
        work = cw.allocate_work(elements, starts, cap, scratch_groups=7)
        branch_counts = np.tile(distribution, copies)
        branch_capacity = growth_capacities(branch_counts, work.branch_capacity)
        branch_start, branch_size = _layout(branch_capacity)
        work = work._replace(
            **{
                name: np.zeros(branch_size, dtype=np.int64) for name in cw.BRANCH_FIELDS
            },
            branch_start=branch_start,
            branch_capacity=branch_capacity,
        )
        work.len_discharge_element[:] = branch_counts
        work.sign_array[:] = 1
        state = adaptive.AdaptiveState(
            adaptive.AdaptiveStorage(
                elements,
                *(np.zeros(size, dtype=complex) for _ in range(3)),
                starts,
                cap,
            ),
            work,
        )
        adaptive.validate_state(state, finite=True)
        nc, ni = int(elements["ncoef"].sum()), int(elements["nint"].sum())
        slots = work.scratch_layout[1]
        categories = {
            "fixed_metadata_and_layout": (
                elements.nbytes
                + sum(
                    a.nbytes
                    for a in (
                        starts,
                        cap,
                        work.integration_start,
                        work.integration_capacity,
                        work.scratch_layout,
                        work.set_zero,
                    )
                ),
            )
            * 2,
            "coefficient_state": (nc * 64, size * 64),
            "integration_cache": (ni * 64, work.thetas.size * 64),
            "scratch": (
                slots
                * (
                    int(elements["ncoef"].max()) * 48 + int(elements["nint"].max()) * 24
                ),
                sum(getattr(work, name).nbytes for name in cw.SCRATCH_FIELDS),
            ),
            "branch_events": (
                int(branch_counts.sum()) * 24 + count * 25,
                branch_size * 24 + count * 25,
            ),
            "compatibility": (nc * 16 + count * 24, size * 16 + count * 24),
            "quadrature_in_kernel_probe": (0, 0),
        }
        cases = {
            name: {
                "logical_bytes": int(logical),
                "allocated_bytes": int(allocated),
                "utilization": logical / allocated if allocated else 1.0,
            }
            for name, (logical, allocated) in categories.items()
        }
        assert sum(v["allocated_bytes"] for v in cases.values()) == cw.state_payload(
            state
        )
        report["cases"][count] = {
            "categories": cases,
            "compact_payload": cw.state_payload(state),
            "phase4b_payload_formula": 23041 * count,
            "legacy_element_and_work_bytes": count
            * (e.dtype.itemsize + hs.dtype_work.itemsize),
            "legacy_work_bytes": count * hs.dtype_work.itemsize,
            "driver_two_retained_padded_quadratures_bytes": 2
            * count
            * hs.dtype_z_arrays.itemsize,
            "temporary_bytes_during_kernel_timing": 0,
            "owned_numpy_peak_during_kernel_timing": cw.state_payload(state),
            "storage_only": count == 100000,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", action="store_true")
    parser.add_argument("--groups", type=int)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    if args.memory:
        memory_probe()
    elif args.solver:
        solver_probe(groups=args.groups)
    else:
        main()
