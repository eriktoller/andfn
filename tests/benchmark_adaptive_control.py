"""Warm transaction profiles and publication/dispatch experiments."""

import cProfile
import io
import json
import pstats
import sys
from collections import Counter
from statistics import median
from time import perf_counter

import numba as nb
import numpy as np
from benchmark_adaptive import prepare
from test_compact_work import fixture as transaction_fixture

from andfn.hpc import hpc_adaptive as adaptive


@nb.njit
def serial_probe(a):
    a[0] += 1


@nb.njit(parallel=True)
def parallel_probe(a):
    for i in nb.prange(a.size):
        a[i] += 1


@nb.njit
def invalidate_events(counts, ids):
    for i in ids:
        counts[i] = 0


def requests(owner):
    e = owner.state.storage.elements
    nc, ni = e["ncoef"].copy(), e["nint"].copy()
    required = e["_type"] != 2
    nc[required] += 1
    ni[required] = 2 * nc[required]
    return nc, ni, required


def profile(operation):
    for _ in range(3):
        operation()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        operation()
    profiler.disable()
    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative")
    stats.print_stats(45)
    return output.getvalue()


def call_counts(operation):
    counts = Counter()

    def trace(frame, event, arg):
        if event == "c_call":
            name = getattr(arg, "__qualname__", str(arg))
            counts[name] += 1
        elif event == "call":
            filename = frame.f_code.co_filename
            if "\\andfn\\" in filename or "fromnumeric.py" in filename:
                counts[frame.f_code.co_name] += 1

    sys.setprofile(trace)
    try:
        for _ in range(100):
            operation()
    finally:
        sys.setprofile(None)
    return dict(counts)


def dispatcher_counts(operation):
    names = (
        "validate_affected",
        "patch_layout",
        "prepare_cache_patch",
        "validate_patch",
    )
    originals = {name: getattr(adaptive, name) for name in names}
    counts = Counter()

    def wrapper(name, dispatcher):
        def counted(*args, **kwargs):
            counts[name] += 1
            return dispatcher(*args, **kwargs)

        return counted

    try:
        for name, dispatcher in originals.items():
            setattr(adaptive, name, wrapper(name, dispatcher))
        for _ in range(100):
            operation()
    finally:
        for name, dispatcher in originals.items():
            setattr(adaptive, name, dispatcher)
    return dict(counts)


def elapsed(operation, samples=101):
    values = []
    for _ in range(samples):
        start = perf_counter()
        operation()
        values.append(perf_counter() - start)
    return {
        "median_us": median(values) * 1e6,
        "p10_us": float(np.percentile(values, 10)) * 1e6,
        "p90_us": float(np.percentile(values, 90)) * 1e6,
    }


def scenario_accounts():
    cases = {}
    for name, coefficient, integration, ids in (
        ("no_growth", False, False, ()),
        ("within_capacity", False, False, (0,)),
        ("coefficient_only", True, False, (0,)),
        ("integration_only", False, True, (0,)),
        ("both", True, True, (0,)),
        ("several_both", True, True, (0, 1, 3, 4, 5, 6)),
    ):
        _, owner = transaction_fixture(coefficient, integration)
        state = owner.state
        nc = state.storage.elements["ncoef"].copy()
        ni = state.storage.elements["nint"].copy()
        mask = np.zeros(nc.size, dtype=bool)
        mask[list(ids)] = True
        nc[mask] += 2
        ni[mask] = 2 * nc[mask]
        tx, stats = adaptive.build_replacement(state, nc, ni, mask, 20, 2)
        cases[name] = stats
        adaptive.validate_state(adaptive.publish(tx), finite=True)
    return cases


def main():
    report = {}
    normal = nb.get_num_threads()
    parallel_publication = nb.njit(parallel=True)(adaptive.publish_logical.py_func)
    try:
        for threads in sorted({1, 2, normal}):
            nb.set_num_threads(threads)
            _, _, _, owner = prepare(16, True)
            req = requests(owner)

            def operation(owner=owner, req=req):
                return adaptive.build_replacement(owner.state, *req, 10, 1)

            tx, stats = operation()
            adaptive.publish(tx)  # Compile before any publication measurements.
            owner.state.storage.elements["ncoef"][req[2]] -= 1
            owner.state.storage.elements["nint"][req[2]] -= 2
            tx, stats = operation()

            # Use the same Phase 4A builder as an instrumented control:
            # all ownership reconstruction/finite-prefix checks remain intact.
            def old_operation(owner=owner, req=req):
                return adaptive.build_repack(owner.state, *req, 10, 1)

            a = np.zeros(28)
            serial_probe(a)
            parallel_probe(a)
            arguments = (
                owner.state.storage.elements,
                owner.state.storage.coef_data,
                owner.state.storage.candidate_coef_data,
                owner.state.storage.old_coef_data,
                owner.state.storage.coef_start,
                owner.state.work,
                tx.ids,
                tx.ncoef,
                tx.nint,
                tx.starts,
                tx.caches,
            )
            parallel_publication(*arguments)
            invalidate_events(owner.state.work.len_discharge_element, tx.ids)
            owner.state.storage.elements["ncoef"][req[2]] -= 1
            owner.state.storage.elements["nint"][req[2]] -= 2
            result = {
                "before_100_profile": profile(old_operation),
                "after_100_profile": profile(operation),
                "before_100_call_counts": call_counts(old_operation),
                "after_100_call_counts": call_counts(operation),
                "after_100_dispatcher_calls": dispatcher_counts(operation),
                "after_stats": stats,
                "construction": elapsed(operation),
                "finite_check": elapsed(
                    lambda tx=tx: adaptive.validate_patch(tx.caches)
                ),
                "serial_dispatch": elapsed(lambda a=a: serial_probe(a)),
                "parallel_dispatch_launch": elapsed(lambda a=a: parallel_probe(a)),
                "publication": elapsed(
                    lambda arguments=arguments: adaptive.publish_logical(*arguments)
                ),
                "parallel_publication": elapsed(
                    lambda arguments=arguments: parallel_publication(*arguments)
                ),
                "event_invalidation_dispatch_included": elapsed(
                    lambda owner=owner, tx=tx: invalidate_events(
                        owner.state.work.len_discharge_element, tx.ids
                    )
                ),
            }
            report[threads] = result
    finally:
        nb.set_num_threads(normal)
    report["transaction_scenarios"] = scenario_accounts()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
