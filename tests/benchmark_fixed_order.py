"""Run with uv run python tests\\benchmark_fixed_order.py [--copies 16].

Reports warm medians separately from first-call compilation/setup. Timed solver
iterations include discharge RHS/LU solve, candidates, and the staged commits.
Fixtures/reset/conversion/factorization and diagnostics are outside that timing.
"""

import argparse
import json
import logging
import time
from statistics import median

import numba as nb
import numpy as np
from scipy.sparse.linalg import splu
from test_hpc_coefficients import network, run_solver, work_for

from andfn.hpc import hpc_bounding_circle as bc
from andfn.hpc import hpc_const_head_line as ch
from andfn.hpc import hpc_fracture as hf
from andfn.hpc import hpc_imp_object as imp
from andfn.hpc import hpc_intersection as intersection
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_solve as hs
from andfn.hpc.hpc_coefficients import coefficient_view
from andfn.storage import connectivity_from_hpc, fixed_order_from_hpc


@nb.njit
def scalar_batch(elements, data=None, starts=None):
    value = 0j
    for _ in range(100):
        for e in elements:
            value += mf.taylor_series(0.3 + 0.1j, coefficient_view(e, data, starts))
    return value


@nb.njit
def vector_batch(elements, points, output, data=None, starts=None):
    for e in elements:
        output[:] = 0
        if e["_type"] == 0:
            intersection.calc_omega_array(e, output, points, e["frac0"], data, starts)
        elif e["_type"] == 1:
            bc.calc_omega_array(e, output, points, data, starts)
        elif e["_type"] == 3:
            ch.calc_omega_array(e, output, points, data, starts)
        elif e["_type"] == 4:
            imp.calc_omega_circle_array(e, output, points, data, starts)
        elif e["_type"] == 5:
            imp.calc_omega_line_array(e, output, points, data, starts)


def fixture(copies):
    f0, e0 = network()
    e0["ncoef"] = [8, 12, 0, 16, 20, 24, 12, 0]
    e0["nint"] = np.maximum(2 * e0["ncoef"], 2)
    fractures = np.tile(f0, copies)
    elements = np.tile(e0, copies)
    for i in range(copies):
        f = fractures[i * 2 : (i + 1) * 2]
        e = elements[i * 8 : (i + 1) * 8]
        f["_id"] += i * 2
        for row in f:
            row["elements"][: row["nelements"]] += i * 8
        e["_id"] += i * 8
        e["frac0"] += i * 2
        e["frac1"][e["frac1"] >= 0] += i * 2
    return fractures, elements


def prepare(fractures, elements, compact):
    f, legacy = fractures.copy(), elements.copy()
    work = work_for(legacy)
    storage = fixed_order_from_hpc(legacy) if compact else None
    e = storage.elements if compact else legacy
    args = (
        (storage.coef_data, storage.coef_start, storage.candidate_coef_data)
        if compact
        else (None, None, None)
    )
    c = connectivity_from_hpc(f, e.size)
    indices = hs.get_discharge_indices(e)
    z = np.zeros(e.size, dtype=hs.dtype_z_arrays)
    hs.get_z_int_array(z, e, 8, indices)
    lu = splu(hs.build_discharge_matrix_csr(f, e, c.ids, c.indptr, indices, 8, z))
    size = indices.size + f.size
    q, old, rhs = (np.zeros(size) for _ in range(3))

    def iterate(count):
        for nit in range(1, count + 1):
            hs.solve_discharge_matrix(
                f, e, e, 8, rhs, q, old, z, lu, c.ids, c.indptr, indices, *args[:2]
            )
            hs.fixed_order_step(e, f, work, 1e-12, nit, 0, 0.5, *args)
            hs.get_error(e)

    initial_e, initial_f, initial_work = e.copy(), f.copy(), work.copy()
    initial_coef = storage.coef_data.copy() if compact else None

    def reset():
        e[:] = initial_e
        f[:] = initial_f
        work[:] = initial_work
        q[:] = 0
        old[:] = 0
        rhs[:] = 0
        if compact:
            storage.coef_data[:] = initial_coef
            storage.candidate_coef_data[:] = 0

    points = np.linspace(0.7, 1.3, 256).astype(complex) + 0.7j
    output = np.zeros(points.size, dtype=complex)

    def vector():
        vector_batch(e, points, output, *args[:2])

    return iterate, reset, lambda: scalar_batch(e, *args[:2]), vector, storage, work


def measure_pair(legacy, compact, count=None, repeats=21):
    results = [[], []]
    for sample in range(repeats):
        for i in (0, 1) if sample % 2 == 0 else (1, 0):
            operation, reset = (legacy, compact)[i]
            reset()
            start = time.perf_counter()
            if count is None:
                operation()
            else:
                operation(count)
            results[i].append(time.perf_counter() - start)
    a, b = map(median, results)
    return {"legacy_ms": a * 1000, "compact_ms": b * 1000, "ratio": b / a}


def differences(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Nonfinite numerical comparison; fixture must be investigated")
    absolute = np.abs(a - b)
    return {
        "absolute": float(np.max(absolute, initial=0)),
        "relative": float(np.max(absolute / np.maximum(np.abs(a), 1e-30), initial=0)),
    }


def main(copies):
    logging.getLogger("andfn").setLevel(logging.ERROR)
    normal = nb.get_num_threads()
    f, e = fixture(copies)
    started = time.perf_counter()
    legacy = prepare(f, e, False)
    compact = prepare(f, e, True)
    # First calls compile every timed signature before warm timings begin.
    nb.set_num_threads(1)
    for prepared in (legacy, compact):
        prepared[0](1)
        prepared[2]()
        prepared[3]()
    first_calls = time.perf_counter() - started
    storage, work = compact[4:]
    report = {
        "elements": e.size,
        "ncoef": sorted(set(map(int, e["ncoef"]))),
        "normal_threads": normal,
        "first_call_compile_and_setup_seconds": first_calls,
        "payload_bytes": {
            "legacy_current": e["coef"].nbytes,
            "legacy_old": e["old_coef"].nbytes,
            "compact_current": storage.coef_data.nbytes,
            "compact_candidate": storage.candidate_coef_data.nbytes,
            "compact_compatibility_old": storage.old_coef_data.nbytes,
            "compact_starts_capacities": storage.coef_start.nbytes
            + storage.coef_capacity.nbytes,
            "retained_work_coefficient_fields": sum(
                work[name].nbytes
                for name in ("coef", "coef0", "coef1", "old_coef", "coef_error")
            ),
            "retained_work_total": work.nbytes,
            "compact_metadata_including_padded_integration": storage.elements.nbytes,
            "legacy_elements_retained_if_caller_keeps_source": e.nbytes,
        },
        "timings": {},
    }
    try:
        for threads in sorted({1, 2, normal}):
            nb.set_num_threads(threads)
            report["timings"][threads] = {
                "scalar_batch": measure_pair(
                    (legacy[2], legacy[1]), (compact[2], compact[1])
                ),
                "vectorized_elements": measure_pair(
                    (legacy[3], legacy[1]), (compact[3], compact[1])
                ),
                "one_iteration": measure_pair(
                    (legacy[0], legacy[1]), (compact[0], compact[1]), 1
                ),
                "six_iterations": measure_pair(
                    (legacy[0], legacy[1]), (compact[0], compact[1]), 6
                ),
            }
        nb.set_num_threads(1)
        a = run_solver(False, 100, legacy_no_growth=True)
        b = run_solver(True, 100)
        report["numerical_differences_by_iteration"] = [
            {
                "iteration": left[0],
                **{
                    name: differences(x, y)
                    for name, x, y in zip(
                        (
                            "coefficients",
                            "candidates",
                            "error",
                            "coefficient_error",
                            "discharge",
                            "discharge_error",
                            "fracture_constant",
                        ),
                        left[2:],
                        right[2:],
                    )
                },
            }
            for left, right in zip(a[0], b[0])
        ]
        report["final_differences"] = {
            name: differences(x, y)
            for name, x, y in zip(
                ("heads", "boundary_errors", "dfn_discharge"), a[1:4], b[1:4]
            )
        }
        fractures, elements = network()
        state = fixed_order_from_hpc(elements)
        points = np.array([1.3 + 0.7j, -1.2 + 0.8j])
        values = [[], []]
        derivatives = [[], []]
        vectors = [[], []]
        for side, (metadata, kwargs) in enumerate(
            (
                (elements, {}),
                (
                    state.elements,
                    {"coef_data": state.coef_data, "coef_start": state.coef_start},
                ),
            )
        ):
            for f in fractures:
                for z in points:
                    values[side].append(hf.calc_omega(f, z, metadata, **kwargs))
                    derivatives[side].append(hf.calc_w(f, z, metadata, **kwargs))
                output = np.zeros(points.size, dtype=complex)
                hf.calc_omega_array(f, output, points, metadata, **kwargs)
                vectors[side].extend(output)
        report["evaluation_differences"] = {
            "scalar_potential": differences(*values),
            "vectorized_potential": differences(*vectors),
            "derivative": differences(*derivatives),
        }
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=16)
    main(parser.parse_args().copies)
