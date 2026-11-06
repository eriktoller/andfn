"""Phase 7 paired error timings and unique NumPy payload accounting."""

import argparse
import contextlib
import json
import logging
import os
from time import perf_counter

import numba as nb
import numpy as np
from benchmark_boundary import prepare as model
from benchmark_quadrature import memory_cases

from andfn.constants import Constants
from andfn.hpc import hpc_error as ce
from andfn.hpc import hpc_error_solve as es
from andfn.hpc import hpc_fracture as hf
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_work as cw


def prepare(count):
    f, e, owner, fractures = model(count)
    for i, element in enumerate(e):
        record = cw.work_record(owner.state.work, i)
        n = element["nint"]
        mf.calc_thetas(n, element["_type"], record["thetas"][:n])
        mf.fill_exp_array(n, record["thetas"], record["exp_array_m"], -1)
        mf.fill_exp_array(n, record["thetas"], record["exp_array_p"], 1)
        mf.fill_z_integral(element, record)
        element["thetas"][:n] = record["thetas"][:n]
    return f, e, owner, fractures


def arrays(error):
    return {
        id(a): a
        for a in (
            *cw.state_arrays(error.state).values(),
            *error.elements,
            *error.fractures,
        )
    }


def memory():
    controls = memory_cases()
    report = {}
    for count in (128, 1024, 10000):
        f, e, owner, fractures = prepare(count)
        normal = {**cw.state_arrays(owner.state), **{id(a): a for a in fractures}}
        error = ce.prepare(owner, fractures)
        private = {key: a for key, a in arrays(error).items() if key not in normal}
        work, storage = error.state.work, error.state.storage
        control = controls[count]
        sparse = sum(
            control[key]
            for key in (
                "sparse_matrix_triplet_input_bytes",
                "matrix_count_offset_position_payload",
                "selected_index_bytes",
            )
        )
        selected = control["selected"]
        branches = sum(
            getattr(work, name).nbytes
            for name in (
                *cw.BRANCH_FIELDS,
                "branch_start",
                "branch_capacity",
                "len_discharge_element",
                "branch_valid",
            )
        )
        coefficients = sum(
            getattr(storage, name).nbytes
            for name in (
                "coef_data",
                "candidate_coef_data",
                "old_coef_data",
            )
        )
        scratch = sum(getattr(work, name).nbytes for name in cw.SCRATCH_FIELDS)
        owned = sum(a.nbytes for a in private.values())
        normal_bytes = sum(a.nbytes for a in normal.values())
        # Conservative sum of allocation-site payloads, not an RSS measurement.
        branch_temporary = error.branch_stats["branch_count_temporary_numpy_payload"]
        # Bound host finite-check expressions (one family at a time).
        finite_temporary = max(storage.coef_data.size, work.thetas.size, count * 5)
        legacy_normal = e.nbytes + f.nbytes + count * es.dtype_work.itemsize
        report[count] = {
            "normal_state": normal_bytes,
            "error_steady": owned,
            "error_scalars_and_fracture_constants": error.elements.scalars.nbytes
            + error.fractures.constants.nbytes,
            "error_coefficients": coefficients,
            "error_correction_cache": work.dpsi_corr.nbytes,
            "error_worker_scratch": scratch,
            "error_branch_state": branches,
            "error_compatibility_and_scalar_work": owned
            - coefficients
            - scratch
            - branches
            - work.dpsi_corr.nbytes
            - error.elements.scalars.nbytes
            - error.fractures.constants.nbytes,
            "error_quadrature": control["compact_discharge_payload"],
            "sparse_inputs": sparse,
            "rhs_discharge_history": control["discharge_rhs_and_history_vector_bytes"],
            "branch_preparation_temporary_envelope": branch_temporary + count * 8,
            "finite_check_temporary_envelope": finite_temporary,
            "total_simultaneous_envelope": normal_bytes
            + owned
            + branch_temporary
            + count * 8
            + finite_temporary
            + control["compact_discharge_payload"]
            + sparse
            + control["discharge_rhs_and_history_vector_bytes"],
            "legacy_error_records": e.nbytes,
            "legacy_selected_record_copy": selected * e.dtype.itemsize,
            "legacy_two_work_arrays": 2 * count * es.dtype_work.itemsize,
            "legacy_two_quadrature_arrays": 2 * count * es.dtype_z_arrays.itemsize,
            "legacy_final_dead_work_and_grid": count
            * (es.dtype_work.itemsize + es.dtype_z_arrays.itemsize),
            "retained_legacy_normal_elements_and_fractures": e.nbytes + f.nbytes,
            "retained_legacy_normal_work": count * es.dtype_work.itemsize,
            "legacy_total_simultaneous_envelope": legacy_normal
            + e.nbytes
            + selected * e.dtype.itemsize
            + 3 * count * es.dtype_work.itemsize
            + 2 * count * es.dtype_z_arrays.itemsize
            + sparse
            + control["discharge_rhs_and_history_vector_bytes"],
            "storage_only": False,
        }
    seed = report[10000]
    projection = {key: value * 10 for key, value in seed.items()}
    for key in ("error_worker_scratch",):
        projection[key] = seed[key]
    for key in ("normal_state", "error_steady", "total_simultaneous_envelope"):
        fixed = (
            64552
            if key == "normal_state"
            else 64544
            if key == "error_steady"
            else 129096
        )
        projection[key] = (seed[key] - fixed) * 10 + fixed
    projection["error_compatibility_and_scalar_work"] = (
        seed["error_compatibility_and_scalar_work"] - 32
    ) * 10 + 32
    projection["storage_only"] = True
    report[100000] = projection
    return report


def measure(count, repeats):
    f, e, owner, fractures = prepare(count)
    constants = Constants().constants.copy()
    output = [[], []]
    final = [None, None]
    for sample in range(repeats + 1):
        for mode in (0, 1) if sample % 2 == 0 else (1, 0):
            timing = {}
            fs = f.copy() if not mode else None
            start = perf_counter()
            if mode:
                error = ce.prepare(owner, fractures)
                timing["state_preparation_seconds"] = perf_counter() - start
                timing["branch_seconds"] = sum(
                    error.branch_stats.get(key, 0)
                    for key in (
                        "branch_direct_seconds",
                        "branch_repack_seconds",
                        "branch_reconstruction_seconds",
                    )
                )
                es.solve_error(
                    error.fractures,
                    error.normal,
                    error.elements,
                    8,
                    constants,
                    None,
                    compact_state=error,
                    timings=timing,
                )
            else:
                source, values = e, e.copy()
                values["coef"][:] = 0
                timing["state_preparation_seconds"] = perf_counter() - start
                es.solve_error(fs, source, values, 8, constants, None, timings=timing)
            timing["total_seconds"] = perf_counter() - start
            final[mode] = (
                [error.elements[i]["coef"].copy() for i in range(count)]
                if mode
                else [row["coef"][: row["ncoef"]].copy() for row in values]
            )
            start = perf_counter()
            if mode:
                hf.get_errors(error.fractures, error.elements, np.array([1.3 + 0.7j]))
            else:
                hf.get_errors(fs, values, np.array([1.3 + 0.7j]))
            timing["postprocessing_seconds"] = perf_counter() - start
            if sample:
                output[mode].append(timing)
    for a, b in zip(*final):
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        np.testing.assert_array_equal(a, b)
    result = {
        key: {
            name: float(np.median([value[name] for value in values]))
            for name in values[0]
        }
        for key, values in zip(("legacy", "compact"), output)
    }
    result["ratio"] = (
        result["compact"]["total_seconds"] / result["legacy"]["total_seconds"]
    )
    result["exact"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int)
    parser.add_argument("--counts", nargs="+", type=int, default=[128, 1024, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    logging.getLogger("andfn").setLevel(logging.ERROR)
    normal = nb.get_num_threads()
    report = {"normal_threads": normal, "memory": memory(), "threads": {}}
    try:
        with open(os.devnull, "w") as stream, contextlib.redirect_stdout(stream):
            for threads in args.threads or sorted({1, 2, 4, 8, 16, normal}):
                nb.set_num_threads(threads)
                report["threads"][threads] = {
                    n: measure(n, args.repeats) for n in args.counts
                }
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
