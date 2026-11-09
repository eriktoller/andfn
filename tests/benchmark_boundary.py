"""Phase 6 end-to-end boundary diagnostics and storage-only accounting."""

import json
import logging
from time import perf_counter

import numba as nb
import numpy as np
from benchmark_fixed_order import fixture
from benchmark_quadrature import memory_cases
from legacy_hpc_reference import (
    adaptive_from_hpc,
    connectivity_from_hpc,
    dtype_work,
    dtype_z_arrays,
    fracture_metadata_from_hpc,
)

from andfn.constants import Constants
from andfn.hpc import hpc_solve as hs
from andfn.hpc import hpc_work as cw
from andfn.hpc.hpc_connectivity import FractureArrays


def prepare(count):
    fractures, elements = fixture(count // 8)
    owner = adaptive_from_hpc(elements)
    connectivity = connectivity_from_hpc(fractures, elements.size)
    compact = FractureArrays(
        fracture_metadata_from_hpc(fractures), connectivity.ids, connectivity.indptr
    )
    return fractures, elements, owner, compact


def measure(count, repeats=7):
    _, _, owner, fractures = prepare(count)
    constants = Constants().constants.copy()
    storage = owner.state.storage
    results, outputs = [], []
    for sample in range(repeats + 1):
        start = perf_counter()
        output = hs.compute_bnd_error(
            fractures,
            storage.elements,
            8,
            constants,
            storage.coef_data,
            storage.coef_start,
            work=owner.state.work,
        )
        outputs.append(output)
        if sample:
            results.append((perf_counter() - start) * 1000)
    for output in outputs[1:]:
        np.testing.assert_array_equal(outputs[0], output)
    return {
        "compact_ms": float(np.median(results)),
        "p10_p90_ms": np.percentile(results, [10, 90]).tolist(),
        "exact": True,
    }


def memory():
    controls = memory_cases()
    report = {}
    for count in (128, 1024, 10000):
        f, e, owner, fractures = prepare(count)
        storage = owner.state.storage
        work, quadrature, stats = hs.prepare_boundary_work(
            fractures, storage.elements, 8, owner.state.work
        )
        shared = cw.state_arrays(owner.state)
        private = {id(a): a for a in work if id(a) not in shared}
        private_bytes = sum(a.nbytes for a in private.values())
        fields = {}
        for name in (
            "branch_start",
            "branch_capacity",
            "len_discharge_element",
            "branch_valid",
            *cw.BRANCH_FIELDS,
        ):
            fields[name] = getattr(work, name).nbytes
        control = controls[count]
        sparse = sum(
            control[key]
            for key in (
                "sparse_matrix_triplet_input_bytes",
                "matrix_count_offset_position_payload",
                "selected_index_bytes",
            )
        )
        report[count] = {
            "legacy_work": count * dtype_work.itemsize,
            "legacy_boundary_quadrature": count * dtype_z_arrays.itemsize,
            "fixed_metadata": 0,
            "coefficient_references": 0,
            "integration_cache": 0,
            "scratch": sum(
                getattr(work, name).nbytes
                for name in cw.SCRATCH_FIELDS
                if id(getattr(work, name)) not in shared
            ),
            "branch_fields": fields,
            "branch_events": sum(fields.values()),
            "quadrature": quadrature.nbytes,
            "temporary_output": count * 7 * 8,
            "compatibility_allocations": 0,
            "boundary_private_work": private_bytes,
            "boundary_owned_payload": private_bytes + quadrature.nbytes + count * 56,
            "branch_preparation_temporary_envelope": stats[
                "branch_count_temporary_numpy_payload"
            ],
            "boundary_peak_payload_envelope": private_bytes
            + quadrature.nbytes
            + count * 56
            + stats["branch_count_temporary_numpy_payload"]
            + count * 8
            + int((e["_type"] == 1).sum()) * 8,
            "compact_persistent_state": cw.state_payload(owner.state),
            "fracture_metadata_and_connectivity": sum(a.nbytes for a in fractures),
            "coefficient_storage": sum(
                getattr(storage, name).nbytes
                for name in ("coef_data", "candidate_coef_data", "old_coef_data")
            ),
            "integration_cache_storage": sum(
                getattr(owner.state.work, name).nbytes for name in cw.CACHE_FIELDS
            ),
            "worker_scratch": sum(
                getattr(owner.state.work, name).nbytes for name in cw.SCRATCH_FIELDS
            ),
            "solver_branch_storage": sum(
                getattr(owner.state.work, name).nbytes
                for name in (
                    *cw.BRANCH_FIELDS,
                    "branch_start",
                    "branch_capacity",
                    "len_discharge_element",
                    "branch_valid",
                )
            ),
            "compact_metadata_layout_and_compatibility": cw.state_payload(owner.state)
            - sum(
                getattr(storage, name).nbytes
                for name in ("coef_data", "candidate_coef_data", "old_coef_data")
            )
            - sum(getattr(owner.state.work, name).nbytes for name in cw.CACHE_FIELDS)
            - sum(getattr(owner.state.work, name).nbytes for name in cw.SCRATCH_FIELDS)
            - sum(
                getattr(owner.state.work, name).nbytes
                for name in (
                    *cw.BRANCH_FIELDS,
                    "branch_start",
                    "branch_capacity",
                    "len_discharge_element",
                    "branch_valid",
                )
            ),
            "discharge_quadrature": control["compact_discharge_payload"],
            "sparse_matrix_inputs": sparse,
            "retained_legacy_elements": e.nbytes,
            "retained_legacy_fractures": f.nbytes,
            "optional_legacy_work": count * dtype_work.itemsize,
            "storage_only": False,
        }
    seed = report[10000]
    projected = {
        key: {name: value * 10 for name, value in item.items()}
        if isinstance(item, dict)
        else item * 10
        for key, item in seed.items()
    }
    for key in ("worker_scratch",):
        projected[key] = seed[key]
    for key in (
        "compact_persistent_state",
        "compact_metadata_layout_and_compatibility",
    ):
        fixed = 32 + (
            seed["worker_scratch"] if key == "compact_persistent_state" else 0
        )
        projected[key] = (seed[key] - fixed) * 10 + fixed
    projected["fracture_metadata_and_connectivity"] = (
        seed["fracture_metadata_and_connectivity"] - 8
    ) * 10 + 8
    projected["storage_only"] = True
    report[100000] = projected
    return report


def main():
    logging.getLogger("andfn").setLevel(logging.ERROR)
    normal = nb.get_num_threads()
    report = {"normal_threads": normal, "memory": memory(), "threads": {}}
    try:
        for threads in sorted({n for n in (1, 2, 4, 8, 16, normal) if n <= normal}):
            nb.set_num_threads(threads)
            report["threads"][threads] = {n: measure(n) for n in (128, 1024, 10000)}
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
