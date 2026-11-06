"""Phase 5 selected dense quadrature, matrix/RHS and unchanged iteration timings."""

import argparse
import contextlib
import io
import json
import logging
from time import perf_counter

import numba as nb
import numpy as np
from benchmark_adaptive import run_case
from benchmark_fixed_order import fixture

from andfn.hpc import hpc_solve as hs
from andfn.hpc.hpc_quadrature import allocate_quadrature
from andfn.storage import connectivity_from_hpc, fixed_order_from_hpc


def describe(values):
    return {
        "median_ms": float(np.median(values)),
        "p10_p90_ms": np.percentile(values, [10, 90]).tolist(),
    }


def measure(operations, repeats=9):
    for operation in operations.values():
        operation()
    elapsed = {key: [] for key in operations}
    for sample in range(repeats):
        names = list(operations)
        if sample % 2:
            names.reverse()
        for name in names:
            start = perf_counter()
            operations[name]()
            if sample:
                elapsed[name].append(1000 * (perf_counter() - start))
    return {key: describe(values) for key, values in elapsed.items()}


def kernel_case(count):
    fractures, source = fixture(count // 8)
    state = fixed_order_from_hpc(source)
    elements = state.elements
    connectivity = connectivity_from_hpc(fractures, count)
    indices = hs.get_discharge_indices(elements)
    q = 8
    padded = np.zeros(count, dtype=hs.dtype_z_arrays)
    dense = allocate_quadrature(indices.size, q)
    for values in (padded, dense):
        hs.get_z_int_array(values, elements, q, indices)
    rhs = np.zeros(indices.size + fractures.size)

    def prepare(values):
        hs.get_z_int_array(values, elements, q, indices)

    def matrix(values):
        return hs.build_discharge_matrix_csr(
            fractures,
            elements,
            connectivity.ids,
            connectivity.indptr,
            indices,
            q,
            values,
        )

    def head(values):
        hs.build_head_matrix(
            fractures,
            elements,
            elements,
            q,
            rhs,
            values,
            connectivity.ids,
            connectivity.indptr,
            indices,
            state.coef_data,
            state.coef_start,
        )

    def allocate_padded():
        values = np.zeros(count, dtype=hs.dtype_z_arrays)
        prepare(values)

    def allocate_dense():
        values = allocate_quadrature(indices.size, q)
        prepare(values)

    result = {
        "elements": count,
        "selected": int(indices.size),
        "quadrature_count": q,
        "padded_one_buffer_bytes": padded.nbytes,
        "compact_bytes": dense.nbytes,
        "preparation": measure(
            {"padded": lambda: prepare(padded), "dense": lambda: prepare(dense)}
        ),
        "allocation_and_preparation": measure(
            {"padded": allocate_padded, "dense": allocate_dense}
        ),
        "matrix": measure(
            {"padded": lambda: matrix(padded), "dense": lambda: matrix(dense)}
        ),
        "rhs": measure({"padded": lambda: head(padded), "dense": lambda: head(dense)}),
    }
    for side, name in enumerate(("z0", "z1")):
        np.testing.assert_array_equal(dense[side], padded[name][: indices.size, :q])
    a, b = matrix(padded), matrix(dense)
    for field in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    head(padded)
    expected = rhs.copy()
    head(dense)
    np.testing.assert_array_equal(rhs, expected)
    result["exact_coordinate_matrix_rhs"] = True
    return result


def memory_cases():
    fractures, elements = fixture(1)
    indices = hs.get_discharge_indices(elements)
    connectivity = connectivity_from_hpc(fractures, elements.size)
    per_row = hs.count_discharge_nnz(
        fractures, elements, elements, connectivity.ids, connectivity.indptr, indices
    )
    nnz_seed = int(per_row.sum())
    report = {}
    for count in (128, 1024, 10000, 100000):
        copies = count // 8
        selected = indices.size * copies
        q = 8
        compact = allocate_quadrature(selected, q)
        actual = compact.nbytes
        assert actual == selected * q * 16 * 2
        report[count] = {
            "selected": int(selected),
            "quadrature_count": q,
            "legacy_two_padded_payload": 2 * count * hs.dtype_z_arrays.itemsize,
            "compact_discharge_payload": actual,
            "compact_boundary_payload_separate_operation": 2 * count * q * 16,
            "bytes_per_selected_element": q * 16 * 2,
            "temporary_native_coordinate_vector_payload_bound": (q + 2) * 80,
            "retained_after_solve": 0,
            "sparse_matrix_triplet_input_bytes": nnz_seed * copies * 24,
            "matrix_count_offset_position_payload": count * 8
            + per_row.size * copies * 16,
            "selected_index_bytes": selected * 8,
            "csr_connectivity_bytes": connectivity.ids.nbytes * copies
            + (fractures.size * copies + 1) * 8,
            "legacy_elements_retained_bytes": count * elements.dtype.itemsize,
            "optional_legacy_work_bytes": count * hs.dtype_work.itemsize,
            "shared_padded_fracture_input_bytes": copies * fractures.nbytes,
            "discharge_rhs_and_history_vector_bytes": 3
            * (selected + copies * fractures.size)
            * 8,
            "storage_only": count == 100000,
        }
    return report


def main(include_adaptive):
    logging.getLogger("andfn").setLevel(logging.ERROR)
    normal = nb.get_num_threads()
    report = {"memory": memory_cases(), "threads": {}}
    try:
        for threads in sorted({n for n in (1, 2, 4, 8, 16, normal) if n <= normal}):
            nb.set_num_threads(threads)
            cases = {count: kernel_case(count) for count in (128, 1024, 10000)}
            if include_adaptive:
                cases["adaptive"] = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    for name, args in (
                        ("no", (16, 1e100, 2, 0)),
                        ("spare", (16, -1, 1, 10)),
                        ("forced", (16, -1, 5, 40)),
                        ("1024", (128, -1, 5, 40)),
                    ):
                        cases["adaptive"][name] = {
                            layout: run_case(
                                *args,
                                spare=name == "spare",
                                compact_quadrature=layout == "dense",
                            )
                            for layout in ("padded", "dense")
                        }
            report["threads"][threads] = cases
    finally:
        nb.set_num_threads(normal)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", action="store_true")
    main(parser.parse_args().adaptive)
