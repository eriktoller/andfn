"""Fixed-order equivalence with padded integration/work compatibility."""

import numba as nb
import numpy as np
import pytest
from test_hpc_connectivity import _network

from andfn.constants import Constants
from andfn.hpc import hpc_bounding_circle as bc
from andfn.hpc import hpc_fracture as hf
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_solve as hs
from andfn.hpc.hpc_coefficients import coefficient_view
from andfn.storage import (
    connectivity_from_hpc,
    fixed_order_from_hpc,
    fixed_order_to_hpc,
    fracture_metadata_from_hpc,
)

RTOL = 1e-13
ATOL = 1e-12


def network():
    fractures, elements = _network()
    fractures = fractures[:2].copy()
    elements = elements[:8].copy()
    elements["center"][[0, 6]] = 0
    elements["center"][2] = -1.5 + 0.7j
    elements["center"][7] = -1.5 - 0.7j
    elements["center"][4] = 1.5 + 0.4j
    elements["endpoints0"][1] = [-0.8 + 0.5j, 0.9 + 0.5j]
    elements["endpoints1"][1] = [-0.6 + 0.5j, 0.7 + 0.5j]
    elements["endpoints0"][3] = [-0.8 - 1j, 0.9 - 1j]
    elements["endpoints0"][5] = [-0.4 + 1.5j, 0.4 + 1.5j]
    elements["ncoef"] = [4, 5, 0, 6, 3, 7, 5, 0]
    elements["nint"] = np.maximum(2 * elements["ncoef"], 2)
    elements["error"] = 1e30
    elements["error_old"] = 1e30
    elements["error_old2"] = 1e30
    elements["coef"][:] = 0
    for e in elements:
        e["coef"][1 : e["ncoef"]] = 0.01 * np.arange(1, e["ncoef"]) + 0.005j
    return fractures, elements


def work_for(elements):
    work = np.zeros(elements.size, dtype=hs.dtype_work)
    for i, e in enumerate(elements):
        mf.calc_thetas(e["nint"], e["_type"], e["thetas"][: e["nint"]])
        mf.fill_exp_array(e["nint"], e["thetas"], work[i]["exp_array_m"], -1)
        mf.fill_exp_array(e["nint"], e["thetas"], work[i]["exp_array_p"], 1)
        mf.fill_z_integral(e, work[i])
    return work


def logical(storage, data):
    return np.concatenate(
        [
            data[start : start + n]
            for start, n in zip(storage.coef_start, storage.elements["ncoef"])
        ]
    )


def legacy_logical(elements, name="coef"):
    return np.concatenate([e[name][: e["ncoef"]] for e in elements])


def assert_close(a, b):
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(b))
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


@nb.njit
def compiled_view(element, data, starts):
    view = coefficient_view(element, data, starts)
    return view, view.ctypes.data - data.ctypes.data


@pytest.mark.parametrize("count", [0, 1, 3, 8])
def test_logical_views_and_compiled_aliasing(count):
    _, elements = network()
    elements["ncoef"] = count
    storage = fixed_order_from_hpc(elements)
    storage.coef_data[:] = 17 + 3j
    e = storage.elements[1]
    view, byte_offset = compiled_view(e, storage.coef_data, storage.coef_start)
    assert view.size == count
    assert byte_offset == 16 * storage.coef_start[1]
    if count:
        assert np.shares_memory(view, storage.coef_data)
        view[0] = 4
        assert storage.coef_data[storage.coef_start[1]] == 4
    assert compiled_view.nopython_signatures
    llvm = compiled_view.inspect_llvm(compiled_view.signatures[0])
    assert "NRT_MemInfo_alloc_aligned" not in llvm


@pytest.mark.parametrize("count", [0, 1, 5])
def test_scalar_vector_derivative_empty_one_and_slack(count):
    fractures, elements = network()
    elements["ncoef"] = count
    storage = fixed_order_from_hpc(elements)
    for start, n, cap in zip(
        storage.coef_start, elements["ncoef"], storage.coef_capacity
    ):
        storage.coef_data[start + n : start + cap] = 1e99 + 1e99j
    z = np.array([1.3 + 0.7j, -1.2 + 0.8j])
    for i, f in enumerate(fractures):
        ids = f["elements"][: f["nelements"]]
        for point in z:
            for exclude in (-1, 1, 3):
                assert_close(
                    hf.calc_omega(
                        f,
                        point,
                        storage.elements,
                        exclude,
                        ids,
                        storage.coef_data,
                        storage.coef_start,
                    ),
                    hf.calc_omega(f, point, elements, exclude),
                )
            assert_close(
                hf.calc_w(
                    f,
                    point,
                    storage.elements,
                    -1,
                    ids,
                    storage.coef_data,
                    storage.coef_start,
                ),
                hf.calc_w(f, point, elements),
            )
        a = np.zeros(z.size, dtype=complex)
        b = a.copy()
        hf.calc_omega_array(f, a, z, elements)
        hf.calc_omega_array(
            f, b, z, storage.elements, -1, ids, storage.coef_data, storage.coef_start
        )
        assert_close(a, b)
    # Bounding-circle derivative must also support a single constant coefficient.
    assert_close(
        bc.calc_w(storage.elements[0], z[0], storage.coef_data, storage.coef_start),
        bc.calc_w(elements[0], z[0]),
    )
    assert hf.calc_omega.nopython_signatures
    assert hf.calc_w.nopython_signatures
    assert hf.calc_omega_array.nopython_signatures
    connectivity = connectivity_from_hpc(fractures, elements.size, index_dtype=np.int32)
    compact_heads, compact_points = hf.get_heads_csr(
        fracture_metadata_from_hpc(fractures),
        storage.elements,
        connectivity.ids,
        connectivity.indptr,
        z,
        coef_data=storage.coef_data,
        coef_start=storage.coef_start,
    )
    legacy_heads, legacy_points = hf.get_heads(fractures, elements, z)
    assert_close(compact_heads, legacy_heads)
    np.testing.assert_array_equal(compact_points, legacy_points)
    assert hf.get_heads_csr.nopython_signatures


@pytest.mark.parametrize("damping", [0.0, 0.5, 1.0])
def test_candidates_are_separate_and_commit_preserves_operation_order(damping):
    f, e = network()
    work = work_for(e)
    storage = fixed_order_from_hpc(e)
    compact_work = work.copy()
    compact_work["coef"][:] = 123456
    for i, n in enumerate(e["ncoef"]):
        start, cap = storage.coef_start[i], storage.coef_capacity[i]
        storage.candidate_coef_data[start + n : start + cap] = 1e99
        storage.coef_data[start + n : start + cap] = 1e99
    current = storage.coef_data.copy()
    hs.calculate_candidates(e.size, e, f, work, 1e-12, 1, 0, damping)
    hs.calculate_candidates(
        e.size,
        storage.elements,
        f,
        compact_work,
        1e-12,
        1,
        0,
        damping,
        storage.coef_data,
        storage.coef_start,
        storage.candidate_coef_data,
    )
    np.testing.assert_array_equal(storage.coef_data, current)
    np.testing.assert_array_equal(compact_work["coef"], 123456)
    assert_close(
        logical(storage, storage.candidate_coef_data),
        np.concatenate([w["coef"][:n] for w, n in zip(work, e["ncoef"])]),
    )
    for field in ("error", "error_coef", "error_old", "error_old2"):
        assert_close(storage.elements[field], e[field])
    hs.commit_coefficients(e, work, damping)
    hs.commit_coefficients(
        storage.elements,
        compact_work,
        damping,
        storage.coef_data,
        storage.coef_start,
        storage.candidate_coef_data,
    )
    assert_close(logical(storage, storage.coef_data), legacy_logical(e))
    assert hs.calculate_candidates.nopython_signatures
    assert hs.commit_coefficients.nopython_signatures


def test_fixed_step_matches_no_growth_adaptive_schedule():
    f, e = network()
    work = work_for(e)
    storage = fixed_order_from_hpc(e)
    compact_work = work.copy()
    initial = e[["ncoef", "nint"]].copy()
    # A huge ratio threshold disables growth without skipping the second commit.
    for nit in range(1, 7):
        hs.element_solver2(e.size, e, f, work, 1e-12, nit, 0, 1e100, 0, 0, 0.5)
        hs.fixed_order_step(
            storage.elements,
            f,
            compact_work,
            1e-12,
            nit,
            0,
            0.5,
            storage.coef_data,
            storage.coef_start,
            storage.candidate_coef_data,
        )
        np.testing.assert_array_equal(e[["ncoef", "nint"]], initial)
        assert_close(logical(storage, storage.coef_data), legacy_logical(e))
        assert_close(storage.elements["error_coef"], e["error_coef"])


def run_solver(compact, iterations=8, *, legacy_no_growth=False):
    fractures, elements = network()
    storage = fixed_order_from_hpc(elements) if compact else None
    constants = Constants().constants.copy()
    constants["MAX_ITERATIONS"] = iterations
    constants["MAX_ERROR"] = 1e-12
    constants["DAMPING"] = 0.5
    if legacy_no_growth:
        constants["MAX_NCOEF"] = 0
    history = []

    def observe(nit, e, f, work, q, error_q, cnt_error):
        np.testing.assert_array_equal(e["_id"], np.arange(elements.size))
        np.testing.assert_array_equal(e["ncoef"], [4, 5, 0, 6, 3, 7, 5, 0])
        np.testing.assert_array_equal(e["nint"], [8, 10, 2, 12, 6, 14, 10, 2])
        coefficients = (
            logical(storage, storage.coef_data) if compact else legacy_logical(e)
        )
        candidates = (
            logical(storage, storage.candidate_coef_data)
            if compact
            else np.concatenate([w["coef"][:n] for w, n in zip(work, e["ncoef"])])
        )
        history.append(
            (
                nit,
                cnt_error,
                coefficients.copy(),
                candidates.copy(),
                e["error"].copy(),
                e["error_coef"].copy(),
                q.copy(),
                error_q,
                f["constant"].copy(),
            )
        )

    if compact:
        hs.solve_fixed_order(
            storage, fractures, 8, constants, iteration_callback=observe
        )
        result = storage.elements
        kwargs = {"coef_data": storage.coef_data, "coef_start": storage.coef_start}
    else:
        hs.solve(
            fractures,
            elements,
            8,
            constants,
            None,
            fixed_order=not legacy_no_growth,
            iteration_callback=observe,
        )
        result = elements
        kwargs = {}
    boundary = hs.compute_bnd_error(fractures, result, 8, constants, **kwargs)
    heads = np.array(
        [
            [
                hf.head_from_phi(f, hf.calc_omega(f, z, result, **kwargs).real)
                for z in (1.3 + 0.7j, -1.2 + 0.8j)
            ]
            for f in fractures
        ]
    )
    # Match DFN.get_dfn_discharge: wells and head lines, excluding intersections.
    discharge = np.sum(np.abs(result["q"][np.isin(result["_type"], (2, 3))])) / 2
    return history, heads, boundary, discharge, storage


def test_full_fixed_solve_iteration_equivalence_and_parallel_repeatability():
    normal = nb.get_num_threads()
    configs = sorted({1, min(2, nb.config.NUMBA_NUM_THREADS), normal})
    try:
        reference = None
        for threads in configs:
            nb.set_num_threads(threads)
            legacy = run_solver(False)
            compact = run_solver(True)
            repeat = run_solver(True)
            for a, b in ((legacy, compact), (compact, repeat)):
                assert len(a[0]) == len(b[0])
                for left, right in zip(a[0], b[0]):
                    assert left[:2] == right[:2]
                    for x, y in zip(left[2:], right[2:]):
                        assert_close(x, y)
                for x, y in zip(a[1:4], b[1:4]):
                    assert_close(x, y)
            if reference is not None:
                assert_close(reference[1], compact[1])
                assert_close(reference[2], compact[2])
            reference = compact
            storage = compact[4]
            np.testing.assert_array_equal(storage.elements["_id"], np.arange(8))
            np.testing.assert_array_equal(
                storage.elements["ncoef"], [4, 5, 0, 6, 3, 7, 5, 0]
            )
            np.testing.assert_array_equal(
                storage.coef_start, [0, 4, 12, 12, 20, 24, 32, 40]
            )
    finally:
        nb.set_num_threads(normal)


def test_converged_compact_solve_matches_legacy_adaptive_with_growth_disabled():
    normal = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        legacy = run_solver(False, 100, legacy_no_growth=True)
        compact = run_solver(True, 100)
        assert len(legacy[0]) == len(compact[0]) < 100
        assert legacy[0][-1][1] == compact[0][-1][1] == 2
        for left, right in zip(legacy[0], compact[0]):
            assert left[:2] == right[:2]
            for x, y in zip(left[2:], right[2:]):
                assert_close(x, y)
        for x, y in zip(legacy[1:4], compact[1:4]):
            assert_close(x, y)
    finally:
        nb.set_num_threads(normal)


def test_payload_scales_with_capacity_and_rejects_aliasing():
    _, e = network()
    storage = fixed_order_from_hpc(e)
    assert "coef" not in storage.elements.dtype.names
    assert "old_coef" not in storage.elements.dtype.names
    assert storage.elements.dtype["thetas"] == e.dtype["thetas"]
    assert storage.elements.dtype["dpsi_corr"] == e.dtype["dpsi_corr"]
    for buffer in (
        storage.coef_data,
        storage.candidate_coef_data,
        storage.old_coef_data,
    ):
        assert buffer.nbytes == 16 * storage.coef_capacity.sum()
        assert buffer.nbytes < e["coef"].nbytes
    storage.candidate_coef_data = storage.coef_data
    with pytest.raises(ValueError, match="alias"):
        storage.validate()


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_one_coefficient_candidates(count):
    f, e = network()
    e["ncoef"] = count
    work = work_for(e)
    storage = fixed_order_from_hpc(e)
    compact_work = work.copy()
    hs.fixed_order_step(e, f, work, 1e-12, 2, 0, 0.5)
    hs.fixed_order_step(
        storage.elements,
        f,
        compact_work,
        1e-12,
        2,
        0,
        0.5,
        storage.coef_data,
        storage.coef_start,
        storage.candidate_coef_data,
    )
    assert_close(logical(storage, storage.coef_data), legacy_logical(e))
    assert_close(storage.elements["error_coef"], e["error_coef"])


def test_explicit_export_is_independent_and_preserves_compatibility_fields():
    _, elements = network()
    elements["old_coef"][:, :3] = 4 + 2j
    storage = fixed_order_from_hpc(elements)
    exported = fixed_order_to_hpc(storage)
    for name in storage.elements.dtype.names:
        np.testing.assert_array_equal(exported[name], elements[name])
    for i, n in enumerate(elements["ncoef"]):
        np.testing.assert_array_equal(exported["coef"][i, :n], elements["coef"][i, :n])
        np.testing.assert_array_equal(
            exported["old_coef"][i, :n], elements["old_coef"][i, :n]
        )
    assert not np.shares_memory(exported, storage.elements)


def test_compact_buffers_cannot_enter_adaptive_solver():
    fractures, elements = network()
    storage = fixed_order_from_hpc(elements)
    with pytest.raises(ValueError, match="fixed-order"):
        hs.solve(
            fractures,
            storage.elements,
            8,
            Constants().constants,
            None,
            coef_data=storage.coef_data,
            coef_start=storage.coef_start,
            candidate_coef_data=storage.candidate_coef_data,
        )
