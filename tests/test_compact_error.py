"""Error storage migration: exact transitions and normal-state isolation."""

import numba as nb
import numpy as np
import pytest
from benchmark_boundary import prepare as model
from legacy_hpc_reference import (
    dtype_work,
    dtype_z_arrays,
    element_dtype_hpc,
    fracture_dtype_hpc,
)

from andfn.constants import Constants
from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_error as ce
from andfn.hpc import hpc_error_solve as es
from andfn.hpc import hpc_fracture as hf
from andfn.hpc import hpc_work as cw


def solved(count=8):
    f, e, owner, fractures = model(count)
    constants = Constants().constants.copy()
    constants["MAX_ITERATIONS"] = 2
    adaptive.solve_adaptive(owner, fractures, 8, constants)
    s = owner.state.storage
    # Explicit reference fixture only, never part of compact execution.
    for i, row in enumerate(e):
        for name in s.elements.dtype.names:
            row[name] = s.elements[name][i]
        c, n = s.coef_start[i], row["ncoef"]
        row["coef"][:] = 0
        row["coef"][:n] = s.coef_data[c : c + n]
        for name in ("thetas", "dpsi_corr"):
            row[name][: row["nint"]] = cw.work_record(owner.state.work, i)[name][
                : row["nint"]
            ]
    for name in f.dtype.names:
        if name in fractures.metadata.dtype.names:
            f[name] = fractures[name]
    return f, e, owner, fractures, constants


def snapshot(stage, nit, elements, work, fractures, candidates, convergence):
    result = {
        "stage": stage,
        "nit": nit,
        "scalars": np.array(
            [[elements[i][name] for name in ce.SCALARS] for i in range(elements.size)]
        ),
        "constants": fractures["error_constant"].copy(),
        "convergence": convergence,
        "orders": np.array(
            [
                [elements[i][name] for name in ("_id", "ncoef", "nint")]
                for i in range(elements.size)
            ]
        ),
        "error_potential": np.array(
            [
                hf.calc_omega_error(fractures[i], 1.3 + 0.7j, elements)
                for i in range(fractures.size)
            ]
        ),
    }
    for name in ("coef", "thetas", "dpsi_corr"):
        result[name] = [
            elements[i][name][
                : elements[i]["ncoef"] if name == "coef" else elements[i]["nint"]
            ].copy()
            for i in range(elements.size)
        ]
    for name in (
        "old_coef",
        "exp_array_m",
        "exp_array_p",
        "z_integral",
        *cw.BRANCH_FIELDS,
    ):
        values = []
        for i in range(elements.size):
            row = cw.work_record(work, i)
            n = (
                elements[i]["ncoef"]
                if name == "old_coef"
                else (
                    row["len_discharge_element"]
                    if name in cw.BRANCH_FIELDS
                    else elements[i]["nint"]
                )
            )
            values.append(row[name][:n].copy())
        result[name] = values
    result["candidate"] = [
        ce.error_work_record(
            work, i, candidates, work.coef_start if candidates is not None else None
        )["coef"][: elements[i]["ncoef"]].copy()
        if candidates is None
        else candidates[
            work.coef_start[i] : work.coef_start[i] + elements[i]["ncoef"]
        ].copy()
        for i in range(elements.size)
    ]
    return result


@pytest.mark.parametrize(
    "threads", sorted({1, 2, 4, 8, 16, nb.config.NUMBA_NUM_THREADS})
)
def test_error_transitions_are_exact_and_normal_state_is_preserved(threads, capsys):
    previous = nb.get_num_threads()
    nb.set_num_threads(threads)
    try:
        f, _e, owner, fractures, constants = solved()
        normal = {
            id(a): a.copy()
            for a in (*cw.state_arrays(owner.state).values(), *fractures)
        }
        errors = [ce.prepare(owner, fractures) for _ in range(2)]
        traces = [[], []]
        for index, error in enumerate(errors):
            es.solve_error(
                error.fractures,
                error.normal,
                error.elements,
                8,
                constants,
                compact_state=error,
                iteration_callback=lambda *args, index=index: traces[index].append(
                    snapshot(*args)
                ),
            )
        assert len(traces[0]) == len(traces[1])
        for a, b in zip(*traces):
            assert a.keys() == b.keys()
            for name in a:
                if isinstance(a[name], list):
                    for x, y in zip(a[name], b[name]):
                        assert np.all(np.isfinite(x))
                        assert np.all(np.isfinite(y))
                        np.testing.assert_array_equal(x, y, err_msg=name)
                else:
                    if name != "stage":
                        assert np.all(np.isfinite(a[name]))
                        assert np.all(np.isfinite(b[name]))
                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        for a in (*cw.state_arrays(owner.state).values(), *fractures):
            np.testing.assert_array_equal(a, normal[id(a)])
        np.testing.assert_array_equal(error.fractures.ids, fractures.ids)
        np.testing.assert_array_equal(error.fractures.indptr, fractures.indptr)
        output = capsys.readouterr().out
        assert "Warning: Impermeable circle not implemented yet. Skipping." in output
        assert "Warning: Impermeable line not implemented yet. Skipping." in output
        error = errors[0]
        for i in range(f.size):
            for point in (1.3 + 0.7j, 0.7 + 0.2j):
                np.testing.assert_array_equal(
                    hf.calc_omega_error(
                        errors[1].fractures[i], point, errors[1].elements
                    ),
                    hf.calc_omega_error(error.fractures[i], point, error.elements),
                )
    finally:
        nb.set_num_threads(previous)


def test_public_compact_error_never_allocates_padded_records_and_repeats(monkeypatch):
    from types import SimpleNamespace

    from andfn.dfn import DFN

    _f, _e, owner, fractures, constants = solved()
    dfn = DFN("compact-error", discharge_int=8)
    dfn._compact_owner, dfn._compact_fractures = owner, fractures
    dfn.constants = constants
    dfn.fractures = [SimpleNamespace(_id=i) for i in range(fractures.size)]
    dfn.solve_error()
    first = dfn._compact_error.state.storage.coef_data.copy()
    expected = dfn.calc_error(1.3 + 0.7j, dfn.fractures[0])
    originals = np.zeros, np.empty

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy conversion entered compact error solve")

    monkeypatch.setattr(dfn, "consolidate_dfn", forbidden)
    for name, allocator in zip(("zeros", "empty"), originals):

        def guarded(*args, allocator=allocator, **kwargs):
            dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
            if dtype is not None and np.dtype(dtype) in (
                element_dtype_hpc,
                fracture_dtype_hpc,
                dtype_work,
                dtype_z_arrays,
            ):
                raise AssertionError("Padded error allocation")
            return allocator(*args, **kwargs)

        monkeypatch.setattr(np, name, guarded)
    dfn.solve_error()
    np.testing.assert_array_equal(first, dfn._compact_error.state.storage.coef_data)
    assert dfn.calc_error(1.3 + 0.7j, dfn.fractures[0]) == expected
    assert not hasattr(dfn, "elements_struc_array_hpc")
    assert not hasattr(dfn, "fractures_struc_array_hpc")
    monkeypatch.setattr(np, "zeros", originals[0])
    monkeypatch.setattr(np, "empty", originals[1])
    points = np.array([0.1 + 0.2j])
    all_errors = hf.get_errors(
        dfn._compact_error.fractures, dfn._compact_error.elements, points
    )
    subset = hf.get_errors(
        dfn._compact_error.fractures,
        dfn._compact_error.elements,
        points,
        fracture_indices=np.array([1, 0], dtype=np.int64),
    )
    for a, b in zip(all_errors, subset):
        np.testing.assert_array_equal(a[[1, 0]], b)


def test_error_payload_and_results_are_independent_of_unused_storage_maxima(
    monkeypatch,
):
    from andfn import element

    _f, _e, owner, fractures, constants = solved()
    results = []
    normal_ids = set(cw.state_arrays(owner.state)) | {id(a) for a in fractures}
    for m, n in ((200, 500), (500, 5000), (1000, 5000)):
        monkeypatch.setattr(element, "MAX_NCOEF", m, raising=False)
        monkeypatch.setattr(element, "MAX_ELEMENTS", n, raising=False)
        monkeypatch.setattr(es, "MAX_NCOEF", m, raising=False)
        monkeypatch.setattr(es, "MAX_ELEMENTS", n, raising=False)
        error = ce.prepare(owner, fractures)
        es.solve_error(
            error.fractures,
            error.normal,
            error.elements,
            8,
            constants,
            compact_state=error,
        )
        arrays = {
            id(a): a
            for a in (
                *cw.state_arrays(error.state).values(),
                *error.elements,
                *error.fractures,
            )
        }
        payload = sum(a.nbytes for key, a in arrays.items() if key not in normal_ids)
        results.append((payload, error.state.storage.coef_data.copy()))
    for payload, coefficients in results[1:]:
        assert payload == results[0][0]
        np.testing.assert_array_equal(coefficients, results[0][1])


def test_error_preparation_rejects_nonfinite_normal_state():
    _f, _e, owner, fractures, _constants = solved()
    owner.state.storage.coef_data[0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        ce.prepare(owner, fractures)


def test_error_branch_overflow_uses_private_batch_recovery(monkeypatch):
    _f, _e, owner, fractures, _constants = solved()
    normal = {key: a.copy() for key, a in cw.state_arrays(owner.state).items()}
    expected = ce.prepare(owner, fractures)
    original = ce.allocate_work

    def zero_capacity(*args, **kwargs):
        work = original(*args, **kwargs)
        return work._replace(
            branch_capacity=np.zeros(work.branch_capacity.size, dtype=np.int64),
            branch_start=np.zeros(work.branch_start.size, dtype=np.int64),
            **{name: np.empty(0, dtype=np.int64) for name in cw.BRANCH_FIELDS},
        )

    monkeypatch.setattr(ce, "allocate_work", zero_capacity)
    actual = ce.prepare(owner, fractures)
    assert actual.branch_stats["branch_overflow_elements"] > 0
    assert actual.branch_stats["branch_repacks"] == 1
    for i in range(actual.elements.size):
        a = cw.work_record(actual.state.work, i)
        b = cw.work_record(expected.state.work, i)
        n = b["len_discharge_element"]
        assert a["len_discharge_element"] == n
        for name in cw.BRANCH_FIELDS:
            np.testing.assert_array_equal(a[name][:n], b[name][:n])
    for key, array in cw.state_arrays(owner.state).items():
        np.testing.assert_array_equal(array, normal[key])


def test_error_preserves_overcapacity_integration_layout():
    _f, _e, owner, fractures, constants = solved()
    expected = ce.prepare(owner, fractures)
    s, old_work = owner.state.storage, owner.state.work
    larger = s.elements.copy()
    larger["nint"] += 25
    expanded = cw.allocate_work(larger, s.coef_start, s.coef_capacity, scratch_groups=7)
    for i in range(s.elements.size):
        a, b = cw.work_record(expanded, i), cw.work_record(old_work, i)
        for name in cw.CACHE_FIELDS:
            a[name][: s.elements["nint"][i]] = b[name][: s.elements["nint"][i]]
    owner.state = adaptive.AdaptiveState(s, expanded)
    actual = ce.prepare(owner, fractures)
    assert actual.state.work.integration_start is expanded.integration_start
    assert actual.state.work.integration_capacity is expanded.integration_capacity
    for name in ("thetas", "exp_array_m", "exp_array_p", "z_integral"):
        assert getattr(actual.state.work, name) is getattr(expanded, name)
    assert not np.shares_memory(actual.state.work.dpsi_corr, expanded.dpsi_corr)
    for error in (expected, actual):
        es.solve_error(
            error.fractures,
            error.normal,
            error.elements,
            8,
            constants,
            compact_state=error,
        )
    np.testing.assert_array_equal(
        actual.state.storage.coef_data, expected.state.storage.coef_data
    )


def test_public_normal_error_postprocessing_and_failed_publication(monkeypatch):
    from andfn import ConstantHeadLine, Fracture
    from andfn import dfn as dfn_module
    from andfn.dfn import DFN

    fracture = Fracture(
        "f",
        1.0,
        radius=4.0,
        center=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        ncoef=8,
        nint=16,
        aperture=1.0,
    )
    for x, head in ((-2.0, 10.0), (2.0, 8.0)):
        ConstantHeadLine(str(x), np.array([x - 0.1j, x + 0.1j]), head, fracture, 8, 16)
    dfn = DFN("normal-and-error", discharge_int=8, MAX_ITERATIONS=6)
    dfn.add_fracture(fracture)
    dfn.check_connectivity()
    dfn.solve()
    before = {
        key: a.copy() for key, a in cw.state_arrays(dfn._compact_owner.state).items()
    }
    normal_head = dfn.calc_head(1.3 + 0.7j, fracture)
    normal_discharge = dfn.get_dfn_discharge()
    normal_constants = dfn._compact_fractures.metadata.copy()
    dfn.solve_error()
    for key, a in cw.state_arrays(dfn._compact_owner.state).items():
        np.testing.assert_array_equal(a, before[key])
    np.testing.assert_array_equal(dfn._compact_fractures.metadata, normal_constants)
    assert dfn.calc_head(1.3 + 0.7j, fracture) == normal_head
    assert dfn.get_dfn_discharge() == normal_discharge
    for point in (1.3 + 0.7j, 0.7 + 0.2j):
        assert np.isfinite(dfn.calc_error(point, dfn.fractures[0]))
    points = np.array([0.1 + 0.2j, 0.5 + 0.3j])
    for heads in (False, True):
        values = hf.get_errors(
            dfn._compact_error.fractures,
            dfn._compact_error.elements,
            points,
            heads=heads,
        )
        assert all(np.isfinite(a).all() for a in values)
    boundary, _ = dfn.check_boundary_conditions(8)
    assert np.isfinite(boundary).all()
    published = dfn._compact_error
    output = dfn.error_structured_array

    def fail(*args, **kwargs):
        args[2].scalars["q"][:] = np.nan
        raise ValueError("Injected error-solver failure")

    monkeypatch.setattr(dfn_module, "hpc_solve_error", fail)
    with pytest.raises(ValueError, match="Injected"):
        dfn.solve_error()
    assert dfn._compact_error is published
    assert dfn.error_structured_array is output
    dfn.reset()
    assert dfn._compact_owner is None
    assert dfn._compact_error is None
    assert dfn.error_structured_array is None
