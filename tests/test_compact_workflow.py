"""Phase 6: padded-free public workflow and private boundary topology."""

from types import SimpleNamespace

import numba as nb
import numpy as np
import pytest
from test_hpc_coefficients import network

from andfn.constants import Constants
from andfn.dfn import DFN
from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_compact as compact
from andfn.hpc import hpc_fracture as hf
from andfn.hpc import hpc_solve as hs
from andfn.hpc import hpc_work as cw
from andfn.hpc.hpc_connectivity import FractureArrays
from andfn.storage import connectivity_from_hpc, fracture_metadata_from_hpc


def objects():
    fractures, elements = network()
    fs = [
        SimpleNamespace(
            **{
                name: f[name].copy()
                for name in fracture_metadata_from_hpc(fractures).dtype.names
            },
            elements=[],
            aperture=1.0,
        )
        for f in fractures
    ]
    es = []
    for row in elements:
        values = {name: row[name].copy() for name in row.dtype.names}
        values["frac0"] = fs[row["frac0"]] if row["frac0"] >= 0 else None
        values["frac1"] = fs[row["frac1"]] if row["frac1"] >= 0 else None
        es.append(SimpleNamespace(**values))
    for f, row in zip(fs, fractures):
        f.elements = [es[i] for i in row["elements"][: row["nelements"]]]
        f.head_from_phi = lambda phi, t=f.t: phi / t
    return fs, es


def bundle(fractures, elements):
    connectivity = connectivity_from_hpc(fractures, elements.size)
    return FractureArrays(
        fracture_metadata_from_hpc(fractures), connectivity.ids, connectivity.indptr
    )


@pytest.mark.parametrize("count", [1, 8, 100, 401])
def test_boundary_exact_and_persistent_state_unchanged(count):
    f, e = network()
    owner = adaptive.adaptive_from_hpc(e)
    storage = owner.state.storage
    fractures = bundle(f, e)
    before = {key: value.copy() for key, value in cw.state_arrays(owner.state).items()}
    constants = Constants().constants
    if count <= hs.MAX_NCOEF * 2:
        expected = hs.compute_bnd_error(f, e, count, constants)
    else:
        expected = hs.compute_bnd_error(
            f, storage.elements, count, constants, storage.coef_data, storage.coef_start
        )
    result = hs.compute_bnd_error(
        fractures,
        storage.elements,
        count,
        constants,
        storage.coef_data,
        storage.coef_start,
        work=owner.state.work,
    )
    np.testing.assert_array_equal(result, expected)
    scratch = {id(getattr(owner.state.work, name)) for name in cw.SCRATCH_FIELDS}
    for key, value in cw.state_arrays(owner.state).items():
        if key not in scratch:
            np.testing.assert_array_equal(value, before[key])
    repeated = hs.compute_bnd_error(
        fractures,
        storage.elements,
        count,
        constants,
        storage.coef_data,
        storage.coef_start,
        work=owner.state.work,
    )
    np.testing.assert_array_equal(result, repeated)


def test_direct_consolidation_matches_legacy_logical_state():
    fs, es = objects()
    owner, fractures = compact.consolidate(es, fs)
    f, e = network()
    expected = adaptive.adaptive_from_hpc(e)
    for name in owner.state.storage.elements.dtype.names:
        np.testing.assert_array_equal(
            owner.state.storage.elements[name], expected.state.storage.elements[name]
        )
    np.testing.assert_array_equal(
        owner.state.storage.coef_data, expected.state.storage.coef_data
    )
    np.testing.assert_array_equal(fractures.ids, bundle(f, e).ids)
    view = connectivity_from_hpc(fractures, len(es))
    assert view.ids is fractures.ids
    narrow = connectivity_from_hpc(fractures, len(es), index_dtype=np.int32)
    np.testing.assert_array_equal(narrow.ids, fractures.ids)
    assert narrow.ids.dtype == np.int32
    with pytest.raises(ValueError, match="index_dtype"):
        connectivity_from_hpc(fractures, len(es), index_dtype=float)


@pytest.mark.parametrize("copies", [1, 2, 3])
def test_compiled_fracture_extent_is_metadata_size_not_tuple_arity(copies):
    from benchmark_fixed_order import fixture

    f, e = fixture(copies)
    owner = adaptive.adaptive_from_hpc(e)
    storage = owner.state.storage
    fractures = bundle(f, e)
    expected = hf.get_flow_nets(f, 2, 2, e)
    actual = hf.get_flow_nets(
        fractures, 2, 2, storage.elements, storage.coef_data, storage.coef_start
    )
    for a, b in zip(expected, actual):
        assert b.shape[0] == f.size
        np.testing.assert_array_equal(a, b)


def test_empty_boundary_and_worker_scratch_reuse_and_resize():
    fs, es = objects()
    owner, _fractures = compact.consolidate(es, fs)
    state = owner.state
    small = cw.boundary_work(state.storage.elements, 1, state.work)
    for name in cw.SCRATCH_FIELDS + cw.CACHE_FIELDS:
        assert getattr(small, name) is getattr(state.work, name)
    count = int(state.work.scratch_layout[3]) + 1
    large = cw.boundary_work(state.storage.elements, count, state.work)
    for name in cw.CACHE_FIELDS:
        assert getattr(large, name) is getattr(state.work, name)
    for name in ("phi", "psi", "dpsi"):
        assert getattr(large, name).size == nb.config.NUMBA_NUM_THREADS * count
        assert not np.shares_memory(getattr(large, name), getattr(state.work, name))
    empty, no_fractures = compact.consolidate([], [])
    output = hs.compute_bnd_error(
        no_fractures,
        empty.state.storage.elements,
        1,
        Constants().constants,
        empty.state.storage.coef_data,
        empty.state.storage.coef_start,
        work=empty.state.work,
    )
    assert output.shape == (0, 7)


@pytest.mark.parametrize("threads", [1, 2, nb.config.NUMBA_NUM_THREADS])
def test_csr_solver_and_evaluations_match_legacy_fractures_exactly(threads):
    previous = nb.get_num_threads()
    nb.set_num_threads(threads)
    try:
        f, e = network()
        owners = [adaptive.adaptive_from_hpc(e) for _ in range(2)]
        constants = Constants().constants.copy()
        constants["MAX_ITERATIONS"] = 6
        for owner, fractures in zip(owners, (f, bundle(f, e))):
            adaptive.solve_adaptive(owner, fractures, 8, constants)
        a, b = owners
        for name in a.state.storage.elements.dtype.names:
            np.testing.assert_array_equal(
                a.state.storage.elements[name], b.state.storage.elements[name]
            )
        np.testing.assert_array_equal(
            a.state.storage.coef_data, b.state.storage.coef_data
        )
        for name in cw.BRANCH_FIELDS + cw.CACHE_FIELDS:
            np.testing.assert_array_equal(
                getattr(a.state.work, name), getattr(b.state.work, name)
            )
        for i in range(f.size):
            for fn in (hf.calc_omega, hf.calc_w):
                args = {
                    "coef_data": a.state.storage.coef_data,
                    "coef_start": a.state.storage.coef_start,
                }
                np.testing.assert_array_equal(
                    fn(f[i], 1.3 + 0.7j, a.state.storage.elements, **args),
                    fn(bundle(f, e)[i], 1.3 + 0.7j, a.state.storage.elements, **args),
                )
    finally:
        nb.set_num_threads(previous)


def test_public_workflow_does_not_allocate_legacy_records(monkeypatch):
    fs, es = objects()
    dfn = DFN("compact", discharge_int=8, MAX_ITERATIONS=2)
    dfn.fractures, dfn.elements = fs, es

    # Logical objects already have the required IDs; consolidation must not call
    # any per-object padded converter.
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy consolidation entered compact workflow")

    monkeypatch.setattr(dfn, "consolidate_dfn", forbidden)
    dfn.solve(compact=True)
    dfn.check_boundary_conditions(8)
    for evaluate in (dfn.calc_head, dfn.calc_w, dfn.calc_velocity):
        evaluate(1.3 + 0.7j, fs[0])
    dfn.fractures, dfn.elements = objects()
    allocators = (np.zeros, np.empty)
    legacy_dtypes = (hs.dtype_work, hs.dtype_z_arrays)
    from andfn.element import element_dtype_hpc, fracture_dtype_hpc

    for name, original in zip(("zeros", "empty"), allocators):

        def guarded(*args, _original=original, **kwargs):
            dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
            if dtype is not None and np.dtype(dtype) in (
                *legacy_dtypes,
                element_dtype_hpc,
                fracture_dtype_hpc,
            ):
                raise AssertionError("Padded record allocated")
            return _original(*args, **kwargs)

        monkeypatch.setattr(np, name, guarded)
    dfn.solve(compact=True)
    errors, index = dfn.check_boundary_conditions(8)
    assert errors.shape == (len(es), 7)
    assert index == np.argmax(errors[:, 0])
    assert np.isfinite(dfn.calc_head(1.3 + 0.7j, fs[0]))
    assert np.isfinite(dfn.calc_w(1.3 + 0.7j, fs[0]))
    assert np.isfinite(dfn.calc_velocity(1.3 + 0.7j, fs[0]))
    monkeypatch.setattr(np, "zeros", allocators[0])
    monkeypatch.setattr(np, "empty", allocators[1])
    heads, points = dfn._head_samples(np.array([0.1 + 0.2j]))
    assert heads.shape == (2, 1)
    assert points.shape == (2, 1, 3)
    assert np.isfinite(dfn.get_dfn_discharge())
    dfn.unconsolidate_dfn(hpc=True)
    assert dfn.elements_struc_array_hpc is None
    assert dfn.fractures_struc_array_hpc is None


def test_storage_constants_do_not_affect_compact_payload_or_results(monkeypatch):
    from andfn import element

    results = []
    constants = Constants().constants.copy()
    for nc, ne in ((200, 500), (500, 5000), (1000, 5000)):
        monkeypatch.setattr(element, "MAX_NCOEF", nc)
        monkeypatch.setattr(element, "MAX_ELEMENTS", ne)
        monkeypatch.setattr(hs, "MAX_NCOEF", nc)
        monkeypatch.setattr(hs, "MAX_ELEMENTS", ne)
        fs, es = objects()
        owner, fractures = compact.consolidate(es, fs)
        storage = owner.state.storage
        work = cw.boundary_work(storage.elements, 8, owner.state.work)
        error = hs.compute_bnd_error(
            fractures,
            storage.elements,
            8,
            constants,
            storage.coef_data,
            storage.coef_start,
            work=owner.state.work,
        )
        results.append(
            (
                cw.state_payload(owner.state),
                cw.payload(work),
                sum(a.nbytes for a in fractures),
                error,
            )
        )
    for value in results[1:]:
        assert value[:3] == results[0][:3]
        np.testing.assert_array_equal(value[3], results[0][3])


def test_boundary_overflow_uses_actual_grid_and_does_not_change_solver(monkeypatch):
    f, e = network()
    owner = adaptive.adaptive_from_hpc(e)
    storage = owner.state.storage
    before = {key: value.copy() for key, value in cw.state_arrays(owner.state).items()}
    original = hs.boundary_work

    def zero_capacity(*args):
        work = original(*args)
        return work._replace(
            branch_capacity=np.zeros(e.size, dtype=np.int64),
            branch_start=np.zeros(e.size, dtype=np.int64),
            **{name: np.empty(0, dtype=np.int64) for name in cw.BRANCH_FIELDS},
        )

    monkeypatch.setattr(hs, "boundary_work", zero_capacity)
    work, coordinates, stats = hs.prepare_boundary_work(
        bundle(f, e), storage.elements, 32, owner.state.work
    )
    assert stats["branch_overflow_elements"] > 0
    assert stats["branch_repacks"] == 1
    legacy = np.zeros(e.size, dtype=hs.dtype_work)
    for i in np.flatnonzero(e["_type"] == 1):
        hs.mf.find_branch_cuts(e[i], coordinates[0, i], f, e, legacy[i], 32)
        row = cw.work_record(work, i)
        count = legacy[i]["len_discharge_element"]
        assert row["len_discharge_element"] == count
        for name in cw.BRANCH_FIELDS:
            np.testing.assert_array_equal(row[name][:count], legacy[i][name][:count])
    for key, value in cw.state_arrays(owner.state).items():
        np.testing.assert_array_equal(value, before[key])


def test_actual_public_model_matches_legacy_and_compact_postprocessing():
    from andfn import ConstantHeadLine, Fracture

    models = []
    for is_compact in (False, True):
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
            ConstantHeadLine(
                str(x), np.array([x - 0.1j, x + 0.1j]), head, fracture, 8, 16
            )
        dfn = DFN("real", discharge_int=8, MAX_ITERATIONS=6)
        dfn.add_fracture(fracture)
        dfn.check_connectivity(compact=is_compact)
        dfn.solve(compact=is_compact)
        models.append(dfn)
    legacy, compact_dfn = models
    f = legacy.fractures[0]
    points = np.array([0.1 + 0.2j, 0.5 + 0.3j])
    for fn in ("calc_head", "calc_w", "calc_velocity"):
        for point in points:
            np.testing.assert_array_equal(
                getattr(legacy, fn)(point, f),
                getattr(compact_dfn, fn)(point, compact_dfn.fractures[0]),
            )
    np.testing.assert_array_equal(
        legacy.check_boundary_conditions(8)[0],
        compact_dfn.check_boundary_conditions(8)[0],
    )
    for a, b in zip(legacy._head_samples(points), compact_dfn._head_samples(points)):
        np.testing.assert_array_equal(a, b)
    state = compact_dfn._compact_owner.state.storage
    args = (
        compact_dfn._compact_fractures,
        2,
        2,
        state.elements,
        state.coef_data,
        state.coef_start,
    )
    expected = hf.get_flow_nets(
        legacy.fractures_struc_array_hpc, 2, 2, legacy.elements_struc_array_hpc
    )
    actual = hf.get_flow_nets(*args)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_compact_constructors_do_not_import_storage_maxima():
    import ast
    import inspect

    from andfn.hpc import hpc_connectivity, hpc_quadrature

    for module in (compact, cw, hpc_connectivity, hpc_quadrature):
        tree = ast.parse(inspect.getsource(module))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.alias)
        )
        names.update(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )
        assert not names & {"MAX_NCOEF", "MAX_ELEMENTS", "MAX_DISCHARGE_INT"}
