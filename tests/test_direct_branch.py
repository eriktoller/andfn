"""Direct event preparation matches both references and never exposes partial writes."""

import numba as nb
import numpy as np
import pytest
from test_compact_branch import assert_unchanged, snapshot
from test_hpc_coefficients import network

from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_branch as branch
from andfn.hpc import hpc_work as cw


def fixture(overflow=False):
    fractures, elements = network()
    if overflow:
        elements["center"][4], elements["radius"][4] = 0, 1
        elements["center"][2] = 2 + 0.5j
        fractures["nelements"] = fractures.dtype["elements"].shape[0]
        fractures["elements"][:] = 2
    owner = adaptive.adaptive_from_hpc(elements)
    return fractures, owner.state


@pytest.mark.parametrize("threads", [1, 2, nb.config.NUMBA_NUM_THREADS])
@pytest.mark.parametrize("overflow", [False, True])
def test_direct_exact_count_and_legacy_events_agree(threads, overflow):
    from andfn.hpc import hpc_math_functions as mf
    from andfn.hpc import hpc_solve as hs

    previous = nb.get_num_threads()
    nb.set_num_threads(threads)
    try:
        fractures, state = fixture(overflow)
        e, w = state.storage.elements, state.work
        legacy_elements = adaptive.to_hpc(state)
        ids = np.arange(e.size, dtype=np.int64)
        before = snapshot(state)
        prepared, stats = branch.prepare_direct(w, e, fractures, ids, e["nint"])
        assert_unchanged(state, before)
        counts = branch.count_events(e, fractures, ids, e["nint"])
        np.testing.assert_array_equal(prepared.counts, counts)
        assert (
            stats["branch_overflow_elements"] > 0
            if overflow
            else stats["branch_repacks"] == 0
        )
        reference, _ = branch.prepare_branches(w, e, fractures, ids, e["nint"])
        for i in ids:
            if e[i]["_type"] not in (1, 4, 5):
                continue
            record = cw.work_record(reference, i)
            theta = np.empty(e["nint"][i])
            mf.calc_thetas(theta.size, e["_type"][i], theta)
            mf.fill_exp_array(theta.size, theta, record["exp_array_p"], 1)
            mf.get_dpsi_corr(e[i], fractures, e, record)
            a, n = prepared.starts[i], counts[i]
            for name, value in zip(cw.BRANCH_FIELDS, prepared.values):
                np.testing.assert_array_equal(value[a : a + n], record[name][:n])
            if n <= hs.dtype_work["sign_array"].shape[0]:
                legacy = np.zeros(1, dtype=hs.dtype_work)
                mf.fill_exp_array(theta.size, theta, legacy[0]["exp_array_p"], 1)
                mf.get_dpsi_corr(
                    legacy_elements[i], fractures, legacy_elements, legacy[0]
                )
                for name, value in zip(cw.BRANCH_FIELDS, prepared.values):
                    np.testing.assert_array_equal(value[a : a + n], legacy[0][name][:n])
        branch.publish_direct(
            prepared.work, ids, prepared.starts, prepared.counts, prepared.values
        )
        assert np.all(prepared.work.branch_valid)
        if not overflow:
            assert prepared.work is w
            for name in cw.BRANCH_FIELDS:
                assert getattr(prepared.work, name) is getattr(w, name)
        adaptive.validate_state(
            adaptive.AdaptiveState(state.storage, prepared.work), finite=True
        )
    finally:
        nb.set_num_threads(previous)


@pytest.mark.parametrize(
    "failure", ["construct", "position", "id", "allocate", "reconstruct", "validate"]
)
def test_direct_failures_preserve_published_state(monkeypatch, failure):
    fractures, state = fixture(failure in ("allocate", "reconstruct"))
    e, w = state.storage.elements, state.work
    ids = np.arange(e.size, dtype=np.int64)
    # Publish a small, valid old topology before attempting replacement.
    w.len_discharge_element[0] = 1
    w.sign_array[0], w.discharge_element[0], w.element_pos[0] = 1, 2, 0
    w.branch_valid[0] = True
    before = snapshot(state)
    construct, allocate, validate = (
        branch.construct_events,
        branch.allocate_events,
        branch.validate_events,
    )
    calls = []

    def injected_construct(*args):
        calls.append(args[3].copy())
        if failure == "construct" or (failure == "reconstruct" and len(calls) == 2):
            raise RuntimeError("injected construction")
        construct(*args)
        if failure == "id":
            args[-2][0] = e.size
        if failure == "position":
            args[-1][0] = np.nan

    def injected_allocate(size):
        if failure == "allocate" and calls:
            raise MemoryError("injected replacement allocation")
        values = allocate(size)
        if failure == "position":
            return values[:2] + (np.empty(size, dtype=float),)
        return values

    def injected_validate(*args):
        if failure == "validate":
            raise ValueError("injected prepublication validation")
        validate(*args)

    monkeypatch.setattr(branch, "construct_events", injected_construct)
    monkeypatch.setattr(branch, "allocate_events", injected_allocate)
    monkeypatch.setattr(branch, "validate_events", injected_validate)
    with pytest.raises((ValueError, RuntimeError, MemoryError)):
        branch.prepare_direct(w, e, fractures, ids, e["nint"])
    assert_unchanged(state, before)
    if failure == "reconstruct":
        assert calls[1].size < calls[0].size


def test_default_solver_never_calls_exact_count(monkeypatch):
    fractures, elements = network()
    owner = adaptive.adaptive_from_hpc(elements)

    def forbidden(*args):
        raise AssertionError("exact count must not execute")

    monkeypatch.setattr(branch, "count_events", forbidden)
    for nit in range(1, 5):
        adaptive.adaptive_step(owner, fractures, 1e-12, nit, 0, -1.0, 12, 2, 0.5)
        adaptive.validate_state(owner.state, finite=True)


def test_empty_topology_is_valid_but_grid_change_invalidates():
    fractures, state = fixture()
    e = state.storage.elements
    ids = np.arange(e.size, dtype=np.int64)
    prepared, _ = branch.prepare_direct(state.work, e, fractures, ids, e["nint"])
    branch.publish_direct(
        prepared.work, ids, prepared.starts, prepared.counts, prepared.values
    )
    empty = np.flatnonzero(prepared.counts == 0)
    assert empty.size
    for i in empty:
        record = cw.work_record(prepared.work, i)
        assert cw.branch_topology_valid(record)
        cw.invalidate_branch(record)
        assert not cw.branch_topology_valid(record)


@pytest.mark.parametrize("overflow", [False, True])
def test_only_affected_topology_is_published(overflow):
    fractures, state = fixture(overflow)
    work, elements = state.work, state.storage.elements
    work.len_discharge_element[0] = 1
    work.branch_valid[0] = True
    for name, value in zip(cw.BRANCH_FIELDS, (1, 2, 0)):
        getattr(work, name)[0] = value
    before = snapshot(state)
    ids = np.array([4], dtype=np.int64)
    prepared, stats = branch.prepare_direct(
        work, elements, fractures, ids, elements["nint"]
    )
    assert_unchanged(state, before)
    branch.publish_direct(
        prepared.work, ids, prepared.starts, prepared.counts, prepared.values
    )
    for name, value in zip(cw.BRANCH_FIELDS, (1, 2, 0)):
        assert getattr(prepared.work, name)[prepared.work.branch_start[0]] == value
    assert prepared.work.len_discharge_element[0] == 1
    assert prepared.work.branch_valid[0]
    assert stats["branch_repacks"] == int(overflow)
    adaptive.validate_state(
        adaptive.AdaptiveState(state.storage, prepared.work), finite=True
    )


def test_readonly_publication_destination_is_rejected_before_construction(monkeypatch):
    fractures, state = fixture()
    before = snapshot(state)
    state.work.branch_valid.flags.writeable = False

    def forbidden(*args):
        raise AssertionError("construction must not run")

    monkeypatch.setattr(branch, "construct_events", forbidden)
    with pytest.raises(ValueError, match="writable"):
        branch.prepare_direct(
            state.work,
            state.storage.elements,
            fractures,
            np.arange(state.storage.elements.size, dtype=np.int64),
            state.storage.elements["nint"],
        )
    state.work.branch_valid.flags.writeable = True
    assert_unchanged(state, before)


def test_discharge_change_only_updates_event_values(monkeypatch):
    fractures, elements = network()
    owner = adaptive.adaptive_from_hpc(elements)
    adaptive.adaptive_step(owner, fractures, 1e-12, 1, 0, 1e100, 0, 2, 0.5)
    work = owner.state.work
    topology = {
        name: getattr(work, name).copy()
        for name in cw.BRANCH_FIELDS + ("len_discharge_element", "branch_valid")
    }

    def forbidden(*args):
        raise AssertionError("discharge alone must not reconstruct topology")

    monkeypatch.setattr(adaptive, "prepare_branches", forbidden)
    owner.state.storage.elements["q"] *= 2
    adaptive.adaptive_step(owner, fractures, 1e-12, 2, 0, 1e100, 0, 2, 0.5)
    for name, value in topology.items():
        np.testing.assert_array_equal(getattr(work, name), value)
