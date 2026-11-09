"""Exact ragged event order, capacity publication and numerical isolation."""

import numpy as np
import pytest
from legacy_hpc_reference import (
    adaptive_from_hpc,
    dtype_work,
    export_work,
)
from test_hpc_coefficients import network

from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_branch as branch
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_work as cw


def snapshot(state):
    return {key: (a, a.copy()) for key, a in cw.state_arrays(state).items()}


def assert_unchanged(state, before):
    assert cw.state_arrays(state).keys() == before.keys()
    for key, (original, values) in before.items():
        assert cw.state_arrays(state)[key] is original
        np.testing.assert_array_equal(original, values)
    adaptive.validate_state(state, finite=True)


def test_exact_count_pass_and_event_order_match_padded_reference():
    fractures, elements = network()
    owner = adaptive_from_hpc(elements)
    state = owner.state
    ids = np.arange(elements.size, dtype=np.int64)
    counts = branch.count_events(
        state.storage.elements, fractures, ids, elements["nint"]
    )
    work = state.work
    work = work._replace(
        branch_start=np.zeros(elements.size, dtype=np.int64),
        branch_capacity=np.zeros(elements.size, dtype=np.int64),
        **{name: np.zeros(0, dtype=np.int64) for name in cw.BRANCH_FIELDS},
    )
    prepared, stats = branch.prepare_branches(
        work, state.storage.elements, fractures, ids, elements["nint"]
    )
    assert stats["branch_repacks"] == 1
    assert stats["branch_event_bytes_copied"] == 0
    assert np.all(prepared.branch_capacity >= counts)
    # Count-only passes may not consume or publish the event buffers.
    np.testing.assert_array_equal(prepared.len_discharge_element, 0)
    for i, element in enumerate(elements):
        if element["_type"] not in (1, 4, 5):
            continue
        legacy = np.zeros(1, dtype=dtype_work)
        mf.fill_exp_array(
            element["nint"], element["thetas"], legacy[0]["exp_array_p"], 1
        )
        # Match the production preparation's grid, not uninitialized input theta.
        theta = np.empty(element["nint"])
        mf.calc_thetas(element["nint"], element["_type"], theta)
        record = cw.work_record(prepared, i)
        mf.fill_exp_array(element["nint"], theta, legacy[0]["exp_array_p"], 1)
        mf.fill_exp_array(element["nint"], theta, record["exp_array_p"], 1)
        mf.get_dpsi_corr(element, fractures, elements, legacy[0])
        mf.get_dpsi_corr(
            state.storage.elements[i], fractures, state.storage.elements, record
        )
        assert legacy[0]["len_discharge_element"] == counts[i]
        assert record["len_discharge_element"] == counts[i]
        for name in cw.BRANCH_FIELDS:
            np.testing.assert_array_equal(
                record[name][: counts[i]], legacy[0][name][: counts[i]]
            )
    adaptive.validate_state(
        adaptive.AdaptiveState(state.storage, prepared), finite=True
    )
    reused, stats = branch.prepare_branches(
        prepared, state.storage.elements, fractures, ids, elements["nint"]
    )
    assert reused is prepared
    assert stats["branch_event_bytes_copied"] == 0


def test_branch_preparation_failure_is_before_growth_publication(monkeypatch):
    fractures, elements = network()
    owner = adaptive_from_hpc(elements)
    adaptive.adaptive_step(owner, fractures, 1e-12, 1, 0, -1.0, 12, 2, 0.5)
    observed = {}

    def capture(label, state, requests):
        if label == "after_assessment":
            observed["state"] = state
            observed["before"] = snapshot(state)

    def failing(*args):
        raise MemoryError("injected branch preparation")

    owner.transition_callback = capture
    monkeypatch.setattr(adaptive, "prepare_branches", failing)
    with pytest.raises(MemoryError, match="branch preparation"):
        adaptive.adaptive_step(owner, fractures, 1e-12, 2, 0, -1.0, 12, 2, 0.5)
    assert owner.state is observed["state"]
    assert_unchanged(owner.state, observed["before"])


def test_repeated_crossings_exceed_legacy_padding_and_keep_position_order():
    fractures, elements = network()
    elements["center"][4], elements["radius"][4] = 0, 1
    elements["center"][2] = 2 + 0.5j
    owner = adaptive_from_hpc(elements)
    state = owner.state
    small = fractures.copy()
    small["nelements"] = 1
    small["elements"][:, 0] = 2
    record = np.zeros(1, dtype=dtype_work)
    theta = np.empty(elements[4]["nint"])
    mf.calc_thetas(theta.size, elements[4]["_type"], theta)
    mf.fill_exp_array(theta.size, theta, record[0]["exp_array_p"], 1)
    mf.get_dpsi_corr(elements[4], small, elements, record[0])
    assert record[0]["len_discharge_element"] == 2
    dense = fractures.copy()
    dense["nelements"] = dense.dtype["elements"].shape[0]
    dense["elements"][:] = 2
    repetitions = int(dense["nelements"][0])
    ids = np.array([4], dtype=np.int64)
    prepared, stats = branch.prepare_branches(
        state.work, state.storage.elements, dense, ids, elements["nint"]
    )
    assert stats["branch_repacks"] == 1
    local = cw.work_record(prepared, 4)
    mf.fill_exp_array(theta.size, theta, local["exp_array_p"], 1)
    mf.get_dpsi_corr(state.storage.elements[4], dense, state.storage.elements, local)
    assert local["len_discharge_element"] == 2 * repetitions
    assert local["len_discharge_element"] > dtype_work["sign_array"].shape[0]
    for name in cw.BRANCH_FIELDS:
        np.testing.assert_array_equal(
            local[name][: 2 * repetitions], np.repeat(record[0][name][:2], repetitions)
        )
    adaptive.validate_state(
        adaptive.AdaptiveState(state.storage, prepared), finite=True
    )
    with pytest.raises(ValueError, match="exceed legacy"):
        export_work(prepared, dtype_work, include_scratch=False)


def test_z_integral_is_retained_for_diagnostics_not_numerical_input():
    fractures, elements = network()
    owners = [adaptive_from_hpc(elements) for _ in range(2)]
    owners[1].state.work.z_integral[:] = 1234 + 56j
    for nit in range(1, 4):
        for owner in owners:
            adaptive.adaptive_step(owner, fractures, 1e-12, nit, 0, 1e100, 12, 2, 0.5)
        a, b = [owner.state for owner in owners]
        np.testing.assert_array_equal(a.storage.elements, b.storage.elements)
        np.testing.assert_array_equal(a.storage.coef_data, b.storage.coef_data)
        np.testing.assert_array_equal(
            a.storage.candidate_coef_data, b.storage.candidate_coef_data
        )
    assert not np.array_equal(a.work.z_integral, b.work.z_integral)
    assert (
        "z_integral"
        in export_work(a.work, dtype_work, include_scratch=False).dtype.names
    )


@pytest.mark.parametrize("fail", [False, True])
def test_adaptive_branch_repack_is_private_until_publication(fail):
    fractures, elements = network()
    elements["center"][4], elements["radius"][4] = 0, 1
    elements["center"][2] = 2 + 0.5j
    elements["ncoef"][4], elements["nint"][4] = 2, 2
    elements["coef"][4, 2:] = 0
    fractures["nelements"] = fractures.dtype["elements"].shape[0]
    fractures["elements"][:] = 2
    owner = adaptive_from_hpc(elements)
    adaptive.adaptive_step(owner, fractures, 1e-12, 1, 0, -1.0, 4, 2, 0.5)
    old = owner.state
    old_branch = tuple(getattr(old.work, name) for name in cw.BRANCH_FIELDS)
    observed = {}

    def observe(label, state, requests):
        if label == "after_assessment":
            observed["snapshot"] = snapshot(state)

    def abort(state):
        assert state is old
        assert_unchanged(state, observed["snapshot"])
        raise RuntimeError("abort branch repack publication")

    owner.transition_callback = observe
    if fail:
        owner.before_publication = abort
        with pytest.raises(RuntimeError, match="branch repack publication"):
            adaptive.adaptive_step(owner, fractures, 1e-12, 2, 0, -1.0, 4, 2, 0.5)
        assert owner.state is old
        assert_unchanged(old, observed["snapshot"])
    else:
        adaptive.adaptive_step(owner, fractures, 1e-12, 2, 0, -1.0, 4, 2, 0.5)
        new = owner.state
        expected_events = 2 * fractures["nelements"][0]
        assert new.work.len_discharge_element[4] == expected_events
        assert new.work.branch_capacity[4] >= expected_events
        for before, name in zip(old_branch, cw.BRANCH_FIELDS):
            assert getattr(new.work, name) is not before
        assert owner.batches[-1]["branch_repacks"] == 1
        assert owner.batches[-1]["work_bytes_copied"] == 0
        adaptive.validate_state(new, finite=True)
