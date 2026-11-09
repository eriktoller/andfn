"""Adaptive trajectory, barrier, cache and transaction regressions."""

import numba as nb
import numpy as np
import pytest
from legacy_hpc_reference import (
    adaptive_from_hpc,
    connectivity_from_hpc,
    dtype_work,
    export_work,
    fixed_order_from_hpc,
    fracture_metadata_from_hpc,
    import_work,
    to_hpc,
)
from test_hpc_coefficients import (
    assert_close,
    legacy_logical,
    logical,
    network,
    work_for,
)

from andfn.constants import Constants
from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_solve as hs
from andfn.hpc import hpc_work as cw
from andfn.hpc.hpc_connectivity import FractureArrays
from andfn.storage import _layout


def prepared():
    f, e = network()
    # Independent integration lengths must survive until an element grows.
    e["nint"][[0, 3]] = [30, 17]
    w = work_for(e)
    owner = adaptive_from_hpc(e)
    import_work(owner.state.work, w, e)
    return f, e, w, owner


def snapshot(state):
    s = state.storage
    arrays = {name: value for name, value in vars(s).items()}
    arrays.update({f"work.{k}": a for k, a in zip(cw.WorkArrays._fields, state.work)})
    return {name: (value, value.copy()) for name, value in arrays.items()}


def assert_unchanged(state, before):
    actual = {
        **vars(state.storage),
        **{f"work.{k}": a for k, a in zip(cw.WorkArrays._fields, state.work)},
    }
    for name, (original, copy) in before.items():
        assert actual[name] is original
        np.testing.assert_array_equal(actual[name], copy)


def requests(owner, ids=(0, 1, 3), increase=2):
    e = owner.state.storage.elements
    nc, ni = e["ncoef"].copy(), e["nint"].copy()
    mask = np.zeros(e.size, dtype=bool)
    mask[list(ids)] = True
    nc[mask] += increase
    ni[mask] = 2 * nc[mask]
    return nc, ni, mask


@pytest.mark.parametrize(
    "failure",
    [
        "negative_coef",
        "shrinking_coef",
        "negative_nint",
        "wrong_nint",
        "wrong_increment",
        "maximum",
        "invalid_id",
        "overflow",
        "allocation",
        "cache",
        "nonfinite_cache",
    ],
)
def test_transaction_failure_preserves_every_published_array(failure, monkeypatch):
    _, _, _, owner = prepared()
    state = owner.state
    nc, ni, mask = requests(owner)
    maximum = 20
    if failure == "negative_coef":
        nc[0] = -1
    elif failure == "shrinking_coef":
        nc[0] = 3
    elif failure == "negative_nint":
        ni[0] = -1
    elif failure == "wrong_nint":
        ni[0] += 1
    elif failure == "wrong_increment":
        nc[0] += 1
    elif failure == "maximum":
        maximum = 4
    elif failure == "invalid_id":
        state.storage.elements["_id"][0] = -1
    elif failure == "overflow":
        monkeypatch.setattr(
            adaptive,
            "_layout",
            lambda _: (_ for _ in ()).throw(ValueError("simulated offset overflow")),
        )
    elif failure == "allocation":
        monkeypatch.setattr(
            adaptive.np,
            "zeros",
            lambda *a, **k: (_ for _ in ()).throw(
                MemoryError("injected coefficient allocation failure")
            ),
        )

    def rebuild(e, w, required):
        adaptive.rebuild_growth_caches(e, w, required)
        if failure == "cache":
            raise RuntimeError("injected cache failure")
        if failure == "nonfinite_cache":
            w.exp_array_p[0] = np.nan

    before = snapshot(state)
    with pytest.raises((ValueError, MemoryError, RuntimeError)):
        adaptive.build_replacement(state, nc, ni, mask, maximum, 2, rebuild=rebuild)
    assert owner.state is state
    assert_unchanged(state, before)


def test_no_growth_transaction_is_identity_and_has_no_repack():
    _, _, _, owner = prepared()
    before = snapshot(owner.state)
    replacement, stats = adaptive.build_replacement(
        owner.state, *requests(owner, ids=()), 20, 2
    )
    assert replacement is owner.state
    assert stats is None
    assert_unchanged(owner.state, before)


@pytest.mark.parametrize("increase", [1, 2, 5])
def test_replacement_preserves_prefixes_zeroes_new_entries_and_rebuilds_cache(increase):
    _, _, _, owner = prepared()
    old = owner.state
    for name in ("coef_data", "candidate_coef_data", "old_coef_data"):
        getattr(old.storage, name)[:] += 0.125
    req = requests(owner, increase=increase)
    old_lengths = old.storage.elements["ncoef"].copy()
    before = snapshot(old)
    transaction, stats = adaptive.build_replacement(old, *req, 20, increase)
    assert_unchanged(old, before)
    replacement = adaptive.publish(transaction)
    s = replacement.storage
    for i, previous in enumerate(old_lengths):
        a, b = old.storage.coef_start[i], s.coef_start[i]
        for name in ("coef_data", "candidate_coef_data", "old_coef_data"):
            assert_close(
                getattr(s, name)[b : b + previous],
                getattr(old.storage, name)[a : a + previous],
            )
            np.testing.assert_array_equal(
                getattr(s, name)[b + previous : b + s.elements["ncoef"][i]], 0
            )
    assert s.elements["nint"][0] == 2 * (4 + increase)
    assert s.elements["nint"][4] == old.storage.elements["nint"][4]
    np.testing.assert_array_equal(replacement.work.len_discharge_element[req[2]], 0)
    for i in np.flatnonzero(req[2]):
        n = s.elements["nint"][i]
        theta = np.zeros(n)
        mf.calc_thetas(n, s.elements["_type"][i], theta)
        assert_close(cw.work_record(replacement.work, i)["thetas"][:n], theta)
        assert_close(
            cw.work_record(replacement.work, i)["exp_array_p"][:n], np.exp(1j * theta)
        )
    assert stats["work_bytes_copied"] == 0
    assert stats["coefficient_repacks"] == int(
        np.any(s.coef_capacity != old.storage.coef_capacity)
    )


def test_growth_within_spare_capacity_and_ignored_nonzero_slack():
    _, _, _, owner = prepared()
    s = owner.state.storage
    capacities = s.coef_capacity + 8
    starts, size = _layout(capacities)
    arrays = [np.full(size, 1e99, dtype=complex) for _ in range(3)]
    for i, n in enumerate(s.elements["ncoef"]):
        for source, target in zip(
            (s.coef_data, s.candidate_coef_data, s.old_coef_data), arrays
        ):
            target[starts[i] : starts[i] + n] = source[
                s.coef_start[i] : s.coef_start[i] + n
            ]
    state = adaptive.AdaptiveState(
        adaptive.AdaptiveStorage(s.elements.copy(), *arrays, starts, capacities),
        cw.allocate_work(s.elements, starts, capacities),
    )
    transaction, stats = adaptive.build_replacement(state, *requests(owner), 20, 2)
    new = adaptive.publish(transaction)
    np.testing.assert_array_equal(new.storage.coef_start, starts)
    np.testing.assert_array_equal(new.storage.coef_capacity, capacities)
    assert stats["coefficient_repacks"] == 0
    assert stats["coefficient_bytes_copied"] == 0
    assert stats["coefficient_relocation_bytes"] == 0
    assert stats["work_bytes_copied"] == 0
    assert new.storage.coef_data is state.storage.coef_data
    for i in (0, 1, 3):
        old_n = s.elements["ncoef"][i]
        np.testing.assert_array_equal(
            new.storage.coef_data[starts[i] + old_n : starts[i] + old_n + 2], 0
        )


def test_assessment_is_read_only_and_preserves_overshoot():
    _, _, _, owner = prepared()
    state = owner.state
    before = snapshot(state)
    nc, ni, mask = adaptive.assess_growth(
        state.storage.elements,
        state.storage.coef_data,
        state.storage.coef_start,
        2,
        1e-12,
        -1,
        5,
        3,
    )
    assert_unchanged(state, before)
    assert nc[0] == 7
    assert ni[0] == 14  # Approved logical shrink from independent nint=30.
    assert mask[0]
    assert not mask[2]
    assert adaptive.assess_growth.nopython_signatures


def legacy_transition(elements, fractures, work, maximum, increase, ratio, nit):
    history = {}

    def capture(name):
        history[name] = (
            elements.copy(),
            work["coef"].copy(),
            work.copy(),
        )

    capture("before_first_candidate")
    hs.calculate_candidates(
        elements.size, elements, fractures, work, 1e-12, nit, 0, 0.5
    )
    capture("after_first_candidate")
    hs.commit_coefficients(elements, work, 0.5)
    capture("after_first_commit")
    if nit < 2:
        return history
    # Assess from an immutable committed snapshot; verify final results against
    # the actual interleaved legacy loop separately below.
    s = fixed_order_from_hpc(elements)
    request = adaptive.assess_growth(
        elements, s.coef_data, s.coef_start, nit, 1e-12, ratio, maximum, increase
    )
    for i in np.flatnonzero(request[2]):
        elements["ncoef"][i] = request[0][i]
        elements["nint"][i] = request[1][i]
    adaptive.rebuild_growth_caches(elements, work, request[2])
    capture("after_growth_publication")
    hs.calculate_candidates(
        elements.size,
        elements,
        fractures,
        work,
        1e-12,
        nit,
        0,
        0.5,
        selected=request[2],
    )
    capture("after_adaptive_candidate")
    hs.commit_coefficients(elements, work, 0.5)
    capture("after_second_commit")
    return history


@pytest.mark.parametrize(
    "ratio,maximum,increase",
    [
        (1e100, 8, 2),
        (-1, 5, 1),
        (-1, 8, 2),
        (0.05, 10, 3),
    ],
)
def test_transition_snapshots_and_actual_legacy_loop(ratio, maximum, increase):
    f, e, w, owner = prepared()
    snapshots = {}

    def observe(label, state, requests):
        snapshots[label] = (
            to_hpc(state),
            state.storage.candidate_coef_data.copy(),
            state.storage.coef_start.copy(),
            requests,
            export_work(state.work, dtype_work, include_scratch=False),
        )

    owner.transition_callback = observe
    for nit in range(1, 5):
        reference = legacy_transition(e, f, w, maximum, increase, ratio, nit)
        adaptive.adaptive_step(owner, f, 1e-12, nit, 0, ratio, maximum, increase, 0.5)
        for label, (elements, candidates, reference_work) in reference.items():
            compact, flat, starts, _request, compact_work = snapshots[label]
            np.testing.assert_array_equal(compact["ncoef"], elements["ncoef"])
            np.testing.assert_array_equal(compact["nint"], elements["nint"])
            for name in (
                "coef",
                "old_coef",
                "error",
                "error_coef",
                "error_old",
                "error_old2",
            ):
                if name in ("coef", "old_coef"):
                    assert_close(
                        legacy_logical(compact, name), legacy_logical(elements, name)
                    )
                    np.testing.assert_array_equal(
                        legacy_logical(compact, name), legacy_logical(elements, name)
                    )
                else:
                    assert_close(compact[name], elements[name])
                    np.testing.assert_array_equal(compact[name], elements[name])
            for i, n in enumerate(elements["ncoef"]):
                assert_close(flat[starts[i] : starts[i] + n], candidates[i, :n])
                np.testing.assert_array_equal(
                    flat[starts[i] : starts[i] + n], candidates[i, :n]
                )
                assert_close(
                    compact_work["old_coef"][i, :n],
                    reference_work["old_coef"][i, :n],
                )
                np.testing.assert_array_equal(
                    compact_work["old_coef"][i, :n], reference_work["old_coef"][i, :n]
                )
                ni = elements["nint"][i]
                for name in ("thetas", "dpsi_corr"):
                    assert_close(compact[name][i, :ni], elements[name][i, :ni])
                    np.testing.assert_array_equal(
                        compact[name][i, :ni], elements[name][i, :ni]
                    )
                for name in ("exp_array_m", "exp_array_p", "z_integral"):
                    assert_close(
                        compact_work[name][i, :ni], reference_work[name][i, :ni]
                    )
                    np.testing.assert_array_equal(
                        compact_work[name][i, :ni], reference_work[name][i, :ni]
                    )
                events = reference_work["len_discharge_element"][i]
                assert compact_work["len_discharge_element"][i] == events
                for name in ("discharge_element", "element_pos", "sign_array"):
                    np.testing.assert_array_equal(
                        compact_work[name][i, :events],
                        reference_work[name][i, :events],
                    )


def run_adaptive(
    *,
    ratio=0.05,
    maximum=12,
    increase=2,
    iterations=12,
    request_history=None,
):
    f, e = network()
    e["nint"][[0, 3]] = [30, 17]
    owner = adaptive_from_hpc(e)
    if request_history is not None:

        def capture_requests(label, state, requests):
            if label == "after_assessment":
                request_history.append(tuple(x.copy() for x in requests))

        owner.transition_callback = capture_requests
    c = Constants().constants.copy()
    c["MAX_ITERATIONS"], c["MAX_ERROR"], c["DAMPING"] = iterations, 1e-12, 0.5
    c["MAX_COEFFICIENT_ORDER"], c["COEF_INCREASE"], c["COEF_RATIO"] = (
        maximum,
        increase,
        ratio,
    )
    history = []
    connectivity = connectivity_from_hpc(f, e.size)
    fractures = FractureArrays(
        fracture_metadata_from_hpc(f), connectivity.ids, connectivity.indptr
    )

    def observe(nit, elements, fractures, work, discharge, error_q, counter):
        s = owner.state.storage
        coef = logical(s, s.coef_data)
        candidate = logical(s, s.candidate_coef_data)
        old = logical(s, s.old_coef_data)
        kwargs = {"coef_data": s.coef_data, "coef_start": s.coef_start}
        heads = np.array(
            [
                hs.hpc_fracture.calc_omega(
                    frac,
                    1.3 + 0.7j,
                    elements,
                    element_ids=connectivity.element_ids(frac["_id"]),
                    **kwargs,
                ).real
                / frac["t"]
                for frac in fractures.metadata
            ]
        )
        boundary = hs.compute_bnd_error(fractures, elements, 8, c, **kwargs)
        dfn_q = np.sum(np.abs(elements["q"][np.isin(elements["_type"], (2, 3))])) / 2
        history.append(
            (
                nit,
                counter,
                elements["ncoef"].copy(),
                elements["nint"].copy(),
                coef.copy(),
                candidate.copy(),
                old.copy(),
                elements["error"].copy(),
                elements["error_coef"].copy(),
                fractures.metadata["constant"].copy(),
                discharge.copy(),
                dfn_q,
                heads,
                boundary,
                error_q,
            )
        )

    adaptive.solve_adaptive(owner, fractures, 8, c, iteration_callback=observe)
    return history, owner


@pytest.mark.parametrize("ratio", [0.05, -1.0, 1e100])
def test_adaptive_iteration_trajectory_thread_equivalence_and_repeatability(ratio):
    normal = nb.get_num_threads()
    reference = None
    reference_requests = None
    try:
        for threads in sorted({1, 2, normal}):
            nb.set_num_threads(threads)
            for _ in range(2):
                growth_requests = []
                history, owner = run_adaptive(
                    ratio=ratio, request_history=growth_requests
                )
                if reference is not None:
                    assert len(history) == len(reference)
                    for expected, actual in zip(reference, history):
                        assert expected[:2] == actual[:2]
                        for index, (x, y) in enumerate(zip(expected[2:], actual[2:])):
                            assert_close(x, y)
                            if index < 2:
                                np.testing.assert_array_equal(x, y)
                owner.state.storage.validate()
                s = owner.state.storage
                assert np.all(
                    s.coef_start[1:] >= (s.coef_start[:-1] + s.coef_capacity[:-1])
                )
                if reference_requests is not None:
                    assert len(reference_requests) == len(growth_requests)
                    for a, b in zip(reference_requests, growth_requests):
                        for x, y in zip(a, b):
                            np.testing.assert_array_equal(x, y)
                for i, (nc, ni, mask) in enumerate(growth_requests, start=1):
                    np.testing.assert_array_equal(nc, history[i][2])
                    np.testing.assert_array_equal(ni, history[i][3])
                    np.testing.assert_array_equal(mask, nc != history[i - 1][2])
                reference = history
                reference_requests = growth_requests
    finally:
        nb.set_num_threads(normal)


def test_offset_prefix_overflow_checked_without_allocation():
    with pytest.raises(ValueError, match="offset"):
        _layout(np.array([np.iinfo(np.int64).max, 1], dtype=np.int64))


def test_single_growing_element_and_unaffected_cache_state():
    _, _, _, owner = prepared()
    before = snapshot(owner.state)
    required = requests(owner, ids=(0,), increase=1)
    transaction, stats = adaptive.build_replacement(owner.state, *required, 5, 1)
    assert_unchanged(owner.state, before)
    replacement = adaptive.publish(transaction)
    assert stats["expanded_elements"] == 1
    unchanged = ~required[2]
    for name in ("thetas", "dpsi_corr"):
        for i in np.flatnonzero(unchanged):
            np.testing.assert_array_equal(
                cw.work_record(replacement.work, i)[name],
                cw.work_record(owner.state.work, i)[name],
            )
    np.testing.assert_array_equal(
        export_work(replacement.work, dtype_work, include_scratch=False)[unchanged],
        export_work(owner.state.work, dtype_work, include_scratch=False)[unchanged],
    )


@pytest.mark.parametrize("increase", [0, -1, np.iinfo(np.int64).max // 2])
def test_request_arithmetic_overflow_rejected_before_assessment(increase):
    _, _, _, owner = prepared()
    before = snapshot(owner.state)
    s = owner.state.storage
    with pytest.raises(ValueError, match="overflow"):
        adaptive.assess_growth(
            s.elements, s.coef_data, s.coef_start, 2, 1e-12, -1, 5, increase
        )
    assert_unchanged(owner.state, before)


@pytest.mark.parametrize("spare,previous_nint", [(False, 12), (True, 30), (False, 30)])
def test_coefficient_and_integration_growth_bridge_cases(spare, previous_nint):
    _, _, _, owner = prepared()
    s = owner.state.storage
    s.elements["nint"][0] = previous_nint
    if spare:
        capacities = s.coef_capacity.copy()
        capacities[0] += 8
        starts, size = _layout(capacities)
        arrays = [np.zeros(size, dtype=complex) for _ in range(3)]
        for i, n in enumerate(s.elements["ncoef"]):
            for name, target in zip(
                ("coef_data", "candidate_coef_data", "old_coef_data"), arrays
            ):
                target[starts[i] : starts[i] + n] = getattr(s, name)[
                    s.coef_start[i] : s.coef_start[i] + n
                ]
        owner.state = adaptive.AdaptiveState(
            adaptive.AdaptiveStorage(s.elements.copy(), *arrays, starts, capacities),
            cw.allocate_work(s.elements, starts, capacities),
        )
    transaction, stats = adaptive.build_replacement(
        owner.state, *requests(owner, ids=(0,), increase=2), 20, 2
    )
    state = adaptive.publish(transaction)
    assert stats["coefficient_repacks"] == int(not spare)
    assert stats["work_bytes_copied"] == 0
    assert state.storage.elements["nint"][0] == 12
    np.testing.assert_array_equal(
        state.storage.elements["nint"][1:], owner.state.storage.elements["nint"][1:]
    )
