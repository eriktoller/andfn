"""Compact ownership and changed-state transaction-cost contracts."""

import numba as nb
import numpy as np
import pytest
from test_hpc_coefficients import network, work_for

from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_solve as hs
from andfn.hpc import hpc_work as cw
from andfn.storage import _layout


def fixture(coef_repack=False, integration_repack=False):
    fractures, elements = network()
    elements["ncoef"][elements["_type"] != 2] = 10
    elements["nint"][elements["_type"] != 2] = 20
    legacy = work_for(elements)
    owner = adaptive.adaptive_from_hpc(elements)
    state = owner.state
    c = state.storage.coef_capacity.copy()
    c[elements["_type"] != 2] = 10 if coef_repack else 16
    starts, size = _layout(c)
    buffers = [np.zeros(size, dtype=complex) for _ in range(3)]
    for i, n in enumerate(elements["ncoef"]):
        for source, target in zip(
            (
                state.storage.coef_data,
                state.storage.candidate_coef_data,
                state.storage.old_coef_data,
            ),
            buffers,
        ):
            target[starts[i] : starts[i] + n] = source[
                state.storage.coef_start[i] : state.storage.coef_start[i] + n
            ]
    work = cw.allocate_work(elements, starts, c)
    if not integration_repack:
        cap = work.integration_capacity.copy()
        cap[elements["_type"] != 2] = 32
        ip, length = _layout(cap)
        work = work._replace(
            **{
                name: np.zeros(length, dtype=getattr(work, name).dtype)
                for name in cw.INTEGRATION_FIELDS
            },
            integration_start=ip,
            integration_capacity=cap,
        )
    cw.import_work(work, legacy, elements)
    owner.state = adaptive.AdaptiveState(
        adaptive.AdaptiveStorage(
            elements=state.storage.elements,
            coef_data=buffers[0],
            candidate_coef_data=buffers[1],
            old_coef_data=buffers[2],
            coef_start=starts,
            coef_capacity=c,
        ),
        work,
    )
    return fractures, owner


@pytest.mark.parametrize(
    "scenario,coefficient,integration,ids",
    [
        ("A-no-growth", False, False, ()),
        ("B-within-capacity", False, False, (0,)),
        ("C-coefficient-only", True, False, (0,)),
        ("D-integration-only", False, True, (0,)),
        ("E-both", True, True, (0,)),
        ("F-several", True, True, (0, 1, 3, 4, 5, 6)),
    ],
)
def test_transaction_bytes_and_buffer_identity(scenario, coefficient, integration, ids):
    _, owner = fixture(coefficient, integration)
    old = owner.state
    before = {key: a.copy() for key, a in cw.state_arrays(old).items()}
    nc = old.storage.elements["ncoef"].copy()
    ni = old.storage.elements["nint"].copy()
    required = np.zeros(nc.size, dtype=bool)
    required[list(ids)] = True
    nc[required] += 2
    ni[required] = 2 * nc[required]
    transaction, stats = adaptive.build_replacement(old, nc, ni, required, 20, 2)
    for key, a in cw.state_arrays(old).items():
        np.testing.assert_array_equal(a, before[key])
    new = adaptive.publish(transaction)
    if not ids:
        assert new is old
        assert stats is None
        return
    assert stats["coefficient_repacks"] == int(coefficient)
    assert stats["integration_repacks"] == int(integration)
    assert stats["work_bytes_copied"] == 0
    assert stats["branch_event_bytes_copied"] == 0
    if not coefficient:
        assert stats["coefficient_repack_bytes"] == 0
        assert stats["coefficient_relocation_bytes"] == 0
        for name in ("coef_data", "candidate_coef_data", "old_coef_data"):
            assert getattr(new.storage, name) is getattr(old.storage, name)
        for name in cw.COEFFICIENT_FIELDS + ("integral",):
            assert getattr(new.work, name) is getattr(old.work, name)
    if not integration:
        for name in cw.INTEGRATION_FIELDS:
            assert getattr(new.work, name) is getattr(old.work, name)
    for name in cw.BRANCH_FIELDS + cw.SCALAR_FIELDS:
        assert getattr(new.work, name) is getattr(old.work, name)
    np.testing.assert_array_equal(new.storage.elements["ncoef"], nc)
    np.testing.assert_array_equal(new.storage.elements["nint"], ni)
    assert stats["cache_rebuild_bytes"] == len(ids) * 24 * 64


def test_compact_has_no_padded_candidate_and_explicit_sizes():
    _, owner = fixture()
    work = owner.state.work
    assert "coef" not in work._fields
    for name in cw.COEFFICIENT_FIELDS + ("integral",):
        assert getattr(work, name).size == work.coef_capacity.sum()
    for name in cw.INTEGRATION_FIELDS:
        assert getattr(work, name).size == work.integration_capacity.sum()
    assert cw.state_payload(owner.state) < 400_000
    assert "coef" in hs.dtype_work.names


def test_within_capacity_failure_before_publication_preserves_all_buffers():
    _, owner = fixture()
    state = owner.state
    nc = state.storage.elements["ncoef"].copy()
    ni = state.storage.elements["nint"].copy()
    mask = np.zeros(nc.size, dtype=bool)
    mask[0] = True
    nc[0], ni[0] = 12, 24
    arrays = cw.state_arrays(state)
    before = {key: a.copy() for key, a in arrays.items()}

    def failing_cache(elements, work, required):
        adaptive.rebuild_growth_caches(elements, work, required)
        raise RuntimeError("injected affected-cache failure")

    with pytest.raises(RuntimeError, match="affected-cache"):
        adaptive.build_replacement(state, nc, ni, mask, 20, 2, rebuild=failing_cache)
    assert owner.state is state
    assert cw.state_arrays(state).keys() == arrays.keys()
    for key, a in arrays.items():
        np.testing.assert_array_equal(a, before[key])


@nb.njit
def compiled_record(work):
    record = cw.work_record(work, 0)
    record["len_discharge_element"] = 1
    record["sign_array"][0] = -1
    return record["old_coef"], record["sign_array"][0]


def test_compiled_views_alias_owned_arrays():
    _, owner = fixture()
    values, sign = compiled_record(owner.state.work)
    assert np.shares_memory(values, owner.state.work.old_coef)
    assert sign == -1
    assert owner.state.work.len_discharge_element[0] == 1
    assert compiled_record.nopython_signatures


def test_invalid_cache_invalidation_rejected_before_publication():
    _, owner = fixture()
    state = owner.state
    nc = state.storage.elements["ncoef"].copy()
    ni = state.storage.elements["nint"].copy()
    required = np.zeros(nc.size, dtype=bool)
    required[0] = True
    nc[0], ni[0] = 12, 24
    arrays = cw.state_arrays(state)
    before = {key: a.copy() for key, a in arrays.items()}

    def stale_events(elements, work, mask):
        adaptive.rebuild_growth_caches(elements, work, mask)
        work.len_discharge_element[0] = 1

    with pytest.raises(ValueError, match="invalidate"):
        adaptive.build_replacement(state, nc, ni, required, 20, 2, rebuild=stale_events)
    for key, a in arrays.items():
        np.testing.assert_array_equal(a, before[key])
