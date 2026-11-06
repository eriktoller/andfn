"""Affected-only control plane and preparation/publication failure contract."""

import numpy as np
import pytest
from test_compact_work import fixture

from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_work as cw


def request(state):
    nc = state.storage.elements["ncoef"].copy()
    ni = state.storage.elements["nint"].copy()
    required = np.zeros(nc.size, dtype=bool)
    required[0] = True
    nc[0], ni[0] = 12, 24
    return nc, ni, required


def snapshot(state):
    return {key: (a, a.copy()) for key, a in cw.state_arrays(state).items()}


def unchanged(state, before):
    actual = cw.state_arrays(state)
    assert actual.keys() == before.keys()
    for key, (original, values) in before.items():
        assert actual[key] is original
        np.testing.assert_array_equal(actual[key], values)
    adaptive.validate_state(state, finite=True)


@pytest.mark.parametrize("custom", [False, True])
def test_logical_path_never_constructs_ownership_or_scans_unchanged_values(
    monkeypatch, custom
):
    _, owner = fixture()
    state = owner.state
    before = snapshot(state)
    req = request(state)

    def forbidden(*args, **kwargs):
        raise AssertionError("Global repack/structural scan/ownership reconstruction")

    monkeypatch.setattr(adaptive, "build_repack", forbidden)
    monkeypatch.setattr(adaptive, "validate_requests", forbidden)
    monkeypatch.setattr(adaptive, "AdaptiveStorage", forbidden)
    monkeypatch.setattr(adaptive, "AdaptiveState", forbidden)

    def callback(elements, caches, mask):
        assert isinstance(caches, adaptive.CacheStage)
        adaptive.rebuild_growth_caches(elements, caches, mask)

    kwargs = {"rebuild": callback} if custom else {}
    tx, stats = adaptive.build_replacement(state, *req, 20, 2, **kwargs)
    assert tx.state is state
    assert stats["complete_state_scans"] == 0
    assert stats["coefficient_bytes_copied"] == 0
    assert stats["work_bytes_copied"] == 0
    for key, (_, a) in before.items():
        np.testing.assert_array_equal(cw.state_arrays(state)[key], a)
    adaptive.validate_state(state, finite=True)


@pytest.mark.parametrize("invalid", ["requested", "integration", "mask"])
def test_invalid_request_arrays_fail_before_publication(invalid):
    _, owner = fixture()
    state = owner.state
    before = snapshot(state)
    nc, ni, mask = request(state)
    if invalid == "requested":
        nc = nc.astype(float)
    elif invalid == "integration":
        ni = None
    else:
        mask = mask.astype(np.int64)
    with pytest.raises(ValueError, match="Invalid growth request arrays"):
        adaptive.build_replacement(state, nc, ni, mask, 20, 2)
    unchanged(state, before)


@pytest.mark.parametrize("failure", ["cache", "finite", "prepublication", "readonly"])
def test_preparation_failures_preserve_complete_state(failure, monkeypatch):
    _, owner = fixture()
    state = owner.state
    req = request(state)
    before = snapshot(state)

    def failing(*args):
        raise RuntimeError("injected before publication")

    if failure == "cache":
        monkeypatch.setattr(adaptive, "prepare_cache_patch", failing)
    elif failure == "finite":
        original = adaptive.prepare_cache_patch

        def nonfinite(*args):
            original(*args)
            args[-2].exp_array_p[0] = np.nan

        monkeypatch.setattr(adaptive, "prepare_cache_patch", nonfinite)
    elif failure == "readonly":
        state.work.thetas.flags.writeable = False
    try:
        with pytest.raises((ValueError, RuntimeError)):
            tx, _ = adaptive.build_replacement(state, *req, 20, 2)
            if failure == "prepublication":
                failing(tx)
            adaptive.publish(tx)
    finally:
        state.work.thetas.flags.writeable = True
    assert owner.state is state
    unchanged(state, before)


def test_prepublication_hook_observes_committed_state_and_can_abort():
    fractures, owner = fixture()
    observed = {}

    def abort(state):
        observed["state"] = state
        observed["snapshot"] = snapshot(state)
        raise RuntimeError("injected immediately before publication")

    owner.before_publication = abort
    with pytest.raises(RuntimeError, match="immediately"):
        adaptive.adaptive_step(owner, fractures, 1e-12, 2, 0, -1.0, 20, 2, 0.5)
    assert owner.state is observed["state"]
    unchanged(owner.state, observed["snapshot"])


def test_publication_preserves_owner_identity_and_structural_validity():
    _, owner = fixture()
    state = owner.state
    tx, _ = adaptive.build_replacement(state, *request(state), 20, 2)
    assert isinstance(tx, adaptive.LogicalGrowth)
    for array in (*tx.caches, tx.ids, tx.starts, tx.ncoef, tx.nint):
        assert array.flags.c_contiguous
    assert all(a.flags.writeable for a in cw.state_arrays(state).values())
    owner.state = adaptive.publish(tx)
    assert owner.state is state
    adaptive.validate_state(state, finite=True)


def test_complete_finite_validation_is_explicit_not_a_transaction_scan():
    _, owner = fixture()
    state = owner.state
    # Unchanged element, deliberately poisoned only for the debug validator.
    i = 3
    offset = state.storage.coef_start[i]
    state.storage.coef_data[offset] = np.nan
    tx, stats = adaptive.build_replacement(state, *request(state), 20, 2)
    assert stats["complete_state_scans"] == 0
    assert tx.state is state
    with pytest.raises(ValueError, match="Nonfinite coefficient"):
        adaptive.validate_state(state, finite=True)
    state.storage.coef_data[offset] = 0
    adaptive.validate_state(state, finite=True)


@pytest.mark.parametrize("name", ["old_coef", "phi", "error"])
def test_debug_validation_includes_noncache_work(name):
    _, owner = fixture()
    value = getattr(owner.state.work, name)
    original = value[0]
    value[0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        adaptive.validate_state(owner.state, finite=True)
    value[0] = original
    adaptive.validate_state(owner.state, finite=True)
