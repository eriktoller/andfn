"""Public Numba scratch ownership and write isolation."""

import numba as nb
import numpy as np
import pytest
from benchmark_working_set import scratch_case
from test_hpc_coefficients import network, work_for

from andfn.hpc import hpc_adaptive as adaptive
from andfn.hpc import hpc_solve as hs
from andfn.hpc import hpc_work as cw


@pytest.mark.parametrize("threads", sorted({1, 2, nb.get_num_threads()}))
def test_scratch_ownership_prototype(threads):
    previous = nb.get_num_threads()
    try:
        nb.set_num_threads(threads)
        results = scratch_case(1024, threads, repetitions=5)
        assert all(case["isolated"] for case in results.values())
        assert (
            results["worker"]["allocated_bytes"]
            <= results["element"]["allocated_bytes"]
        )
    finally:
        nb.set_num_threads(previous)


@pytest.mark.parametrize("groups", [1, 3, 7])
def test_worker_groups_preserve_exact_candidates_and_diagnostics(groups):
    fractures, elements = network()
    worker = adaptive.adaptive_from_hpc(elements, scratch_groups=groups)
    diagnostic = adaptive.adaptive_from_hpc(elements, scratch_groups=0)
    for iteration in range(1, 4):
        for owner in (worker, diagnostic):
            adaptive.adaptive_step(
                owner, fractures, 1e-12, iteration, 0, -1.0, 12, 2, 0.5
            )
        a, b = worker.state, diagnostic.state
        np.testing.assert_array_equal(a.storage.elements, b.storage.elements)
        for name in ("coef_data", "candidate_coef_data", "old_coef_data"):
            np.testing.assert_array_equal(
                getattr(a.storage, name), getattr(b.storage, name)
            )
        for name in cw.CACHE_FIELDS + cw.BRANCH_FIELDS:
            np.testing.assert_array_equal(getattr(a.work, name), getattr(b.work, name))
        adaptive.validate_state(a, finite=True)
    with pytest.raises(ValueError, match="no element history"):
        cw.export_work(worker.state.work, hs.dtype_work)
    exported = cw.export_work(diagnostic.state.work, hs.dtype_work)
    assert exported.shape == elements.shape
    work = worker.state.work
    slots, c, n = work.scratch_layout[1:]
    for name in cw.scratch_fields(work):
        width = n if name in cw.SCRATCH_GROUPS[2] else c
        assert getattr(work, name).size == slots * width


def test_element_owned_diagnostic_mode_preserves_all_six_temporary_fields():
    fractures, elements = network()
    legacy = work_for(elements)
    owner = adaptive.adaptive_from_hpc(elements, scratch_groups=0)
    cw.import_work(owner.state.work, legacy, elements)
    hs.element_solver2(
        elements.size, elements, fractures, legacy, 1e-12, 1, 0, 1e100, 12, 2, 0.5
    )
    adaptive.adaptive_step(owner, fractures, 1e-12, 1, 0, 1e100, 12, 2, 0.5)
    exported = cw.export_work(owner.state.work, hs.dtype_work)
    for i, element in enumerate(elements):
        for name in cw.SCRATCH_FIELDS:
            length = (
                element["nint"] if name in cw.SCRATCH_GROUPS[2] else element["ncoef"]
            )
            np.testing.assert_array_equal(
                exported[name][i, :length], legacy[name][i, :length]
            )
