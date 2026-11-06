"""Characterize the unchanged legacy adaptive algorithm before migration."""

import numba as nb
import numpy as np
from test_hpc_coefficients import assert_close, legacy_logical, network, work_for

from andfn.hpc import hpc_solve as hs


def legacy_step(elements, fractures, work, maximum, increase, nit=2):
    return hs.element_solver2(
        elements.size,
        elements,
        fractures,
        work,
        1e-12,
        nit,
        0,
        -1.0,
        maximum,
        increase,
        0.5,
    )


def test_legacy_checks_old_order_and_can_overshoot_maximum():
    fractures, elements = network()
    elements["ncoef"][elements["_type"] != 2] = 4
    work = work_for(elements)
    legacy_step(elements, fractures, work, maximum=5, increase=3)
    np.testing.assert_array_equal(elements["ncoef"][elements["_type"] != 2], 7)
    np.testing.assert_array_equal(elements["nint"][elements["_type"] != 2], 14)


def test_legacy_replaces_independent_nint_even_when_it_shrinks():
    fractures, elements = network()
    elements["nint"][0] = 30
    work = work_for(elements)
    legacy_step(elements, fractures, work, maximum=5, increase=1)
    assert elements["ncoef"][0] == 5
    assert elements["nint"][0] == 10


def test_legacy_parallel_growth_repeatability_with_zero_padding():
    normal = nb.get_num_threads()
    reference = None
    try:
        for threads in sorted({1, 2, normal}):
            nb.set_num_threads(threads)
            for _ in range(3):
                fractures, elements = network()
                work = work_for(elements)
                legacy_step(elements, fractures, work, maximum=8, increase=2)
                actual = (
                    elements["ncoef"].copy(),
                    elements["nint"].copy(),
                    legacy_logical(elements),
                    work["coef"].copy(),
                    elements["error_coef"].copy(),
                )
                if reference is not None:
                    for a, b in zip(reference[:2], actual[:2]):
                        np.testing.assert_array_equal(a, b)
                    for a, b in zip(reference[2:], actual[2:]):
                        assert_close(a, b)
                reference = actual
    finally:
        nb.set_num_threads(normal)
