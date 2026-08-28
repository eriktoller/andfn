import numpy as np
import pytest

from andfn.element import element_dtype_hpc, fracture_dtype_hpc
from andfn.hpc import hpc_solve as hs


def _build_small_arrays():
    fractures = np.zeros(2, dtype=fracture_dtype_hpc)
    fractures[0]["_id"] = 0
    fractures[0]["t"] = 2.0
    fractures[1]["_id"] = 1
    fractures[1]["t"] = 4.0

    elements = np.zeros(4, dtype=element_dtype_hpc)

    elements[0]["_id"] = 0
    elements[0]["_type"] = 0
    elements[0]["frac0"] = 0
    elements[0]["frac1"] = 1
    elements[0]["endpoints0"] = np.array([0 + 0j, 2 + 0j])
    elements[0]["endpoints1"] = np.array([0 + 1j, 0 + 3j])

    elements[1]["_id"] = 1
    elements[1]["_type"] = 2
    elements[1]["frac0"] = 0
    elements[1]["radius"] = 1.0

    elements[2]["_id"] = 2
    elements[2]["_type"] = 3
    elements[2]["frac0"] = 1
    elements[2]["endpoints0"] = np.array([0 + 0j, 1 + 0j])

    elements[3]["_id"] = 3
    elements[3]["_type"] = 1
    elements[3]["frac0"] = 0

    fractures[0]["elements"][:3] = np.array([0, 1, 3])
    fractures[0]["nelements"] = 3
    fractures[1]["elements"][:2] = np.array([0, 2])
    fractures[1]["nelements"] = 2

    return fractures, elements


def test_get_error_discharge_elements_prefix_sum_and_old_discharges():
    fractures, elements = _build_small_arrays()
    elements[0]["error"] = 0.2
    elements[1]["error"] = 0.7
    elements[2]["error"] = 0.1
    elements[3]["error"] = 0.05

    err, idx = hs.get_error(elements)
    assert err == pytest.approx(0.7)
    assert idx == 1

    discharge = hs.get_discharge_elements(elements)
    assert discharge.size == 3
    assert set(discharge["_type"].tolist()) == {0, 2, 3}

    offsets, total = hs.exclusive_prefix_sum(np.array([3, 0, 2], dtype=np.int64))
    assert np.array_equal(offsets, np.array([0, 3, 3], dtype=np.int64))
    assert total == 5

    elements[0]["q"] = 11.0
    elements[1]["q"] = 22.0
    elements[2]["q"] = 33.0
    fractures[0]["constant"] = 100.0
    fractures[1]["constant"] = 200.0
    old = hs.get_old_discharges(elements, fractures, discharge)
    assert np.allclose(old[:3], np.array([11.0, 22.0, 33.0]))
    assert np.allclose(old[3:], np.array([100.0, 200.0]))


def test_count_nnz_and_post_matrix_and_set_new_ncoef():
    fractures, elements = _build_small_arrays()
    discharge = hs.get_discharge_elements(elements)

    nnz = hs.count_discharge_nnz(fractures, elements, discharge)
    assert nnz.size == discharge.size + fractures.size
    assert np.all(nnz >= 1)

    discharges = np.array([1.5, -2.5, 3.5, 10.0, 20.0], dtype=np.float64)
    old = np.zeros_like(discharges)
    hs.post_matrix_solve(fractures, elements, discharge, discharges, old)
    assert elements[0]["q"] == pytest.approx(1.5)
    assert elements[1]["q"] == pytest.approx(-2.5)
    assert elements[2]["q"] == pytest.approx(3.5)
    assert fractures[0]["constant"] == pytest.approx(10.0)
    assert fractures[1]["constant"] == pytest.approx(20.0)

    e = np.zeros(1, dtype=element_dtype_hpc)
    e[0]["_type"] = 1
    hs.set_new_ncoef(e[0], 6, nint_mult=2)
    assert e[0]["ncoef"] == 6
    assert e[0]["nint"] == 12
