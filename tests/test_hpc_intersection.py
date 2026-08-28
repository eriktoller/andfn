import numpy as np
import pytest

from andfn.element import element_dtype_hpc
from andfn.hpc import hpc_intersection as hint


def _intersection_row():
    arr = np.zeros(1, dtype=element_dtype_hpc)
    arr[0]["_id"] = 0
    arr[0]["_type"] = 0
    arr[0]["frac0"] = 0
    arr[0]["frac1"] = 1
    arr[0]["endpoints0"] = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    arr[0]["endpoints1"] = np.array([0.0 + 1.0j, 0.0 + 3.0j], dtype=np.complex128)
    arr[0]["ncoef"] = 2
    arr[0]["nint"] = 6
    arr[0]["q"] = 2.0
    arr[0]["coef"][:2] = np.array([3.0 + 0.0j, 1.0 + 0.0j])
    return arr[0]


def test_z_array_discharge_term_and_calc_omega_branches():
    it = _intersection_row()
    z0 = hint.z_array(it, 3, 0)
    z1 = hint.z_array(it, 3, 1)
    assert np.allclose(z0, np.array([0.5 + 0j, 1.0 + 0j, 1.5 + 0j]))
    assert np.allclose(z1, np.array([0.0 + 1.5j, 0.0 + 2.0j, 0.0 + 2.5j]))

    val0 = hint.discharge_term(it, z0, 0)
    val1 = hint.discharge_term(it, z1, 1)
    assert np.isfinite(val0)
    assert np.isfinite(val1)

    om0 = hint.calc_omega(it, 1.0 + 0.2j, 0)
    om1 = hint.calc_omega(it, 1.0 + 0.2j, 1)
    assert np.isfinite(np.real(om0))
    assert np.isfinite(np.real(om1))


def test_calc_w_branches():
    it = _intersection_row()

    w0 = hint.calc_w(it, 1.0 + 0.3j, 0)
    w1 = hint.calc_w(it, 1.0 + 0.3j, 1)

    assert np.isfinite(np.real(w0))
    assert np.isfinite(np.real(w1))


def test_solve_pyfunc_updates_error(monkeypatch):
    it_arr = np.zeros(1, dtype=element_dtype_hpc)
    it_arr[0]["_id"] = 0
    it_arr[0]["_type"] = 0
    it_arr[0]["frac0"] = 0
    it_arr[0]["frac1"] = 1
    it_arr[0]["ncoef"] = 2
    it_arr[0]["nint"] = 4
    it_arr[0]["error"] = 2.0
    it_arr[0]["error_old"] = 5.0

    fracs = np.zeros(2, dtype=[("t", np.float64)])
    fracs[0]["t"] = 2.0
    fracs[1]["t"] = 1.0
    elems = it_arr.copy()
    work = np.zeros(
        1,
        dtype=[
            ("old_coef", np.complex128, 10),
            ("coef", np.complex128, 10),
            ("coef0", np.complex128, 10),
            ("coef1", np.complex128, 10),
        ],
    )

    def fake_cauchy(
        nint, ncoef, thetas, frac, element_id, element_arr, endpoints, work_arr, out
    ):
        if frac["t"] == 2.0:
            out[:2] = np.array([2 + 0j, 4 + 0j])
        else:
            out[:2] = np.array([8 + 0j, 10 + 0j])

    monkeypatch.setattr(hint.mf, "cauchy_integral_real", fake_cauchy)
    monkeypatch.setattr(hint.mf, "calc_error", lambda c, o: 0.125)
    monkeypatch.setattr(hint.mf, "calc_coef_error", lambda c, o: 0.25)

    hint.solve.py_func(it_arr[0], fracs, elems, work[0])

    assert it_arr[0]["error_old2"] == pytest.approx(5.0)
    assert it_arr[0]["error_old"] == pytest.approx(2.0)
    assert it_arr[0]["error"] == pytest.approx(0.125)
    assert it_arr[0]["error_coef"] == pytest.approx(0.25)
    assert work[0]["coef"][0] == 0.0
