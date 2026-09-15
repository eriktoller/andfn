import numpy as np
import pytest

from andfn.element import element_dtype_hpc
from andfn.hpc import hpc_imp_object as himp


def _circle_row():
    arr = np.zeros(1, dtype=element_dtype_hpc)
    arr[0]["_id"] = 0
    arr[0]["_type"] = 4
    arr[0]["frac0"] = 0
    arr[0]["radius"] = 2.0
    arr[0]["center"] = 0.0 + 0.0j
    arr[0]["ncoef"] = 3
    arr[0]["nint"] = 6
    arr[0]["coef"][:3] = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    return arr[0]


def _line_row():
    arr = np.zeros(1, dtype=element_dtype_hpc)
    arr[0]["_id"] = 1
    arr[0]["_type"] = 5
    arr[0]["frac0"] = 0
    arr[0]["endpoints0"] = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    arr[0]["ncoef"] = 2
    arr[0]["nint"] = 4
    arr[0]["coef"][:2] = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    return arr[0]


def test_calc_omega_circle_and_array_mask_inside():
    c = _circle_row()
    inside = himp.calc_omega_circle(c, 1.0 + 0.0j)
    outside = himp.calc_omega_circle(c, 4.0 + 0.0j)

    assert np.isnan(inside.real)
    assert np.isfinite(outside.real)

    omega = np.zeros(2, dtype=np.complex128)
    z = np.array([1.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex128)
    himp.calc_omega_circle_array(c, omega, z)
    assert np.isnan(omega[0].real)
    assert np.isfinite(omega[1].real)


def test_calc_omega_line_and_array():
    l = _line_row()
    om = himp.calc_omega_line(l, 1.0 + 0.2j)
    assert np.isfinite(om.real)

    omega = np.zeros(3, dtype=np.complex128)
    z = np.array([0.1 + 0.0j, 1.0 + 0.0j, 1.9 + 0.0j], dtype=np.complex128)
    himp.calc_omega_line_array(l, omega, z)
    assert np.all(np.isfinite(omega.real))


def test_solve_circle_and_line_pyfunc_update_error(monkeypatch):
    circle_arr = np.zeros(1, dtype=element_dtype_hpc)
    circle_arr[0]["_id"] = 0
    circle_arr[0]["_type"] = 4
    circle_arr[0]["frac0"] = 0
    circle_arr[0]["radius"] = 1.0
    circle_arr[0]["ncoef"] = 2
    circle_arr[0]["nint"] = 4
    circle_arr[0]["error"] = 3.0
    circle_arr[0]["error_old"] = 5.0

    line_arr = np.zeros(1, dtype=element_dtype_hpc)
    line_arr[0]["_id"] = 1
    line_arr[0]["_type"] = 5
    line_arr[0]["frac0"] = 0
    line_arr[0]["ncoef"] = 2
    line_arr[0]["nint"] = 4
    line_arr[0]["error"] = 4.0
    line_arr[0]["error_old"] = 6.0

    fracs = np.zeros(1, dtype=[("dummy", np.int64)])
    elems = np.zeros(2, dtype=element_dtype_hpc)
    work = np.zeros(
        1,
        dtype=[
            ("old_coef", np.complex128, 10),
            ("coef", np.complex128, 10),
        ],
    )

    monkeypatch.setattr(himp.mf, "get_dpsi_corr", lambda *a, **k: None)

    def fake_cauchy(*args):
        args[-1][:2] = np.array([1 + 2j, 3 + 4j])

    monkeypatch.setattr(himp.mf, "cauchy_integral_domega", fake_cauchy)
    monkeypatch.setattr(himp.mf, "cauchy_integral_domega_line", fake_cauchy)
    monkeypatch.setattr(himp.mf, "calc_error", lambda c, o: 0.1)
    monkeypatch.setattr(himp.mf, "calc_coef_error", lambda c, o: 0.2)

    himp.solve_circle.py_func(circle_arr[0], fracs, elems, work[0])
    assert circle_arr[0]["error_old2"] == pytest.approx(5.0)
    assert circle_arr[0]["error_old"] == pytest.approx(3.0)
    assert circle_arr[0]["error"] == pytest.approx(0.1)

    himp.solve_line.py_func(line_arr[0], fracs, elems, work[0])
    assert line_arr[0]["error_old2"] == pytest.approx(6.0)
    assert line_arr[0]["error_old"] == pytest.approx(4.0)
    assert line_arr[0]["error"] == pytest.approx(0.1)
