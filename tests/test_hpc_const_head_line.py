import numpy as np
import pytest

from andfn.element import element_dtype_hpc
from andfn.hpc import hpc_const_head_line as hch


def _line_row():
    arr = np.zeros(1, dtype=element_dtype_hpc)
    arr[0]["_id"] = 0
    arr[0]["_type"] = 3
    arr[0]["frac0"] = 0
    arr[0]["endpoints0"] = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    arr[0]["ncoef"] = 3
    arr[0]["nint"] = 6
    arr[0]["q"] = 4.0
    arr[0]["coef"][:3] = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    return arr[0]


def test_discharge_and_z_array():
    line = _line_row()
    z = np.array([0.5 + 0.0j, 1.5 + 0.0j], dtype=np.complex128)

    val = hch.discharge_term(line, z)
    assert np.isfinite(val)

    za = hch.z_array(line, 4)
    assert np.allclose(za, np.array([0.4 + 0j, 0.8 + 0j, 1.2 + 0j, 1.6 + 0j]))


def test_calc_omega_calc_omega_array_and_calc_w():
    line = _line_row()

    om = hch.calc_omega(line, 1.0 + 0.2j)
    assert np.isfinite(om.real)
    assert np.isfinite(om.imag)

    omega = np.zeros(3, dtype=np.complex128)
    z = np.array([0.2 + 0.1j, 1.0 + 0.0j, 1.8 - 0.1j], dtype=np.complex128)
    hch.calc_omega_array(line, omega, z)
    assert np.all(np.isfinite(omega.real))

    w = hch.calc_w(line, 1.0 + 0.3j)
    assert np.isfinite(np.real(w))


def test_solve_pyfunc_updates_error_fields(monkeypatch):
    line_arr = np.zeros(1, dtype=element_dtype_hpc)
    line_arr[0]["_id"] = 0
    line_arr[0]["_type"] = 3
    line_arr[0]["frac0"] = 0
    line_arr[0]["ncoef"] = 3
    line_arr[0]["nint"] = 6
    line_arr[0]["error"] = 2.0
    line_arr[0]["error_old"] = 5.0
    line_arr[0]["coef"][:3] = np.array([1 + 1j, 2 + 2j, 3 + 3j])

    frac = np.zeros(1, dtype=[("t", np.float64)])
    frac[0]["t"] = 1.0
    elems = line_arr.copy()
    work = np.zeros(
        1,
        dtype=[
            ("old_coef", np.complex128, 10),
            ("coef", np.complex128, 10),
        ],
    )

    monkeypatch.setattr(
        hch.mf,
        "cauchy_integral_real",
        lambda *a, **k: a[-1].__setitem__(
            slice(0, 3), np.array([2 + 1j, 4 + 0j, 6 - 2j])
        ),
    )
    monkeypatch.setattr(hch.mf, "calc_error", lambda c, o: 0.25)
    monkeypatch.setattr(hch.mf, "calc_coef_error", lambda c, o: 0.5)

    hch.solve.py_func(line_arr[0], frac, elems, work[0])

    assert line_arr[0]["error_old2"] == pytest.approx(5.0)
    assert line_arr[0]["error_old"] == pytest.approx(2.0)
    assert line_arr[0]["error"] == pytest.approx(0.25)
    assert line_arr[0]["error_coef"] == pytest.approx(0.5)
    assert work[0]["coef"][0] == 0.0
    assert np.all(np.isreal(work[0]["coef"][:3]))
