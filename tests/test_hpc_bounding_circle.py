import numpy as np
import pytest

from andfn.element import element_dtype_hpc
from andfn.hpc import hpc_bounding_circle as hbc


def _bc_row():
    arr = np.zeros(1, dtype=element_dtype_hpc)
    arr[0]["_id"] = 0
    arr[0]["_type"] = 1
    arr[0]["frac0"] = 0
    arr[0]["radius"] = 2.0
    arr[0]["center"] = 0.0 + 0.0j
    arr[0]["ncoef"] = 3
    arr[0]["nint"] = 6
    arr[0]["coef"][:3] = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    return arr[0]


def test_get_chi_z_array_calc_omega_and_array():
    bc = _bc_row()

    chi = hbc.get_chi(bc, 2.0 + 1.0j)
    assert chi == pytest.approx(1.0 + 0.5j)

    z = hbc.z_array(bc, 8)
    assert z.shape == (8,)
    assert np.allclose(np.abs(z), 2.0)

    omega = hbc.calc_omega(bc, 2.0 + 0.0j)
    assert np.isfinite(np.real(omega))

    omega_arr = np.zeros(2, dtype=np.complex128)
    hbc.calc_omega_array(
        bc, omega_arr, np.array([2.0 + 0j, 1.0 + 1.0j], dtype=np.complex128)
    )
    assert np.all(np.isfinite(omega_arr.real))

    w = hbc.calc_w(bc, 2.0 + 0.1j)
    assert np.isfinite(np.real(w))


def test_solve_pyfunc_updates_errors(monkeypatch):
    bc_arr = np.zeros(1, dtype=element_dtype_hpc)
    bc_arr[0]["_id"] = 0
    bc_arr[0]["_type"] = 1
    bc_arr[0]["frac0"] = 0
    bc_arr[0]["radius"] = 1.0
    bc_arr[0]["ncoef"] = 3
    bc_arr[0]["nint"] = 6
    bc_arr[0]["error"] = 3.0
    bc_arr[0]["error_old"] = 7.0

    fracs = np.zeros(1, dtype=[("dummy", np.int64)])
    elems = bc_arr.copy()
    work = np.zeros(
        1,
        dtype=[
            ("old_coef", np.complex128, 10),
            ("coef", np.complex128, 10),
        ],
    )

    monkeypatch.setattr(hbc.mf, "get_dpsi_corr", lambda *a, **k: None)
    monkeypatch.setattr(
        hbc.mf,
        "cauchy_integral_domega",
        lambda *a, **k: a[-1].__setitem__(
            slice(0, 3), np.array([1 + 1j, 2 + 2j, 3 + 3j])
        ),
    )
    monkeypatch.setattr(hbc.mf, "calc_error", lambda c, o: 0.2)
    monkeypatch.setattr(hbc.mf, "calc_coef_error", lambda c, o: 0.4)

    hbc.solve.py_func(bc_arr[0], fracs, elems, work[0])

    assert bc_arr[0]["error_old2"] == pytest.approx(7.0)
    assert bc_arr[0]["error_old"] == pytest.approx(3.0)
    assert bc_arr[0]["error"] == pytest.approx(0.2)
    assert bc_arr[0]["error_coef"] == pytest.approx(0.4)


def test_check_boundary_condition_pyfunc_handles_q_zero(monkeypatch):
    bc = _bc_row()
    bc["thetas"][: bc["nint"]] = np.linspace(
        0.0, 2.0 * np.pi, bc["nint"], endpoint=False
    )

    fracs = np.zeros(1, dtype=[("dummy", np.int64)])
    elems = np.zeros(1, dtype=element_dtype_hpc)

    monkeypatch.setattr(
        hbc.hpc_fracture, "calc_omega", lambda f, z, e: np.exp(1j * np.angle(z))
    )

    err = hbc.check_boundary_condition.py_func(bc, fracs, elems, n=6)
    assert np.isfinite(err)
