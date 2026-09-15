import numpy as np
import pytest

from andfn.element import element_dtype_hpc
from andfn.hpc import hpc_well


def _well_row(radius=1.0, center=0.0 + 0.0j, q=2.0):
    arr = np.zeros(1, dtype=element_dtype_hpc)
    arr[0]["_type"] = 2
    arr[0]["radius"] = radius
    arr[0]["center"] = center
    arr[0]["q"] = q
    return arr[0]


def test_discharge_term_and_z_array():
    w = _well_row(radius=2.0, center=1.0 + 0.0j)
    z = np.array([3.0 + 0.0j, 5.0 + 0.0j], dtype=np.complex128)

    val = hpc_well.discharge_term(w, z)
    chi = (z - w["center"]) / w["radius"]
    expected = np.mean(np.real((1.0 / (2 * np.pi)) * np.log(chi)))
    assert val == pytest.approx(expected)

    pts = hpc_well.z_array(w, 8)
    assert pts.shape == (8,)
    assert np.allclose(np.abs(pts - w["center"]), 2.0)


def test_calc_omega_and_calc_w_scalar_and_array():
    w = _well_row(radius=2.0, center=0.0 + 0.0j, q=4.0)

    inside = hpc_well.calc_omega(w, 1.0 + 0.0j)
    outside = hpc_well.calc_omega(w, 4.0 + 0.0j)
    assert np.isnan(inside.real) and np.isnan(inside.imag)
    assert outside == pytest.approx((4.0 / (2 * np.pi)) * np.log(2.0 + 0.0j))

    omega = np.zeros(2, dtype=np.complex128)
    z = np.array([1.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex128)
    hpc_well.calc_omega_array(w, omega, z)
    assert np.isnan(omega[0].real)
    assert omega[1] == pytest.approx((4.0 / (2 * np.pi)) * np.log(2.0 + 0.0j))

    ws = hpc_well.calc_w(w, 4.0 + 0.0j)
    assert ws == pytest.approx(-4.0 / (2 * np.pi * 2.0) / 2.0)
