import numpy as np
import pytest

from andfn.element import element_dtype_hpc, fracture_dtype_hpc
from andfn.hpc import hpc_fracture as hfr


def _fracture_row():
    f = np.zeros(1, dtype=fracture_dtype_hpc)
    f[0]["_id"] = 0
    f[0]["t"] = 2.0
    f[0]["radius"] = 3.0
    f[0]["center"] = np.array([0.0, 0.0, 0.0])
    f[0]["x_vector"] = np.array([1.0, 0.0, 0.0])
    f[0]["y_vector"] = np.array([0.0, 1.0, 0.0])
    f[0]["constant"] = 1.0
    return f


def test_sunflower_and_head_from_phi():
    z = hfr.sunflower_spiral(5, 3)
    assert z.size == 8
    assert np.allclose(np.abs(z[-3:]), 1.0)

    f = _fracture_row()
    assert hfr.head_from_phi(f[0], 6.0) == pytest.approx(3.0)


def test_calc_omega_and_w_dispatch_with_pyfunc(monkeypatch):
    f = _fracture_row()
    elements = np.zeros(6, dtype=element_dtype_hpc)
    for i in range(6):
        elements[i]["_id"] = i
        elements[i]["_type"] = i
        elements[i]["frac0"] = 0
    f[0]["elements"][:6] = np.arange(6)
    f[0]["nelements"] = 6

    monkeypatch.setattr(hfr.hpc_intersection, "calc_omega", lambda e, z, fid: 10.0 + 0j)
    monkeypatch.setattr(hfr.hpc_bounding_circle, "calc_omega", lambda e, z: 20.0 + 0j)
    monkeypatch.setattr(hfr.hpc_well, "calc_omega", lambda e, z: 30.0 + 0j)
    monkeypatch.setattr(hfr.hpc_const_head_line, "calc_omega", lambda e, z: 40.0 + 0j)
    monkeypatch.setattr(hfr.hpc_imp_object, "calc_omega_circle", lambda e, z: 50.0 + 0j)
    monkeypatch.setattr(hfr.hpc_imp_object, "calc_omega_line", lambda e, z: 60.0 + 0j)

    om = hfr.calc_omega.py_func(f[0], 0.0 + 0.0j, elements, exclude=-1)
    assert om == pytest.approx(1.0 + 210.0)

    monkeypatch.setattr(hfr.hpc_intersection, "calc_w", lambda e, z, fid: 1.0 + 0j)
    monkeypatch.setattr(hfr.hpc_bounding_circle, "calc_w", lambda e, z: 2.0 + 0j)
    monkeypatch.setattr(hfr.hpc_well, "calc_w", lambda e, z: 3.0 + 0j)
    monkeypatch.setattr(hfr.hpc_const_head_line, "calc_w", lambda e, z: 4.0 + 0j)

    w = hfr.calc_w(f[0], 0.0 + 0.0j, elements, exclude=-1)
    assert w == pytest.approx(10.0 + 0.0j)


def test_calc_omega_array_and_get_heads(monkeypatch):
    f = _fracture_row()
    elements = np.zeros(1, dtype=element_dtype_hpc)
    elements[0]["_id"] = 0
    elements[0]["_type"] = 1
    elements[0]["radius"] = 1.0
    f[0]["elements"][0] = 0
    f[0]["nelements"] = 1

    monkeypatch.setattr(
        hfr.hpc_bounding_circle,
        "calc_omega_array",
        lambda e, omega, z: omega.__setitem__(slice(None), omega + 2.0 + 0.0j),
    )

    omega = np.zeros(3, dtype=np.complex128)
    z = np.array([0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex128)
    hfr.calc_omega_array.py_func(f[0], omega, z, elements, exclude=-1)
    assert np.allclose(omega, 3.0 + 0.0j)

    heads = np.zeros(2, dtype=np.float64)
    z_pts = np.array([0.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)
    hfr.calc_heads(f[0], heads, 2, z_pts, elements)
    assert np.all(np.isfinite(heads))
