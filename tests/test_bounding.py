import numpy as np
import pytest

import andfn.bounding as bounding_mod
from andfn.bounding import BoundingCircle


class _FracStub:
    def __init__(self, _id=7):
        self._id = _id
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def calc_omega(self, z, exclude=None):
        # Use unwrapped phase to produce nearly uniform dpsi on circular samples.
        theta = np.angle(z)
        if isinstance(theta, np.ndarray):
            theta = np.unwrap(theta)
        return 1.0 + 2.0j * theta


def test_bounding_circle_init_and_to_dict():
    frac = _FracStub(_id=3)
    bc = BoundingCircle("B1", radius=2.5, frac0=frac, ncoef=4, nint=11)

    assert bc in frac.elements
    assert bc._type == 1
    assert bc.center == 0.0 + 0.0j
    assert bc.coef.shape == (4,)
    assert np.allclose(bc.coef, 0.0 + 0.0j)

    data = bc.to_dict()
    assert data["_id"] == -1
    assert data["_type"] == 1
    assert data["label"] == "B1"
    assert data["frac0"] == 3
    assert data["ncoef"] == 4
    assert data["nint"] == 11


def test_get_chi_scalar_inside_and_outside():
    frac = _FracStub()
    bc = BoundingCircle("B2", radius=2.0, frac0=frac)

    inside = bc.get_chi(1.0 + 0.0j)
    outside = bc.get_chi(3.0 + 0.0j)

    assert inside == pytest.approx(0.5 + 0.0j)
    assert np.isnan(outside.real)
    assert np.isnan(outside.imag)


def test_get_chi_array_masks_outside_points_only():
    frac = _FracStub()
    bc = BoundingCircle("B3", radius=1.0, frac0=frac)

    z = np.array([0.2 + 0.1j, 0.8 + 0.0j, 1.2 + 0.0j], dtype=np.complex128)
    chi = bc.get_chi(z)

    assert np.allclose(chi[:2], z[:2])
    assert np.isnan(chi[2].real)
    assert np.isnan(chi[2].imag)


def test_calc_omega_uses_taylor_series(monkeypatch):
    calls = {}

    def fake_map(z, radius):
        calls["map"] = (z, radius)
        return 0.2 + 0.3j

    def fake_taylor(chi, coef):
        calls["taylor"] = (chi, coef.copy())
        return 9.5 + 1.0j

    monkeypatch.setattr(bounding_mod.gf, "map_z_circle_to_chi", fake_map)
    monkeypatch.setattr(bounding_mod.mf, "taylor_series", fake_taylor)

    frac = _FracStub()
    bc = BoundingCircle("B4", radius=4.0, frac0=frac, ncoef=3)
    bc.coef[:] = np.array([1 + 0j, 2 + 0j, 3 + 0j])

    omega = bc.calc_omega(2.0 + 1.0j)

    assert omega == pytest.approx(9.5 + 1.0j)
    assert calls["map"] == (2.0 + 1.0j, 4.0)
    assert calls["taylor"][0] == 0.2 + 0.3j
    assert np.allclose(calls["taylor"][1], bc.coef)


def test_calc_w_applies_minus_derivative_and_radius_scaling(monkeypatch):
    def fake_map(z, radius):
        return 0.6 + 0.0j

    def fake_d1(chi, coef):
        return 8.0 + 0.0j

    monkeypatch.setattr(bounding_mod.gf, "map_z_circle_to_chi", fake_map)
    monkeypatch.setattr(bounding_mod.mf, "taylor_series_d1", fake_d1)

    frac = _FracStub()
    bc = BoundingCircle("B5", radius=2.0, frac0=frac)

    w = bc.calc_w(0.1 + 0.2j)

    assert w == pytest.approx(-4.0 + 0.0j)


def test_check_boundary_condition_zero_for_uniform_dpsi():
    frac = _FracStub()
    bc = BoundingCircle("B6", radius=1.0, frac0=frac)

    err = bc.check_boundary_condition(n=24)

    assert err == pytest.approx(0.0, abs=1e-12)


def test_check_boundary_condition_detects_nonuniform_dpsi():
    class _NonUniformFrac(_FracStub):
        def calc_omega(self, z, exclude=None):
            theta = np.angle(z)
            return 0.0 + 1j * (theta + 0.2 * np.sin(3 * theta))

    frac = _NonUniformFrac()
    bc = BoundingCircle("B7", radius=1.0, frac0=frac)

    err = bc.check_boundary_condition(n=128)

    assert err > 0.0
