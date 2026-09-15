import numpy as np
import pytest

import andfn.impermeable_object as imp_mod
from andfn.impermeable_object import (
    ImpermeableCircle,
    ImpermeableLine,
    _ImpermeableEllipse,
)


class _FracStub:
    def __init__(self, _id=1):
        self._id = _id
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)


def test_impermeable_ellipse_stores_constructor_state():
    frac = _FracStub(_id=9)
    focis = [1 + 0j, -1 + 0j]

    ellipse = _ImpermeableEllipse("IE1", focis, nu=0.25, ncoef=4, nint=7, frac=frac)

    assert str(ellipse) == "Impermeable ellipse: IE1"
    assert ellipse.focis == focis
    assert ellipse.nu == pytest.approx(0.25)
    assert ellipse.ncoef == 4
    assert ellipse.nint == 7
    assert ellipse.frac is frac
    assert ellipse.error == 1
    assert np.allclose(ellipse.coef, np.zeros(4, dtype=complex))


def test_impermeable_circle_init_to_dict_and_tracking(monkeypatch):
    called = {}

    def fake_map_chi_to_z_circle(chi, radius, center):
        called["args"] = (chi.copy(), radius, center)
        return chi * radius + center

    monkeypatch.setattr(imp_mod.gf, "map_chi_to_z_circle", fake_map_chi_to_z_circle)

    frac = _FracStub(_id=3)
    circle = ImpermeableCircle(
        "IC1",
        radius=2.0,
        center=np.array([1.5 + 0.5j]),
        frac0=frac,
        ncoef=5,
        nint=8,
    )

    assert circle in frac.elements
    assert circle._type == 4
    assert np.allclose(circle.coef, 0.0 + 0.0j)

    data = circle.to_dict()
    assert data["label"] == "IC1"
    assert data["frac0"] == 3
    assert data["radius"] == pytest.approx(2.0)
    assert data["ncoef"] == 5
    assert data["nint"] == 8

    z = circle.z_array_tracking(4, offset=0.2)
    expected_chi = np.exp(1j * np.linspace(0, 2 * np.pi, 4, endpoint=False)) * 1.1
    assert np.allclose(called["args"][0], expected_chi)
    assert called["args"][1] == pytest.approx(2.0)
    assert np.allclose(z, expected_chi * 2.0 + circle.center)


def test_impermeable_circle_calc_omega_scalar_nan_inside(monkeypatch):
    monkeypatch.setattr(
        imp_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.complex128(0.8 + 0.0j),
    )

    frac = _FracStub()
    circle = ImpermeableCircle(
        "IC2", radius=1.0, center=np.array([0.0 + 0.0j]), frac0=frac
    )

    omega = circle.calc_omega(0.5 + 0.0j)

    assert np.isnan(omega.real)
    assert np.isnan(omega.imag)


def test_impermeable_circle_calc_omega_array_masks_inside(monkeypatch):
    monkeypatch.setattr(
        imp_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.array(
            [0.8 + 0.0j, 1.2 + 0.0j], dtype=np.complex128
        ),
    )
    monkeypatch.setattr(
        imp_mod.mf,
        "asym_expansion",
        lambda chi, coef: np.array([10.0 + 1.0j, 20.0 + 2.0j], dtype=np.complex128),
    )

    frac = _FracStub()
    circle = ImpermeableCircle(
        "IC3", radius=1.0, center=np.array([0.0 + 0.0j]), frac0=frac
    )

    omega = circle.calc_omega(np.array([0.1 + 0.0j, 1.2 + 0.0j], dtype=np.complex128))

    assert np.isnan(omega[0].real)
    assert np.isnan(omega[0].imag)
    assert omega[1] == pytest.approx(20.0 + 2.0j)


def test_impermeable_circle_calc_w_scalar_and_array(monkeypatch):
    frac = _FracStub()
    circle = ImpermeableCircle(
        "IC4", radius=2.0, center=np.array([0.0 + 0.0j]), frac0=frac
    )

    monkeypatch.setattr(
        imp_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.complex128(2.0 + 0.0j),
    )
    monkeypatch.setattr(imp_mod.mf, "asym_expansion_d1", lambda chi, coef: 6.0 + 2.0j)
    scalar_w = circle.calc_w(5.0 + 0.0j)
    assert scalar_w == pytest.approx((-6.0 - 2.0j) / 2.0)

    monkeypatch.setattr(
        imp_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.array(
            [0.7 + 0.0j, 1.5 + 0.0j], dtype=np.complex128
        ),
    )
    monkeypatch.setattr(
        imp_mod.mf,
        "asym_expansion_d1",
        lambda chi, coef: np.array([4.0 + 0.0j, 8.0 + 2.0j], dtype=np.complex128),
    )
    array_w = circle.calc_w(np.array([0.7 + 0.0j, 2.0 + 0.0j], dtype=np.complex128))
    assert np.isnan(array_w[0].real)
    assert np.isnan(array_w[0].imag)
    assert array_w[1] == pytest.approx((-8.0 - 2.0j) / 2.0)


def test_impermeable_line_init_and_to_dict():
    frac = _FracStub(_id=11)
    endpoints = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    line = ImpermeableLine("IL1", endpoints0=endpoints, frac0=frac, ncoef=4, nint=6)

    assert line in frac.elements
    assert line._type == 5
    assert line.thetas.shape == (6,)
    assert line.dpsi_corr.shape == (5,)
    assert np.allclose(line.coef, np.zeros(4, dtype=complex))

    data = line.to_dict()
    assert data["label"] == "IL1"
    assert data["frac0"] == 11
    assert data["ncoef"] == 4
    assert data["nint"] == 6


def test_impermeable_line_calc_omega_uses_asym_expansion(monkeypatch):
    calls = {}

    def fake_map(z, endpoints):
        calls["map"] = (z, endpoints.copy())
        return 1.5 + 0.5j

    def fake_asym(chi, coef):
        calls["asym"] = (chi, coef.copy())
        return 12.0 - 1.0j

    monkeypatch.setattr(imp_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(imp_mod.mf, "asym_expansion", fake_asym)

    frac = _FracStub()
    line = ImpermeableLine(
        "IL2", endpoints0=np.array([0j, 2 + 0j]), frac0=frac, ncoef=3
    )
    line.coef[:] = np.array([1 + 0j, 2 + 0j, 3 + 0j])

    omega = line.calc_omega(0.25 + 0.1j)

    assert omega == pytest.approx(12.0 - 1.0j)
    assert calls["map"][0] == 0.25 + 0.1j
    assert calls["asym"][0] == 1.5 + 0.5j
    assert np.allclose(calls["asym"][1], line.coef)


def test_impermeable_line_calc_w_formula(monkeypatch):
    monkeypatch.setattr(
        imp_mod.gf, "map_z_line_to_chi", lambda z, endpoints: 2.0 + 0.0j
    )
    monkeypatch.setattr(imp_mod.mf, "asym_expansion_d1", lambda chi, coef: 3.0 + 0.0j)

    frac = _FracStub()
    line = ImpermeableLine(
        "IL3", endpoints0=np.array([0.0 + 0.0j, 2.0 + 0.0j]), frac0=frac
    )

    w = line.calc_w(0.5 + 0.0j)

    expected = -3.0 * (2 * (2.0**2) / ((2.0**2) - 1) * 2 / 2.0)
    assert w == pytest.approx(expected)
