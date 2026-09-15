import numpy as np
import pytest

import andfn.intersection as intersection_mod
from andfn.intersection import Intersection


class _FracStub:
    def __init__(self, label, _id, t=2.0, omega_value=0.0 + 0.0j):
        self.label = label
        self._id = _id
        self.t = t
        self.elements = []
        self.omega_value = omega_value
        self.last_z = None

    def add_element(self, element):
        self.elements.append(element)

    def calc_omega(self, z, exclude=None):
        self.last_z = z
        return np.zeros_like(np.asarray(z, dtype=np.complex128)) + self.omega_value


def test_init_to_dict_and_length_registers_with_both_fractures():
    frac0 = _FracStub("F0", 1)
    frac1 = _FracStub("F1", 2)
    endpoints0 = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    endpoints1 = np.array([1.0 + 0.0j, 1.0 + 2.0j], dtype=np.complex128)

    inter = Intersection("I1", endpoints0, endpoints1, frac0, frac1, ncoef=4, nint=8)

    assert inter in frac0.elements
    assert inter in frac1.elements
    assert inter._type == 0
    assert inter.q == pytest.approx(0.0)
    assert np.allclose(inter.coef, np.zeros(4, dtype=complex))
    assert inter.length() == pytest.approx(2.0)

    data = inter.to_dict()
    assert data["label"] == "I1"
    assert data["frac0"] == "F0"
    assert data["frac1"] == "F1"
    assert data["ncoef"] == 4
    assert data["nint"] == 8


def test_discharge_term_uses_sign_for_each_fracture_branch(monkeypatch):
    def fake_map(z, endpoints):
        return np.array([1.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128)

    def fake_well_chi(chi, sign):
        return sign * chi

    monkeypatch.setattr(intersection_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(intersection_mod.mf, "well_chi", fake_well_chi)

    frac0 = _FracStub("F0", 1)
    frac1 = _FracStub("F1", 2)
    inter = Intersection(
        "I2", np.array([0j, 1 + 0j]), np.array([0j, 1 + 0j]), frac0, frac1
    )

    z = np.array([0.2 + 0.0j, 0.8 + 0.0j], dtype=np.complex128)
    assert inter.discharge_term(z, frac0) == pytest.approx(2.0)
    assert inter.discharge_term(z, frac1) == pytest.approx(-2.0)


def test_z_array_and_omega_along_element_follow_selected_fracture():
    frac0 = _FracStub("F0", 1, omega_value=1.0 + 2.0j)
    frac1 = _FracStub("F1", 2, omega_value=3.0 + 4.0j)
    endpoints0 = np.array([0.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex128)
    endpoints1 = np.array([0.0 + 1.0j, 0.0 + 5.0j], dtype=np.complex128)
    inter = Intersection("I3", endpoints0, endpoints1, frac0, frac1)

    z0 = inter.z_array(3, frac0)
    z1 = inter.z_array(3, frac1)

    assert np.allclose(z0, np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]))
    assert np.allclose(z1, np.array([0.0 + 2.0j, 0.0 + 3.0j, 0.0 + 4.0j]))

    omega0 = inter.omega_along_element(2, frac0)
    omega1 = inter.omega_along_element(2, frac1)
    assert np.allclose(omega0, np.array([1.0 + 2.0j, 1.0 + 2.0j]))
    assert np.allclose(omega1, np.array([3.0 + 4.0j, 3.0 + 4.0j]))


def test_calc_omega_branch_signs(monkeypatch):
    recorded = []

    def fake_map(z, endpoints):
        return 2.0 + 0.0j if np.allclose(endpoints, [0j, 2 + 0j]) else 3.0 + 0.0j

    def fake_asym(chi, coef):
        recorded.append((chi, coef.copy()))
        return coef[0] + chi

    def fake_well_chi(chi, q):
        return q + 0.5 * chi

    monkeypatch.setattr(intersection_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(intersection_mod.mf, "asym_expansion", fake_asym)
    monkeypatch.setattr(intersection_mod.mf, "well_chi", fake_well_chi)

    frac0 = _FracStub("F0", 1)
    frac1 = _FracStub("F1", 2)
    inter = Intersection(
        "I4", np.array([0j, 2 + 0j]), np.array([0j, 2j]), frac0, frac1, ncoef=2
    )
    inter.coef = np.array([5.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    inter.q = 4.0

    omega0 = inter.calc_omega(0.1 + 0.0j, frac0)
    omega1 = inter.calc_omega(0.1 + 0.0j, frac1)

    assert omega0 == pytest.approx(12.0 + 0.0j)
    assert omega1 == pytest.approx(-4.5 + 0.0j)
    assert np.allclose(recorded[0][1], np.array([5.0 + 0.0j, 1.0 + 0.0j]))
    assert np.allclose(recorded[1][1], np.array([-5.0 + 0.0j, -1.0 + 0.0j]))


def test_calc_w_branch_signs_and_scaling(monkeypatch):
    def fake_map(z, endpoints):
        return 2.0 + 0.0j if np.allclose(endpoints, [0j, 2 + 0j]) else 3.0 + 0.0j

    def fake_asym_d1(chi, coef):
        return coef[0]

    monkeypatch.setattr(intersection_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(intersection_mod.mf, "asym_expansion_d1", fake_asym_d1)

    frac0 = _FracStub("F0", 1)
    frac1 = _FracStub("F1", 2)
    inter = Intersection(
        "I5", np.array([0j, 2 + 0j]), np.array([0j, 4 + 0j]), frac0, frac1, ncoef=1
    )
    inter.coef = np.array([3.0 + 0.0j], dtype=np.complex128)
    inter.q = 2.0

    w0 = inter.calc_w(0.0 + 0.0j, frac0)
    w1 = inter.calc_w(0.0 + 0.0j, frac1)

    expected0 = (-3.0 - 1.0 / (2 * np.pi)) * (8.0 / 3.0)
    expected1 = (3.0 + 1.0 / (3 * np.pi)) * (9.0 / 8.0)
    assert w0 == pytest.approx(expected0)
    assert w1 == pytest.approx(expected1)


def test_check_boundary_condition_zero_and_nonzero(monkeypatch):
    monkeypatch.setattr(
        intersection_mod.gf, "map_chi_to_z_line", lambda chi, endpoints: chi
    )

    frac0 = _FracStub("F0", 1, t=2.0, omega_value=4.0 + 1.0j)
    frac1 = _FracStub("F1", 2, t=4.0, omega_value=8.0 + 2.0j)
    inter = Intersection(
        "I6", np.array([0j, 1 + 0j]), np.array([0j, 1 + 0j]), frac0, frac1
    )

    assert inter.check_boundary_condition(n=10) == pytest.approx(0.0)

    def varying_omega(z, exclude=None):
        z = np.asarray(z, dtype=np.complex128)
        return 8.0 + np.real(z) + 0.0j

    frac1.calc_omega = varying_omega
    assert inter.check_boundary_condition(n=20) > 0.0


def test_check_chi_crossing_branches(monkeypatch):
    frac0 = _FracStub("F0", 1)
    frac1 = _FracStub("F1", 2)
    inter = Intersection("I7", np.array([0j, 1 + 0j]), np.array([0j, 1j]), frac0, frac1)

    monkeypatch.setattr(
        intersection_mod.gf, "line_line_intersection", lambda *args: None
    )
    assert inter.check_chi_crossing(0 + 0j, 1 + 1j, frac0) is False

    monkeypatch.setattr(
        intersection_mod.gf, "line_line_intersection", lambda *args: 2 + 0j
    )
    assert inter.check_chi_crossing(0 + 0j, 1 + 1j, frac0) is False

    monkeypatch.setattr(
        intersection_mod.gf, "line_line_intersection", lambda *args: 0.5 + 0.0j
    )
    hit0 = inter.check_chi_crossing(0.5 - 1j, 0.5 + 1j, frac0)
    assert hit0 == pytest.approx(0.5 + 0.0j)

    monkeypatch.setattr(
        intersection_mod.gf, "line_line_intersection", lambda *args: 0.0 + 0.5j
    )
    hit1 = inter.check_chi_crossing(-1 + 0.5j, 1 + 0.5j, frac1)
    assert hit1 == pytest.approx(0.0 + 0.5j)
