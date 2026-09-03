import numpy as np
import pytest

import andfn.const_head as const_head_mod
from andfn.const_head import ConstantHeadLine


class _FracStub:
    def __init__(self, _id=1, t=2.0, center=0.0 + 0.0j):
        self._id = _id
        self.t = t
        self.center = center
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def phi_from_head(self, head):
        return head * self.t

    def calc_omega(self, z, exclude=None):
        return np.asarray(z, dtype=np.complex128) * 0.0 + (3.0 + 0.5j)


def test_init_and_to_dict_registers_element_and_sets_phi():
    frac = _FracStub(_id=8, t=3.0)
    endpoints = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    ch = ConstantHeadLine("CH1", endpoints, head=4.0, frac0=frac, ncoef=6, nint=12)

    assert ch in frac.elements
    assert ch._type == 3
    assert ch.q == 0.0
    assert ch.phi == pytest.approx(12.0)
    assert ch.coef.shape == (6,)

    data = ch.to_dict()
    assert data["label"] == "CH1"
    assert data["frac0"] == 8
    assert data["head"] == pytest.approx(4.0)
    assert data["phi"] == pytest.approx(12.0)
    assert data["ncoef"] == 6
    assert data["nint"] == 12


def test_update_head_updates_phi_from_fracture_mapping():
    frac = _FracStub(t=5.0)
    ch = ConstantHeadLine("CH2", np.array([0j, 1 + 0j]), head=1.0, frac0=frac)

    ch.update_head(2.5)

    assert ch.head == pytest.approx(2.5)
    assert ch.phi == pytest.approx(12.5)


def test_length_and_z_array():
    frac = _FracStub()
    ch = ConstantHeadLine("CH3", np.array([0 + 0j, 4 + 0j]), head=1.0, frac0=frac)

    assert ch.length() == pytest.approx(4.0)
    assert np.allclose(ch.z_array(3), np.array([1 + 0j, 2 + 0j, 3 + 0j]))


def test_discharge_term_uses_mapped_chi_and_well(monkeypatch):
    def fake_map(z, endpoints):
        return np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex128)

    def fake_well_chi(chi, q):
        return 2.0 * chi + 1j

    monkeypatch.setattr(const_head_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(const_head_mod.mf, "well_chi", fake_well_chi)

    frac = _FracStub()
    ch = ConstantHeadLine("CH4", np.array([0j, 1 + 0j]), head=1.0, frac0=frac)

    val = ch.discharge_term(np.array([0 + 0j, 0.5 + 0j, 1 + 0j]))

    assert val == pytest.approx(4.0)


def test_omega_along_element_calls_given_fracture():
    class _FracOmega:
        def __init__(self):
            self.last_z = None

        def calc_omega(self, z):
            self.last_z = z
            return z + (1.0 + 2.0j)

    frac = _FracStub()
    ch = ConstantHeadLine("CH5", np.array([0 + 0j, 2 + 0j]), head=1.0, frac0=frac)
    frac_is = _FracOmega()

    omega = ch.omega_along_element(4, frac_is)

    assert frac_is.last_z is not None
    assert np.allclose(omega, frac_is.last_z + (1.0 + 2.0j))


def test_z_array_tracking_default_and_center_side_branch(monkeypatch):
    def fake_map_chi_to_z_line(chi, endpoints):
        if np.isscalar(chi):
            # Scalar probe used only for side selection.
            return 0.0 + 0.0j
        return chi

    monkeypatch.setattr(const_head_mod.gf, "map_chi_to_z_line", fake_map_chi_to_z_line)

    frac = _FracStub(center=0.0 + 0.0j)
    ch = ConstantHeadLine("CH6", np.array([1 + 0j, 3 + 0j]), head=1.0, frac0=frac)

    z_default = ch.z_array_tracking(6, offset=1e-2, on_frac_center=False)
    assert len(z_default) == 6
    assert np.allclose(np.abs(z_default), 1.01)

    z_half = ch.z_array_tracking(6, offset=1e-2, on_frac_center=True)
    assert np.all(np.imag(z_half) >= -1e-12)


def test_calc_omega_combines_asym_expansion_and_well(monkeypatch):
    def fake_map(z, endpoints):
        return 0.25 + 0.75j

    def fake_asym(chi, coef):
        return 10.0 + 0.5j

    def fake_well(chi, q):
        return 1.0 + 2.0j

    monkeypatch.setattr(const_head_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(const_head_mod.mf, "asym_expansion", fake_asym)
    monkeypatch.setattr(const_head_mod.mf, "well_chi", fake_well)

    frac = _FracStub()
    ch = ConstantHeadLine("CH7", np.array([0j, 2 + 0j]), head=1.0, frac0=frac)
    ch.q = 3.0

    omega = ch.calc_omega(0.5 + 0.1j)

    assert omega == pytest.approx(11.0 + 2.5j)


def test_calc_w_formula_with_simple_scalar_maps(monkeypatch):
    def fake_map(z, endpoints):
        return 2.0 + 0.0j

    def fake_asym_d1(chi, coef):
        return 3.0 + 0.0j

    monkeypatch.setattr(const_head_mod.gf, "map_z_line_to_chi", fake_map)
    monkeypatch.setattr(const_head_mod.mf, "asym_expansion_d1", fake_asym_d1)

    frac = _FracStub()
    ch = ConstantHeadLine("CH8", np.array([0 + 0j, 2 + 0j]), head=1.0, frac0=frac)
    ch.q = 4.0

    w = ch.calc_w(1.0 + 0.0j)

    expected_base = -3.0 - 1.0 / np.pi
    expected = expected_base * (8.0 / 3.0)
    assert w == pytest.approx(expected)


def test_check_boundary_condition_zero_when_phi_matches_real_omega(monkeypatch):
    def fake_map(chi, endpoints):
        return chi

    monkeypatch.setattr(const_head_mod.gf, "map_chi_to_z_line", fake_map)

    class _FracPhi(_FracStub):
        def calc_omega(self, z, exclude=None):
            return np.asarray(z, dtype=np.complex128) * 0.0 + (6.0 + 1.0j)

    frac = _FracPhi(t=2.0)
    ch = ConstantHeadLine("CH9", np.array([0j, 1 + 0j]), head=3.0, frac0=frac)

    err = ch.check_boundary_condition(n=12)

    assert err == pytest.approx(0.0)


def test_check_chi_crossing_branches(monkeypatch):
    frac = _FracStub()
    ch = ConstantHeadLine("CH10", np.array([0 + 0j, 1 + 0j]), head=1.0, frac0=frac)

    monkeypatch.setattr(const_head_mod.gf, "line_line_intersection", lambda *args: None)
    assert ch.check_chi_crossing(0 + 0j, 1 + 1j) is False

    monkeypatch.setattr(
        const_head_mod.gf, "line_line_intersection", lambda *args: 2 + 0j
    )
    assert ch.check_chi_crossing(0 + 0j, 1 + 1j) is False

    monkeypatch.setattr(
        const_head_mod.gf, "line_line_intersection", lambda *args: 0.5 + 0.0j
    )
    hit = ch.check_chi_crossing(0.5 - 1j, 0.5 + 1j)
    assert hit == pytest.approx(0.5 + 0.0j)
