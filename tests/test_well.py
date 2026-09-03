import numpy as np
import pytest

import andfn.well as well_mod
from andfn.well import Well


class _FracStub:
    def __init__(self, label="F", _id=1, t=2.0):
        self.label = label
        self._id = _id
        self.t = t
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def phi_from_head(self, head):
        return head * self.t


def test_init_and_to_dict_register_well_and_compute_phi():
    frac = _FracStub(label="fracA", t=3.0)
    well = Well("W1", radius=0.5, center=1.0 + 2.0j, head=4.0, frac0=frac)

    assert well in frac.elements
    assert well._type == 2
    assert well.q == pytest.approx(0.0)
    assert well.phi == pytest.approx(12.0)

    data = well.to_dict()
    assert data["label"] == "W1"
    assert data["radius"] == pytest.approx(0.5)
    assert data["center"] == pytest.approx(1.0 + 2.0j)
    assert data["head"] == pytest.approx(4.0)
    assert data["phi"] == pytest.approx(12.0)
    assert data["frac0"] == "fracA"


def test_discharge_term_and_sampling_arrays(monkeypatch):
    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.array(
            [1.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128
        ),
    )
    monkeypatch.setattr(well_mod.mf, "well_chi", lambda chi, q: 2.0 * chi + 1j)

    frac = _FracStub()
    well = Well("W2", radius=2.0, center=1.0 + 0.0j, head=1.0, frac0=frac)

    assert well.discharge_term(np.array([0.0 + 0.0j, 1.0 + 0.0j])) == pytest.approx(4.0)

    z = well.z_array(4)
    z_tracking = well.z_array_tracking(4, offset=0.1)
    assert np.allclose(np.abs(z - (1.0 + 0.0j)), 2.0)
    assert np.allclose(np.abs(z_tracking - (1.0 + 0.0j)), 2.2)


def test_calc_omega_scalar_and_array_inside_masking(monkeypatch):
    frac = _FracStub(t=5.0)
    well = Well("W3", radius=1.0, center=0.0 + 0.0j, head=2.0, frac0=frac)
    well.q = 6.0

    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.complex128(0.8 + 0.0j),
    )
    monkeypatch.setattr(well_mod.mf, "well_chi", lambda chi, q: 99.0 + 1.0j)
    inside = well.calc_omega(0.2 + 0.0j)
    assert inside == pytest.approx(10.0 + 0.0j)

    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.array(
            [0.7 + 0.0j, 1.2 + 0.0j], dtype=np.complex128
        ),
    )
    monkeypatch.setattr(
        well_mod.mf,
        "well_chi",
        lambda chi, q: np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex128),
    )
    omega = well.calc_omega(np.array([0.1 + 0.0j, 2.0 + 0.0j], dtype=np.complex128))
    assert omega[0] == pytest.approx(10.0 + 0.0j)
    assert omega[1] == pytest.approx(2.0 + 2.0j)


def test_calc_w_scalar_and_array_inside_masking(monkeypatch):
    frac = _FracStub()
    well = Well("W4", radius=2.0, center=0.0 + 0.0j, head=1.0, frac0=frac)
    well.q = 4.0

    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.complex128(2.0 + 0.0j),
    )
    scalar_w = well.calc_w(4.0 + 0.0j)
    assert scalar_w == pytest.approx((-4.0 / (4.0 * np.pi)) / 2.0)

    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: np.array(
            [0.5 + 0.0j, 2.0 + 0.0j], dtype=np.complex128
        ),
    )
    array_w = well.calc_w(np.array([0.5 + 0.0j, 2.0 + 0.0j], dtype=np.complex128))
    assert np.isnan(array_w[0].real)
    assert np.isnan(array_w[0].imag)
    assert array_w[1] == pytest.approx((-4.0 / (4.0 * np.pi)) / 2.0)


def test_boundary_condition_and_check_chi_crossing(monkeypatch):
    frac = _FracStub()
    well = Well("W5", radius=1.5, center=1.0 + 0.0j, head=1.0, frac0=frac)

    assert well.check_boundary_condition() == pytest.approx(0.0)

    mapped = []
    chi_values = iter([0.0 + 0.0j, 2.0 + 0.0j])
    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: next(chi_values),
    )
    monkeypatch.setattr(
        well_mod.gf, "line_circle_intersection", lambda a, b, r: (None, None)
    )
    assert well.check_chi_crossing(0.0 + 0.0j, 3.0 + 0.0j) is False

    chi_values = iter([0.0 + 0.0j, 2.0 + 0.0j])
    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: next(chi_values),
    )
    monkeypatch.setattr(
        well_mod.gf,
        "line_circle_intersection",
        lambda a, b, r: (3.0 + 0.0j, -3.0 + 0.0j),
    )
    assert well.check_chi_crossing(0.0 + 0.0j, 2.0 + 0.0j) is False

    chi_values = iter([0.0 + 0.0j, 2.0 + 0.0j])
    monkeypatch.setattr(
        well_mod.gf,
        "map_z_circle_to_chi",
        lambda z, radius, center: next(chi_values),
    )

    def fake_map_back(chi, radius, center):
        mapped.append(chi)
        return chi + center

    monkeypatch.setattr(
        well_mod.gf,
        "line_circle_intersection",
        lambda a, b, r: (1.0 + 0.0j, -1.0 + 0.0j),
    )
    monkeypatch.setattr(well_mod.gf, "map_chi_to_z_circle", fake_map_back)

    hit = well.check_chi_crossing(0.0 + 0.0j, 2.0 + 0.0j)
    assert hit == pytest.approx((-1.0 - 1e-10) + (1.0 + 0.0j))
    assert mapped[1] == pytest.approx(-1.0 - 1e-10)
