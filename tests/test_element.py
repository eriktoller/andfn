import logging

import numpy as np
import pytest

import andfn.element as element_mod
from andfn.element import (
    ELEMENT_COLORS,
    Element,
    element_dtype,
    element_index_dtype,
    initiate_elements_array,
    initiate_elements_array_hpc,
)


class _FracStub:
    def __init__(self, _id=0):
        self._id = _id
        self.elements = []
        self.normal = np.array([0.0, 0.0, 1.0])

    def add_element(self, element):
        self.elements.append(element)


class _PlotterStub:
    def __init__(self):
        self.calls = []

    def add_mesh(self, mesh, **kwargs):
        self.calls.append((mesh, kwargs))


@pytest.fixture(autouse=True)
def _quiet_logging():
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        root.setLevel(prev)


def test_initiate_elements_array_defaults():
    arr = initiate_elements_array()

    assert arr.shape == (1,)
    assert arr["_id"][0] == -1
    assert arr["_type"][0] == -1
    assert np.isnan(arr["radius"][0])
    assert np.isnan(arr["center"][0].real)
    assert np.isnan(arr["center"][0].imag)
    assert arr["thetas"][0].shape == (1,)
    assert np.isnan(arr["thetas"][0][0])


def test_initiate_elements_array_hpc_defaults():
    arr = initiate_elements_array_hpc()

    assert arr.shape == (1,)
    assert arr["_id"][0] == -1
    assert np.isnan(arr["radius"][0])
    assert np.all(arr["thetas"][0] == 0.0)
    assert np.all(arr["coef"][0] == 0.0 + 0.0j)
    assert np.all(np.isnan(arr["endpoints0"][0].real))
    assert np.all(np.isnan(arr["endpoints0"][0].imag))


def test_element_init_reset_set_id_and_str_repr():
    frac = _FracStub(_id=10)
    el = Element("E1", 2, 0, frac0=frac)

    assert el in frac.elements
    assert str(el) == "Element: E1"
    assert repr(el) == "Element: E1"

    el.reset(ncoef=3, nint=8)
    assert el.error == 1.0
    assert el.q == 0.0
    assert el.ncoef == 3
    assert el.nint == 8
    assert np.all(el.coef == 0.0 + 0.0j)

    el.set_id(99)
    assert el._id == 99


def test_change_property_allows_index_fields_and_rejects_other_fields():
    frac = _FracStub(_id=1)
    el = Element("E2", 1, 2, frac0=frac)

    el.change_property(_id=11, _type=3)
    assert el._id == 11
    assert el._type == 3

    with pytest.raises(AssertionError, match="Invalid property name"):
        el.change_property(q=5.0)


def test_consolidate_and_unconsolidate_roundtrip_for_core_fields():
    frac0 = _FracStub(_id=7)
    frac1 = _FracStub(_id=9)
    el = Element("E3", 3, 0, frac0=frac0)
    el.frac1 = frac1
    el.q = 1.25
    el.ncoef = 4
    el.nint = 12
    el.error = 0.2

    struc, index = el.consolidate()

    assert struc["frac0"][0] == 7
    assert struc["frac1"][0] == 9
    assert struc["q"][0] == pytest.approx(1.25)
    assert index["label"][0] == "E3"

    new_frac = _FracStub(_id=100)
    restored = Element("tmp", 0, 0, frac0=new_frac)
    restored.frac1 = new_frac

    restored.unconsolidate(struc[0], index[0], [frac0, frac1])

    assert restored.label == "E3"
    assert restored._id == 3
    assert restored._type == 0
    assert restored.frac0 is frac0
    assert restored.frac1 is frac1
    assert restored.q == pytest.approx(1.25)


def test_consolidate_into_writes_given_slot():
    frac0 = _FracStub(_id=2)
    frac1 = _FracStub(_id=4)
    el = Element("E4", 8, 3, frac0=frac0)
    el.frac1 = frac1
    el.q = 3.4

    struc = np.empty(2, dtype=element_dtype)
    index = np.empty(2, dtype=element_index_dtype)

    el.consolidate_into(struc, index, 1)

    assert struc["_id"][1] == 8
    assert struc["frac0"][1] == 2
    assert struc["frac1"][1] == 4
    assert struc["q"][1] == pytest.approx(3.4)
    assert index["label"][1] == "E4"


def test_consolidate_hpc_and_unconsolidate_hpc_roundtrip():
    frac0 = _FracStub(_id=5)
    frac1 = _FracStub(_id=6)
    el = Element("E5", 12, 0, frac0=frac0)
    el.frac1 = frac1
    el.ncoef = 3
    el.nint = 11
    el.coef = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    el.old_coef = np.array([7 + 8j, 9 + 10j, 11 + 12j], dtype=np.complex128)
    el.thetas = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    el.dpsi_corr = np.array([0.01, 0.02], dtype=np.float64)
    el.q = 4.5
    el.error = 0.03

    struc, _ = el.consolidate_hpc()

    assert struc["frac0"][0] == 5
    assert struc["frac1"][0] == 6
    assert np.allclose(struc["coef"][0][:3], el.coef)
    assert np.allclose(struc["old_coef"][0][:3], el.old_coef)
    assert np.allclose(struc["thetas"][0][:3], el.thetas)
    assert np.allclose(struc["dpsi_corr"][0][:2], el.dpsi_corr)

    restored = Element("tmp", 0, 0, frac0=_FracStub(_id=0))
    restored.unconsolidate_hpc(struc[0], np.zeros(1, dtype=element_index_dtype)[0], [])

    assert restored.ncoef == 3
    assert restored.nint == 11
    assert np.allclose(restored.coef, el.coef)
    assert restored.q == pytest.approx(4.5)
    assert restored.error == pytest.approx(0.03)


def test_plot_line_type_uses_line_geometry(monkeypatch):
    line_calls = []

    def fake_map_2d_to_3d(z, frac):
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def fake_line(p0, p1):
        line_calls.append((p0, p1))
        return ("line", p0, p1)

    monkeypatch.setattr(element_mod.gf, "map_2d_to_3d", fake_map_2d_to_3d)
    monkeypatch.setattr(element_mod.pv, "Line", fake_line)

    frac = _FracStub(_id=1)
    el = Element("L1", 1, 0, frac0=frac)
    el.endpoints0 = np.array([0 + 0j, 1 + 1j], dtype=np.complex128)
    plotter = _PlotterStub()

    el.plot(plotter, line_width=2.0, color=None)

    assert len(line_calls) == 1
    assert len(plotter.calls) == 1
    _, kwargs = plotter.calls[0]
    assert kwargs["color"] == ELEMENT_COLORS[0]
    assert kwargs["line_width"] == 2.0


def test_plot_circle_type_uses_polygon_geometry(monkeypatch):
    polygon_calls = []

    def fake_map_2d_to_3d(z, frac):
        return np.array([1.0, 2.0, 3.0])

    def fake_polygon(**kwargs):
        polygon_calls.append(kwargs)
        return ("polygon", kwargs)

    monkeypatch.setattr(element_mod.gf, "map_2d_to_3d", fake_map_2d_to_3d)
    monkeypatch.setattr(element_mod.pv, "Polygon", fake_polygon)

    frac = _FracStub(_id=1)
    el = Element("C1", 1, 2, frac0=frac)
    el.center = 0.5 + 0.25j
    el.radius = 2.5
    plotter = _PlotterStub()

    el.plot(plotter, line_width=1.5, color="#123456")

    assert len(polygon_calls) == 1
    assert polygon_calls[0]["radius"] == pytest.approx(2.5)
    assert np.allclose(polygon_calls[0]["center"], np.array([1.0, 2.0, 3.0]))
    assert len(plotter.calls) == 1
    _, kwargs = plotter.calls[0]
    assert kwargs["color"] == "#123456"
