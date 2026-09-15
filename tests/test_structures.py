import numpy as np
import pytest

import andfn.structures as structures_mod
from andfn.structures import (
    ConstantHeadPrism,
    ImpermeablePrism,
    RegularPolygonPrism,
    Structure,
)


class _PlotterStub:
    def __init__(self):
        self.point_calls = []
        self.mesh_calls = []

    def add_points(self, points, **kwargs):
        self.point_calls.append((np.array(points), kwargs))

    def add_mesh(self, mesh, **kwargs):
        self.mesh_calls.append((mesh, kwargs))


class _FracStub:
    def __init__(self, label="F", center=None, normal=None, radius=1.0):
        self.label = label
        self.center = np.zeros(3) if center is None else np.array(center, dtype=float)
        self.normal = (
            np.array([0.0, 0.0, 1.0])
            if normal is None
            else np.array(normal, dtype=float)
        )
        self.radius = radius
        self.elements = []


def test_structure_str_and_regular_polygon_prism_guard_and_geometry():
    structure = Structure("S1")
    assert str(structure) == "Structure: S1"

    with pytest.raises(ValueError, match="n_sides must be at least 3"):
        RegularPolygonPrism(
            "bad",
            radius=1.0,
            start=np.array([0.0, 0.0, 0.0]),
            end=np.array([1.0, 0.0, 0.0]),
            n_sides=2,
            _structure_type=0,
        )

    prism = RegularPolygonPrism(
        "RP1",
        radius=1.0,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([0.0, 0.0, 2.0]),
        n_sides=4,
        _structure_type=0,
    )
    length, direction, center = prism.get_lvc()
    assert length == pytest.approx(2.0)
    assert np.allclose(direction, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(center, np.array([0.0, 0.0, 1.0]))
    assert prism.vertices.shape == (8, 3)
    assert prism.faces.ndim == 1
    assert np.allclose(prism.vertices[:4, 2], 0.0)
    assert np.allclose(prism.vertices[4:, 2], 2.0)


def test_plot_possible_intersections_and_static_helpers(monkeypatch):
    monkeypatch.setattr(
        structures_mod.pv,
        "PolyData",
        lambda vertices, faces: ("poly", np.array(vertices), np.array(faces)),
    )

    prism = RegularPolygonPrism(
        "RP2",
        radius=1.0,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([2.0, 0.0, 0.0]),
        n_sides=4,
        _structure_type=0,
    )
    plotter = _PlotterStub()
    prism.plot(plotter, opacity=0.4)

    assert len(plotter.point_calls) == 1
    assert len(plotter.mesh_calls) == 1
    assert plotter.mesh_calls[0][1]["opacity"] == pytest.approx(0.4)

    frac = _FracStub(radius=0.25)

    def fake_map_3d_to_2d(point, frac_obj):
        if np.allclose(point, prism.start):
            return 0.0 + 1.0j
        return 2.0 + 1.0j

    monkeypatch.setattr(structures_mod.gf, "map_3d_to_2d", fake_map_3d_to_2d)
    assert prism.possible_intersections(frac) is False
    frac.radius = 2.0
    assert prism.possible_intersections(frac) is True

    monkeypatch.setattr(
        structures_mod.gf, "map_3d_to_2d", lambda pnt, frac_obj: 0.5 + 0.0j
    )
    assert RegularPolygonPrism.inside_fracture(np.array([0, 0, 0]), frac) is True
    monkeypatch.setattr(
        structures_mod.gf, "map_3d_to_2d", lambda pnt, frac_obj: 3.0 + 0.0j
    )
    assert RegularPolygonPrism.inside_fracture(np.array([0, 0, 0]), frac) is False

    hit = RegularPolygonPrism.line_plane_intersection(
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    miss = RegularPolygonPrism.line_plane_intersection(
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.allclose(hit, np.array([0.0, 0.0, 0.0]))
    assert miss is None


def test_regular_polygon_prism_assign_elements_not_implemented():
    prism = RegularPolygonPrism(
        "RP3",
        radius=1.0,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([1.0, 0.0, 0.0]),
        n_sides=4,
        _structure_type=0,
    )

    with pytest.raises(NotImplementedError):
        prism.assign_elements(_FracStub(), [], [])


def test_constant_head_prism_assign_elements_branches(monkeypatch):
    created = []

    class _FakeConstantHeadLine:
        def __init__(self, label, endpoints, head, frac):
            self.label = label
            self.endpoints0 = endpoints
            self.head = head
            self.frac0 = frac
            created.append(self)

    monkeypatch.setattr(structures_mod, "ConstantHeadLine", _FakeConstantHeadLine)
    monkeypatch.setattr(
        structures_mod.gf, "map_3d_to_2d", lambda pnt, frac: pnt[0] + 1j * pnt[1]
    )

    prism = ConstantHeadPrism(
        "CP1",
        radius=1.0,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([1.0, 0.0, 0.0]),
        head=9.0,
        n_sides=4,
    )
    frac = _FracStub(label="F1")
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])

    prism.assign_elements(frac, [p0, p2], [p0, p1, p2])
    assert len(created) == 1
    assert created[0].label == "tunnel_CP1_frac_F1_0"

    prism.assign_elements(frac, [p0, p1], [p0, p1])
    assert created[-2].label == "tunnel_CP1_frac_F1_0"
    assert created[-1].label == "tunnel_CP1_frac_F1_2"
    assert prism.fracs == [frac, frac]


def test_impermeable_prism_assign_elements(monkeypatch):
    created = []

    class _FakeImpermeableLine:
        def __init__(self, label, endpoints, frac):
            self.label = label
            self.endpoints0 = endpoints
            self.frac0 = frac
            created.append(self)

    monkeypatch.setattr(structures_mod, "ImpermeableLine", _FakeImpermeableLine)
    monkeypatch.setattr(
        structures_mod.gf, "map_3d_to_2d", lambda pnt, frac: pnt[0] + 1j * pnt[1]
    )

    prism = ImpermeablePrism(
        "IP1",
        radius=1.0,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([1.0, 0.0, 0.0]),
        n_sides=4,
    )
    frac = _FracStub(label="F2")
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])

    prism.assign_elements(frac, [p0, p1], [p0, p1])
    assert len(created) == 2
    assert created[0].label == "tunnel_IP1_frac_F2_0"
    assert created[1].label == "tunnel_IP1_frac_F2_2"


def test_check_internal_elements_trims_crossing_endpoints(monkeypatch):
    prism = ConstantHeadPrism(
        "CP2",
        radius=1.0,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([1.0, 0.0, 0.0]),
        head=1.0,
        n_sides=4,
    )
    frac = _FracStub(label="F3")

    elem1 = type("LineStub", (), {})()
    elem1.frac0 = frac
    elem1.endpoints0 = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    prism.elements = [elem1]

    inter = structures_mod.Intersection.__new__(structures_mod.Intersection)
    inter.frac0 = frac
    inter.frac1 = _FracStub(label="other")
    inter.endpoints0 = np.array([1.0 - 1.0j, 1.0 + 1.0j], dtype=np.complex128)
    inter.endpoints1 = np.array([-1.0 - 1.0j, -1.0 + 1.0j], dtype=np.complex128)
    frac.elements = [inter]

    monkeypatch.setattr(
        structures_mod.gf, "line_line_intersection", lambda *args: 1.0 + 0.0j
    )

    def fake_map_z_line_to_chi(z, endpoints):
        if np.allclose(endpoints, elem1.endpoints0):
            if z == inter.endpoints0[0]:
                return -1.0j
            return 1.0 + 1.0j
        return 1.0j

    monkeypatch.setattr(structures_mod.gf, "map_z_line_to_chi", fake_map_z_line_to_chi)

    prism.check_internal_elements(frac, atol=0.1)
    assert inter.endpoints0[0] == pytest.approx(1.0 + 0.0j)
