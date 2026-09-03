import numpy as np
import pytest

import andfn.regions as regions_mod
from andfn.regions import Panel, RectangularRegion, Region


class _PlotterStub:
    def __init__(self):
        self.mesh_calls = []

    def add_mesh(self, mesh, **kwargs):
        self.mesh_calls.append((mesh, kwargs))


class _FracStub:
    def __init__(self, label, center, radius=1.0, normal=None):
        self.label = label
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.normal = (
            np.array([0.0, 0.0, 1.0])
            if normal is None
            else np.array(normal, dtype=float)
        )
        self.elements = []


def _make_const_head_like(q):
    el = regions_mod.ConstantHeadLine.__new__(regions_mod.ConstantHeadLine)
    el.q = q
    return el


def test_region_and_panel_string_state():
    region = Region("R0", _region_type=99)
    panel = Panel(
        "P1",
        center=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        x_vec=np.array([1.0, 0.0, 0.0]),
        y_vec=np.array([0.0, 1.0, 0.0]),
    )

    assert str(region) == "Region: R0"
    assert str(panel) == "Panel: P1"
    assert np.allclose(panel.center, np.array([1.0, 2.0, 3.0]))


def test_rectangular_region_init_vertices_and_orthogonality_guard():
    region = RectangularRegion(
        "RR1",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=2,
        yl=4,
        zl=6,
    )

    assert region.vertices.shape == (8, 3)
    assert region.faces.shape == (30,)
    assert region.faces_dict["top"] == [0, 1, 2, 3]
    assert region.fracs == []
    assert region.elements == []

    with pytest.raises(ValueError, match="x and y vectors are not orthogonal"):
        RectangularRegion(
            "bad",
            center=[0, 0, 0],
            x_vec=[1, 0, 0],
            y_vec=[1, 0, 0],
            z_vec=[0, 0, 1],
            xl=1,
            yl=1,
            zl=1,
        )


def test_get_total_flow_rotate_map_and_check_point():
    region = RectangularRegion(
        "RR2",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=2,
        yl=2,
        zl=2,
    )
    region.elements = [
        _make_const_head_like(-2.0),
        _make_const_head_like(3.0),
        object(),
    ]

    assert region.get_total_flow() == pytest.approx(5.0)
    assert np.allclose(region.map_point([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert region.check_point([0.5, 0.5, 0.5])
    assert not region.check_point([2.0, 0.0, 0.0])

    region.rotate(90, [0, 0, 1])
    assert np.allclose(region.x_vec, np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert np.allclose(region.y_vec, np.array([-1.0, 0.0, 0.0]), atol=1e-12)


def test_plot_and_plot_face(monkeypatch):
    created = []

    def fake_polydata(vertices, faces):
        created.append((np.array(vertices), np.array(faces)))
        return ("poly", np.array(vertices), np.array(faces))

    monkeypatch.setattr(regions_mod.pv, "PolyData", fake_polydata)

    region = RectangularRegion(
        "RR3",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=2,
        yl=2,
        zl=2,
    )
    plotter = _PlotterStub()

    region.plot(plotter, opacity=0.25)
    region.plot_face(plotter, "top", opacity=0.75)

    assert len(plotter.mesh_calls) == 2
    assert plotter.mesh_calls[0][1]["opacity"] == pytest.approx(0.25)
    assert plotter.mesh_calls[1][1]["opacity"] == pytest.approx(0.75)

    with pytest.raises(ValueError, match="Face must be"):
        region.plot_face(plotter, "diagonal")


def test_check_fractures_and_possible_intersections():
    region = RectangularRegion(
        "RR4",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=2,
        yl=2,
        zl=2,
    )
    f_in = _FracStub("in", [0.0, 0.0, 0.0], radius=0.1)
    f_edge = _FracStub("edge", [0.9, 0.0, 0.0], radius=0.1)
    f_out = _FracStub("out", [3.0, 0.0, 0.0], radius=0.1)

    inside, outside = region.check_fractures([f_in, f_edge, f_out])
    assert inside == [f_in, f_edge]
    assert outside == [f_out]
    assert region.possible_intersections(f_in)
    assert not region.possible_intersections(f_out)


def test_assign_elements_helpers_and_static_geometry(monkeypatch):
    created = []

    class _FakeConstantHeadLine:
        def __init__(self, label, endpoints, head, frac):
            self.label = label
            self.endpoints0 = endpoints
            self.head = head
            self.frac0 = frac
            created.append(self)

    monkeypatch.setattr(regions_mod, "ConstantHeadLine", _FakeConstantHeadLine)
    monkeypatch.setattr(
        regions_mod.gf, "map_3d_to_2d", lambda pnt, frac: pnt[0] + 1j * pnt[1]
    )

    region = RectangularRegion(
        "RR5",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=2,
        yl=2,
        zl=2,
    )
    frac = _FracStub("F", [0, 0, 0], radius=2.0)

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])

    region.assign_elements(frac, [p0, p2], [p0, p1, p2], head=7.0)
    assert created[-1].label == "tunnel_RR5_frac_F_0"
    assert np.allclose(created[-1].endpoints0, np.array([0.0 + 0.0j, 2.0 + 0.0j]))

    region.assign_elements(frac, [p0, p1], [p0, p1], head=8.0)
    assert created[-1].label == "tunnel_RR5_frac_F_11"
    assert frac in region.fracs

    monkeypatch.setattr(regions_mod.gf, "map_3d_to_2d", lambda pnt, frac: 0.5 + 0.0j)
    assert RectangularRegion.inside_fracture(np.array([0, 0, 0]), frac)
    monkeypatch.setattr(regions_mod.gf, "map_3d_to_2d", lambda pnt, frac: 3.0 + 0.0j)
    assert not RectangularRegion.inside_fracture(np.array([0, 0, 0]), frac)

    hit = RectangularRegion.line_plane_intersection(
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    miss = RectangularRegion.line_plane_intersection(
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.allclose(hit, np.array([0.0, 0.0, 0.0]))
    assert miss is None
