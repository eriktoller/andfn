import json

import numpy as np
import pytest

from andfn import io as io_mod


class _DummyFracture:
    def __init__(self, label, radius, center=None, elements=None):
        self.label = label
        self.radius = radius
        self.center = np.array(center if center is not None else [0.0, 0.0, 0.0])
        self.elements = [] if elements is None else elements

    def to_dict(self, fracs_file=False):
        if fracs_file:
            return {
                "label": self.label,
                "t": 1.0,
                "radius": float(self.radius),
                "center": self.center.tolist(),
                "normal": [0.0, 0.0, 1.0],
                "aperture": 0.1,
            }
        return {"label": self.label}


class _DummyDFN:
    def __init__(self, fractures):
        self.fractures = fractures


class _IOStub(io_mod.IO):
    def __init__(self):
        self.constants = {"SE_FACTOR": 0.9, "NCOEF": 5, "NINT": 10}
        self.added = None

    def add_fracture(self, fractures):
        self.added = fractures


def test_find_column_matches_alias_case_insensitively():
    class _DF:
        columns = ("Radius", "EAST", "northing")

    assert io_mod.find_column(_DF, ["r", "radius"]) == "Radius"
    assert io_mod.find_column(_DF, ["x", "east"]) == "EAST"
    assert io_mod.find_column(_DF, ["y", "northing"]) == "northing"


def test_infer_columns_respects_existing_kwargs():
    class _DF:
        columns = ("radius", "x", "y", "z", "t", "aperture", "strike", "dip")

    cols = io_mod.infer_columns(_DF, radius_str="custom_radius")
    assert cols["radius_str"] == "custom_radius"
    assert cols["x_str"] == "x"
    assert cols["e_str"] == "aperture"


def test_numpy_converter_for_scalar_and_array_types():
    assert io_mod.numpy_converter(np.int64(3)) == 3
    assert io_mod.numpy_converter(np.float64(1.5)) == 1.5
    assert io_mod.numpy_converter(np.complex128(1.0 + 2.0j)) == {
        "real": 1.0,
        "imag": 2.0,
    }
    assert io_mod.numpy_converter(np.array([1, 2, 3])) == [1, 2, 3]
    assert io_mod.numpy_converter(np.array([1.0 + 1.0j])) == [
        {"real": 1.0, "imag": 1.0}
    ]


def test_numpy_converter_raises_for_unsupported_type():
    with pytest.raises(TypeError):
        io_mod.numpy_converter({"not": "supported"})


def test_polygon_center_and_equivalent_radius_helpers():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    )

    area = io_mod.polygon_area_3d(vertices)
    center = io_mod.center_from_vertices(vertices)
    req = io_mod.equivalent_radius(area)

    np.testing.assert_allclose(area, 4.0)
    np.testing.assert_allclose(center, np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(req, np.sqrt(4.0 / np.pi))


def test_export_fractures_adds_fracs_extension(tmp_path):
    out = tmp_path / "network.json"
    dfn = _DummyDFN([_DummyFracture("f0", 2.0)])

    io_mod.export_fractures(dfn, str(out))

    fracs_path = tmp_path / "network.fracs"
    assert fracs_path.exists()
    payload = json.loads(fracs_path.read_text())
    assert payload[0]["label"] == "f0"


def test_import_fractures_from_json_uses_fracture_from_dict(monkeypatch, tmp_path):
    src = tmp_path / "input.fracs"
    src.write_text(json.dumps([{"label": "a"}, {"label": "b"}]))

    monkeypatch.setattr(io_mod, "fracture_from_dict", lambda d: f"frac:{d['label']}")

    fracs = io_mod.import_fractures_from_json(str(src))
    assert fracs == ["frac:a", "frac:b"]


def test_import_fractures_from_csv_missing_required_column_raises(
    monkeypatch, tmp_path
):
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "t": [1.0],
            "aperture": [0.1],
            "strike": [10.0],
            "dip": [20.0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: df)

    with pytest.raises(ValueError, match="Column for 'radius_str' not found"):
        io_mod.import_fractures_from_csv(str(tmp_path / "f.csv"))


def test_import_fractures_from_csv_missing_orientation_raises(monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "radius": [2.0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "t": [1.0],
            "aperture": [0.1],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: df)

    with pytest.raises(
        ValueError,
        match="Columns for either 'trend' and 'plunge' or 'strike' and 'dip'",
    ):
        io_mod.import_fractures_from_csv(str(tmp_path / "f.csv"))


def test_import_fractures_from_csv_trend_plunge_branch(monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "radius": [2.0],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "t": [4.0],
            "aperture": [0.2],
            "trend": [15.0],
            "plunge": [25.0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: df)

    trend_calls = []

    def _fake_trend_plunge(tr, pl):
        trend_calls.append((tr, pl))
        return np.array([0.0, 0.0, 1.0])

    class _CSVFracture:
        def __init__(self, label, t, radius, center, normal, aperture):
            self.label = label
            self.t = t
            self.radius = radius
            self.center = np.array(center)
            self.normal = np.array(normal)
            self.aperture = aperture

    monkeypatch.setattr(io_mod.gf, "convert_trend_plunge_to_normal", _fake_trend_plunge)
    monkeypatch.setattr(io_mod, "Fracture", _CSVFracture)

    fracs = io_mod.import_fractures_from_csv(str(tmp_path / "f.csv"))

    assert len(fracs) == 1
    assert trend_calls == [(15.0, 25.0)]
    assert fracs[0].radius == 2.0


def test_import_fractures_from_file_validates_path_and_extension(tmp_path):
    io_instance = _IOStub()

    with pytest.raises(FileNotFoundError):
        io_instance.import_fractures_from_file(str(tmp_path / "missing.csv"))

    bad = tmp_path / "bad.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="not a valid fracture file"):
        io_instance.import_fractures_from_file(str(bad))


def test_import_fractures_from_file_routes_to_get_connected_fractures(
    monkeypatch, tmp_path
):
    f0 = _DummyFracture("f0", 1.0, [0.0, 0.0, 0.0])
    f1 = _DummyFracture("f1", 2.0, [1.0, 0.0, 0.0])

    path = tmp_path / "input.fracs"
    path.write_text("[]")

    io_instance = _IOStub()

    monkeypatch.setattr(io_mod, "import_fractures_from_json", lambda _p: [f0, f1])
    monkeypatch.setattr(io_mod.sp.spatial, "KDTree", lambda centers: object())

    calls = {}

    def _fake_get_connected(fracs, se_factor, ncoef, nint, fracture_surface, tolerance):
        calls["surface"] = fracture_surface
        calls["radii"] = [f.radius for f in fracs]
        return fracs

    monkeypatch.setattr(io_mod.gf, "get_connected_fractures", _fake_get_connected)
    monkeypatch.setattr(io_mod.gf, "remove_isolated_fractures", lambda frs: frs)

    io_instance.import_fractures_from_file(str(path), starting_frac=0)

    assert calls["surface"].radius == 2.0
    assert calls["radii"] == [2.0, 1.0]
    assert io_instance.added[0].radius == 2.0


def test_import_fractures_from_file_routes_to_intersections_and_removes_isolated(
    monkeypatch, tmp_path
):
    f0 = _DummyFracture("f0", 1.0, [0.0, 0.0, 0.0])
    f1 = _DummyFracture("f1", 3.0, [1.0, 0.0, 0.0])
    f2 = _DummyFracture("f2", 2.0, [2.0, 0.0, 0.0])

    path = tmp_path / "input.csv"
    path.write_text("x")

    io_instance = _IOStub()

    monkeypatch.setattr(
        io_mod, "import_fractures_from_csv", lambda *_args, **_kwargs: [f0, f1, f2]
    )
    monkeypatch.setattr(io_mod.sp.spatial, "KDTree", lambda centers: object())

    seen = {}

    def _fake_get_intersections(fracs, se_factor, ncoef, nint, tolerance, tree):
        seen["radii"] = [f.radius for f in fracs]
        return fracs

    monkeypatch.setattr(
        io_mod.gf, "get_fracture_intersections", _fake_get_intersections
    )
    monkeypatch.setattr(
        io_mod.gf,
        "remove_isolated_fractures",
        lambda frs: [f for f in frs if f.radius >= 2.0],
    )

    io_instance.import_fractures_from_file(str(path), remove_isolated=True)

    assert seen["radii"] == [3.0, 2.0, 1.0]
    assert [f.radius for f in io_instance.added] == [3.0, 2.0]


def test_import_fractures_from_file_fab_branch(monkeypatch, tmp_path):
    f0 = _DummyFracture("f0", 1.0, [0.0, 0.0, 0.0])
    path = tmp_path / "input.fab"
    path.write_text("x")

    io_instance = _IOStub()

    monkeypatch.setattr(io_mod, "import_fractures_from_fab", lambda _p: [f0])
    monkeypatch.setattr(io_mod.sp.spatial, "KDTree", lambda centers: object())
    monkeypatch.setattr(
        io_mod.gf, "get_fracture_intersections", lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(io_mod.gf, "remove_isolated_fractures", lambda frs: frs)

    io_instance.import_fractures_from_file(str(path))
    assert io_instance.added == [f0]
