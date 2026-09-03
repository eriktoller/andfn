import numpy as np
import pytest

import andfn.fracture as fracture_mod
from andfn.element import fracture_dtype, fracture_dtype_hpc, fracture_index_dtype
from andfn.fracture import Fracture, fracture_from_dict


def _make_intersection(frac0, frac1, q=0.0):
    inter = fracture_mod.Intersection.__new__(fracture_mod.Intersection)
    inter._id = 101
    inter.frac0 = frac0
    inter.frac1 = frac1
    inter.q = q
    inter.head = 0.0
    inter.calc_omega = lambda z, f: 3.0 + 1.0j
    inter.calc_w = lambda z, f: 2.0 + 0.0j
    return inter


def _make_const_head(frac, head=10.0, q=0.0):
    ch = fracture_mod.ConstantHeadLine.__new__(fracture_mod.ConstantHeadLine)
    ch._id = 202
    ch.frac0 = frac
    ch.q = q
    ch.head = head
    ch.calc_omega = lambda z: 1.5 + 0.0j
    ch.calc_w = lambda z: 0.5 + 0.0j
    return ch


def _make_well(frac, head=5.0, q=0.0):
    w = fracture_mod.Well.__new__(fracture_mod.Well)
    w._id = 303
    w.frac0 = frac
    w.q = q
    w.head = head
    w.calc_omega = lambda z: -0.5 + 0.0j
    w.calc_w = lambda z: 1.0 + 1.0j
    return w


class _PlainElement:
    def __init__(self, _id, q=0.0):
        self._id = _id
        self.q = q

    def calc_omega(self, z):
        return 4.0 + 0.0j

    def calc_w(self, z):
        return -1.0 + 0.0j


def test_init_without_auto_elements_and_string_repr():
    frac = Fracture(
        label="F1",
        t=2.0,
        radius=4.0,
        center=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 0.0, 2.0]),
        aperture=0.2,
        elements=False,
    )

    assert frac.label == "F1"
    assert str(frac) == "Fracture F1"
    assert repr(frac) == "Fracture F1"
    assert frac.elements == []
    assert np.isclose(np.linalg.norm(frac.normal), 1.0)
    assert np.isclose(np.linalg.norm(frac.x_vector), 1.0)
    assert np.isclose(np.linalg.norm(frac.y_vector), 1.0)


def test_init_creates_bounding_circle_when_elements_is_none(monkeypatch):
    calls = []

    class _SpyBoundingCircle:
        def __init__(self, label, radius, frac0, ncoef=5, nint=10):
            calls.append((label, radius, ncoef, nint))
            stub = _PlainElement(_id=999)
            frac0.add_element(stub)

    monkeypatch.setattr(fracture_mod, "BoundingCircle", _SpyBoundingCircle)

    frac = Fracture(
        "F2",
        t=1.0,
        radius=3.0,
        center=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        ncoef=7,
        nint=9,
    )

    assert calls == [("F2", 3.0, 7, 9)]
    assert len(frac.elements) == 1
    assert frac.elements[0]._id == 999


def test_to_dict_and_fracture_from_dict_roundtrip():
    frac = Fracture(
        "F3",
        t=4.0,
        radius=2.0,
        center=np.array([1.0, 1.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        aperture=0.05,
        elements=False,
    )
    frac.set_id(12)
    frac.constant = 1.25
    frac.add_element(_PlainElement(_id=8))

    d_full = frac.to_dict()
    d_file = frac.to_dict(fracs_file=True)

    assert d_full["_id"] == 12
    assert d_full["elements"] == [8]
    assert d_full["constant"] == pytest.approx(1.25)
    assert d_file["label"] == "F3"
    assert "_id" not in d_file

    restored = fracture_from_dict(d_file)
    assert restored.label == "F3"
    assert restored.t == pytest.approx(4.0)
    assert restored.radius == pytest.approx(2.0)
    assert np.allclose(restored.center, np.array([1.0, 1.0, 1.0]))


def test_add_delete_and_delete_all_elements_with_intersection_behavior():
    f0 = Fracture("A", 1.0, 1.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), elements=False)
    f1 = Fracture("B", 1.0, 1.0, np.ones(3), np.array([0.0, 1.0, 0.0]), elements=False)

    plain = _PlainElement(_id=1)
    f0.add_element(plain)
    f0.add_element(plain)
    assert f0.elements.count(plain) == 1

    inter = _make_intersection(f0, f1, q=2.0)
    f0.elements.append(inter)
    f1.elements.append(inter)

    f0.delete_element(plain)
    assert plain not in f0.elements

    f0.delete_element(inter)
    assert inter not in f0.elements
    assert inter not in f1.elements

    inter2 = _make_intersection(f0, f1, q=1.0)
    f0.elements.append(inter2)
    f1.elements.append(inter2)
    f0.delete_all_elements()

    assert f0.elements == []
    assert inter2 not in f1.elements


def test_discharge_helpers_and_head_range():
    f0 = Fracture("F", 2.0, 1.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), elements=False)
    f1 = Fracture("G", 2.0, 1.0, np.ones(3), np.array([0.0, 1.0, 0.0]), elements=False)

    inter = _make_intersection(f0, f1, q=1.0)
    ch = _make_const_head(f0, head=10.0, q=-2.0)
    well = _make_well(f0, head=7.0, q=3.0)
    plain = _PlainElement(_id=999, q=50.0)

    f0.elements = [inter, ch, well, plain]

    discharge_elements = f0.get_discharge_elements()
    assert len(discharge_elements) == 3
    assert f0.get_discharge_entries() == 10  # n^2 + n_intersections where n=3
    assert f0.get_total_discharge() == pytest.approx(6.0)
    assert f0.check_discharge() == pytest.approx(2.0)
    assert f0.get_max_min_head() == [10.0, 7.0]


def test_calc_omega_calc_w_exclude_velocity_and_head():
    frac = Fracture(
        "F4",
        t=2.0,
        radius=1.0,
        center=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        aperture=0.5,
        elements=False,
    )
    other = Fracture(
        "H", 1.0, 1.0, np.ones(3), np.array([0.0, 1.0, 0.0]), elements=False
    )

    inter = _make_intersection(frac, other, q=0.0)
    plain = _PlainElement(_id=2)
    frac.elements = [inter, plain]
    frac.constant = 2.0

    omega_all = frac.calc_omega(0.5 + 0.1j)
    omega_ex = frac.calc_omega(0.5 + 0.1j, exclude=plain)
    w_all = frac.calc_w(0.5 + 0.1j)
    vel = frac.calc_velocity(0.5 + 0.1j)

    assert omega_all == pytest.approx(9.0 + 1.0j)
    assert omega_ex == pytest.approx(5.0 + 1.0j)
    assert w_all == pytest.approx(1.0 + 0.0j)
    assert vel == pytest.approx(2.0)

    frac.constant = 4.0
    # calc_head uses full calc_omega (constant + element contributions).
    assert frac.calc_head(0.0 + 0.0j) == pytest.approx(5.5)
    assert frac.phi_from_head(3.0) == pytest.approx(6.0)
    assert frac.head_from_phi(6.0) == pytest.approx(3.0)


def test_set_new_label_and_reset():
    frac = Fracture(
        "old", 1.0, 1.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), elements=False
    )
    frac.constant = 5.0

    frac.set_new_label("new")
    frac.reset()

    assert frac.label == "new"
    assert frac.constant == pytest.approx(0.0)


def test_calc_flow_net_shapes_and_values_for_constant_omega():
    frac = Fracture(
        "flow", 1.0, 2.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), elements=False
    )
    frac.calc_omega = lambda z, exclude=None: (
        np.zeros_like(z, dtype=np.complex128) + (3.0 + 4.0j)
    )

    omega_fn, x_array, y_array = frac.calc_flow_net(n_points=5, margin=0.2)

    assert omega_fn.shape == (5, 5)
    assert x_array.shape == (5,)
    assert y_array.shape == (5,)
    assert np.allclose(omega_fn, 3.0 + 4.0j)


def test_consolidate_and_hpc_consolidation_fields():
    frac = Fracture(
        "C",
        3.0,
        2.0,
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 1.0]),
        elements=False,
    )
    frac.set_id(42)
    frac.constant = 1.75
    frac.elements = [_PlainElement(_id=5), _PlainElement(_id=9)]

    arr, idx = frac.consolidate()
    assert arr["_id"][0] == 42
    assert arr["t"][0] == pytest.approx(3.0)
    assert arr["elements"][0].tolist() == [5, 9]
    assert idx["label"][0] == "C"

    arr_hpc, idx_hpc = frac.consolidate_hpc()
    assert arr_hpc["_id"][0] == 42
    assert arr_hpc["nelements"][0] == 2
    assert arr_hpc["elements"][0][:2].tolist() == [5, 9]
    assert idx_hpc["_id"][0] == 42


def test_consolidate_into_hpc_and_unconsolidate_hpc_filters_elements():
    frac = Fracture(
        "UH", 2.0, 1.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), elements=False
    )
    frac.set_id(7)
    frac.constant = 0.5
    frac.elements = [_PlainElement(_id=1), _PlainElement(_id=4)]

    arr = np.zeros(1, dtype=fracture_dtype_hpc)
    idx = np.zeros(1, dtype=fracture_index_dtype)
    frac.consolidate_into_hpc(arr, idx, 0)

    assert arr["nelements"][0] == 2
    assert idx["label"][0] == "UH"

    target = Fracture(
        "tmp", 1.0, 1.0, np.zeros(3), np.array([0.0, 1.0, 0.0]), elements=False
    )
    keep = _PlainElement(_id=4)
    drop = _PlainElement(_id=99)
    target.elements = [keep, drop]
    target.unconsolidate_hpc(arr[0], idx[0])

    assert [e._id for e in target.elements] == [4]
    assert target._id == 7
    assert target.constant == pytest.approx(0.5)


def test_consolidate_into_non_hpc_dtype_currently_raises_type_error():
    frac = Fracture(
        "X", 1.0, 1.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), elements=False
    )
    frac.elements = [_PlainElement(_id=1)]

    arr = np.zeros(1, dtype=fracture_dtype)
    idx = np.zeros(1, dtype=fracture_index_dtype)

    with pytest.raises(TypeError, match="does not support item assignment"):
        frac.consolidate_into(arr, idx, 0)
