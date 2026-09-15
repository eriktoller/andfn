import numpy as np
import pytest

import andfn.dfn as dfn_mod
from andfn.dfn import DFN
from andfn.fracture import Fracture


def _make_frac(label, center, radius=1.0, t=1.0):
    return Fracture(
        label=label,
        t=t,
        radius=radius,
        center=np.array(center, dtype=float),
        normal=np.array([0.0, 0.0, 1.0]),
        aperture=1.0,
        elements=False,
    )


def _typed_element(cls, _type, frac0, frac1=None, q=0.0, head=0.0):
    e = cls.__new__(cls)
    e._id = -1
    e._type = _type
    e.frac0 = frac0
    if frac1 is not None:
        e.frac1 = frac1
    e.q = q
    e.head = head
    e.endpoints0 = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    e.endpoints1 = np.array([0.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128)
    return e


def test_init_str_and_set_kwargs_updates_constants_and_attrs():
    dfn = DFN("dfnA", discharge_int=25, MAX_ITERATIONS=77, custom_value=9)

    assert str(dfn) == "DFN: dfnA"
    assert dfn.discharge_int == 25
    assert dfn.constants["MAX_ITERATIONS"] == 77
    assert dfn.custom_value == 9


def test_add_fracture_assigns_ids_and_tree_property():
    dfn = DFN("dfnB")
    f0 = _make_frac("f0", [0, 0, 0])
    f1 = _make_frac("f1", [2, 0, 0])

    dfn.add_fracture([f0, f1])

    assert dfn.number_of_fractures == 2
    assert dfn.fractures[0]._id == 0
    assert dfn.fractures[1]._id == 1
    assert dfn.tree is not None


def test_get_elements_order_counts_and_number_of_elements_filters():
    dfn = DFN("dfnC")
    f0 = _make_frac("f0", [0, 0, 0])
    f1 = _make_frac("f1", [1, 0, 0])

    e_b = _typed_element(dfn_mod.BoundingCircle, 1, f0)
    e_ic = _typed_element(dfn_mod.ImpermeableCircle, 4, f0)
    e_il = _typed_element(dfn_mod.ImpermeableLine, 5, f0)
    e_ch = _typed_element(dfn_mod.ConstantHeadLine, 3, f0, q=2.0, head=10.0)
    e_w = _typed_element(dfn_mod.Well, 2, f0, q=-4.0, head=5.0)
    e_i = _typed_element(dfn_mod.Intersection, 0, f0, frac1=f1, q=7.0)

    f0.elements = [e_b, e_ic, e_il, e_ch, e_w, e_i]
    f1.elements = [e_i]
    dfn.add_fracture([f0, f1])

    dfn.get_elements()

    assert [e._type for e in dfn.elements] == [1, 4, 5, 3, 2, 0]
    assert np.array_equal(dfn.ntype_element, np.array([1, 1, 1, 1, 1, 1]))
    assert dfn.number_of_elements() == 6
    assert dfn.number_of_elements("well") == 1

    with pytest.raises(ValueError):
        dfn.number_of_elements("unknown")


def test_get_discharge_elements_and_dfn_discharge():
    dfn = DFN("dfnD")
    f0 = _make_frac("f0", [0, 0, 0])
    f1 = _make_frac("f1", [1, 0, 0])

    e_i = _typed_element(dfn_mod.Intersection, 0, f0, frac1=f1, q=10.0)
    e_w = _typed_element(dfn_mod.Well, 2, f0, q=-4.0)
    e_ch = _typed_element(dfn_mod.ConstantHeadLine, 3, f0, q=2.0)

    f0.elements = [e_i, e_w, e_ch]
    f1.elements = [e_i]
    dfn.add_fracture([f0, f1])
    dfn.get_elements()

    dfn.get_discharge_elements()
    assert len(dfn.discharge_elements) == 3
    assert dfn.get_dfn_discharge() == pytest.approx((4.0 + 2.0) / 2.0)


def test_delete_fracture_removes_items_and_reassigns_ids(monkeypatch):
    dfn = DFN("dfnE")
    f0 = _make_frac("f0", [0, 0, 0])
    f1 = _make_frac("f1", [1, 0, 0])
    f2 = _make_frac("f2", [2, 0, 0])

    deleted = []
    monkeypatch.setattr(f1, "delete_all_elements", lambda: deleted.append("f1"))

    dfn.add_fracture([f0, f1, f2])
    dfn.delete_fracture(f1)

    assert deleted == ["f1"]
    assert [f.label for f in dfn.fractures] == ["f0", "f2"]
    assert [f._id for f in dfn.fractures] == [0, 1]


def test_get_fracture_intersections_uses_geometry_and_updates_elements(monkeypatch):
    dfn = DFN("dfnF")
    f0 = _make_frac("f0", [0, 0, 0])
    f1 = _make_frac("f1", [0, 1, 0])
    dfn.add_fracture([f0, f1])

    ep0 = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    ep1 = np.array([0.0 + 1.0j, 2.0 + 1.0j], dtype=np.complex128)
    monkeypatch.setattr(
        dfn_mod.gf, "fracture_intersection", lambda a, b: (ep0.copy(), ep1.copy())
    )
    monkeypatch.setattr(
        dfn_mod.gf, "shorten_line", lambda endpoints, se: endpoints * se
    )

    calls = []

    def fake_intersection(label, endpoints0, endpoints1, frac0, frac1, ncoef, nint):
        calls.append(
            (
                label,
                endpoints0.copy(),
                endpoints1.copy(),
                frac0.label,
                frac1.label,
                ncoef,
                nint,
            )
        )

    monkeypatch.setattr(dfn_mod, "Intersection", fake_intersection)
    monkeypatch.setattr(DFN, "get_elements", lambda self: setattr(self, "elements", []))

    dfn.get_fracture_intersections(ncoef=6, nint=7, se_factor=0.5)

    assert len(calls) == 1
    assert calls[0][0] == "f0_f1"
    assert np.allclose(calls[0][1], ep0 * 0.5)
    assert np.allclose(calls[0][2], ep1 * 0.5)
    assert calls[0][5:] == (6, 7)


def test_boundary_wrapper_methods_delegate_to_geometry(monkeypatch):
    dfn = DFN("dfnG")
    f0 = _make_frac("f0", [0, 0, 0])
    dfn.add_fracture([f0])

    head_calls = []
    imp_calls = []
    monkeypatch.setattr(
        dfn_mod.gf, "set_head_boundary", lambda *args: head_calls.append(args)
    )
    monkeypatch.setattr(
        dfn_mod.gf, "set_impermeable_boundary", lambda *args: imp_calls.append(args)
    )

    dfn.set_constant_head_boundary(
        center=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        radius=5.0,
        head=100.0,
        label="H",
        ncoef=3,
        nint=4,
        tolerance=0.2,
    )
    dfn.set_impermeable_boundary(
        center=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        radius=2.0,
        label="I",
        ncoef=3,
        nint=4,
    )

    assert len(head_calls) == 1
    assert len(imp_calls) == 1
    assert head_calls[0][0] == dfn.fractures
    assert imp_calls[0][0] == dfn.fractures


def test_solve_calls_hpc_and_optional_unconsolidate(monkeypatch):
    dfn = DFN("dfnH")
    calls = []

    monkeypatch.setattr(
        DFN,
        "get_elements",
        lambda self: (
            setattr(self, "elements", [object()]),
            setattr(self, "ntype_element", np.zeros(6, dtype=int)),
        ),
    )

    def fake_consolidate(self, hpc=False):
        self.fractures_struc_array_hpc = np.zeros(
            1, dtype=np.dtype([("_id", np.int64)])
        )
        self.elements_struc_array_hpc = np.zeros(1, dtype=np.dtype([("_id", np.int64)]))

    monkeypatch.setattr(DFN, "consolidate_dfn", fake_consolidate)
    monkeypatch.setattr(DFN, "print_solver_constants", lambda self: None)
    monkeypatch.setattr(
        DFN, "unconsolidate_dfn", lambda self, hpc=False: calls.append(("un", hpc))
    )

    def fake_hpc_solve(fracs, elems, discharge_int, constants, ntype):
        calls.append(("solve", discharge_int, len(elems)))
        out = np.zeros(1, dtype=np.dtype([("x", np.float64)]))
        out[0]["x"] = 1.0
        return out

    monkeypatch.setattr(dfn_mod, "hpc_solve", fake_hpc_solve)

    dfn.solve(unconsolidate=True)

    assert calls[0][0] == "solve"
    assert calls[1] == ("un", True)
    assert dfn.elements_struc_array[0]["x"] == pytest.approx(1.0)


def test_check_boundary_conditions_paths(monkeypatch):
    dfn = DFN("dfnI")

    assert dfn.check_boundary_conditions() is None

    dfn.elements_struc_array = np.zeros(2, dtype=np.dtype([("dummy", np.int64)]))
    dfn.fractures_struc_array_hpc = np.zeros(1, dtype=np.dtype([("dummy", np.int64)]))

    bnd = np.zeros((2, 6), dtype=float)
    bnd[0, 0] = 0.1
    bnd[0, 1] = 1
    bnd[1, 0] = 0.3
    bnd[1, 1] = 2
    monkeypatch.setattr(dfn_mod, "hpc_compute_bnd_error", lambda *args: bnd)

    out, idx = dfn.check_boundary_conditions(n_points=8)
    assert np.array_equal(out, bnd)
    assert idx == 1


def test_center_size_and_smallest_radius_properties():
    dfn = DFN("dfnJ")
    f0 = _make_frac("f0", [0, 0, 0], radius=2.0)
    f1 = _make_frac("f1", [4, 0, 0], radius=1.0)
    dfn.add_fracture([f0, f1])

    assert np.allclose(dfn.center, np.array([2.0, 0.0, 0.0]))
    assert np.allclose(dfn.size, np.array([7.0, 4.0, 4.0]))
    assert dfn.smallest_fracture_radius() == pytest.approx(1.0)
