import numpy as np
import pytest
from scipy.spatial import KDTree

from andfn import geometry_functions as gf
from andfn.const_head import ConstantHeadLine
from andfn.intersection import Intersection


class _FractureStub:
    def __init__(self, center, normal, x_vector, y_vector, radius=1.0):
        self.center = np.array(center, dtype=np.float64)
        self.normal = np.array(normal, dtype=np.float64)
        self.x_vector = np.array(x_vector, dtype=np.float64)
        self.y_vector = np.array(y_vector, dtype=np.float64)
        self.radius = float(radius)
        self.elements = []


class _PairFracture:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)


def test_map_z_line_to_chi_array_matches_scalar_calls():
    endpoints = np.array([-1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    z = np.array([1.5 + 0.2j, -2.0 + 0.3j, 3.0 - 0.1j], dtype=np.complex128)

    result = gf.map_z_line_to_chi(z, endpoints)
    expected = np.array([gf.map_z_line_to_chi(v, endpoints) for v in z])

    np.testing.assert_allclose(result, expected)


def test_map_chi_to_z_line_matches_its_closed_form():
    endpoints = np.array([-2.0 + 1.0j, 3.0 - 1.0j], dtype=np.complex128)
    chi = np.array([2.0 + 0.0j, 1.5 + 0.2j], dtype=np.complex128)

    result = gf.map_chi_to_z_line(chi, endpoints)
    big_z = 0.5 * (chi + 1.0 / chi)
    expected = 0.5 * (
        big_z * (endpoints[1] - endpoints[0]) + endpoints[0] + endpoints[1]
    )

    np.testing.assert_allclose(result, expected)


def test_circle_mappings_round_trip_scalar_and_array():
    center = 0.5 - 0.25j
    radius = 2.5

    z_scalar = 1.2 + 0.8j
    z_array = np.array([1.2 + 0.8j, -0.2 + 0.3j, 3.0 - 1.0j], dtype=np.complex128)

    back_scalar = gf.map_chi_to_z_circle(
        gf.map_z_circle_to_chi(z_scalar, radius, center), radius, center
    )
    back_array = gf.map_chi_to_z_circle(
        gf.map_z_circle_to_chi(z_array, radius, center), radius, center
    )

    np.testing.assert_allclose(back_scalar, z_scalar)
    np.testing.assert_allclose(back_array, z_array)


def test_map_2d_to_3d_and_back_round_trip():
    frac = _FractureStub(
        center=[10.0, -3.0, 2.0],
        normal=[0.0, 0.0, 1.0],
        x_vector=[1.0, 0.0, 0.0],
        y_vector=[0.0, 1.0, 0.0],
    )

    z_scalar = 2.5 - 1.5j
    point_scalar = gf.map_2d_to_3d(z_scalar, frac)
    back_scalar = gf.map_3d_to_2d(point_scalar, frac)

    z_array = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128)
    points_array = gf.map_2d_to_3d(z_array, frac)
    back_array = np.array([gf.map_3d_to_2d(p, frac) for p in points_array])

    np.testing.assert_allclose(back_scalar, z_scalar)
    np.testing.assert_allclose(back_array, z_array)


def test_line_circle_intersection_hits_and_misses():
    z0, z1 = gf.line_circle_intersection(-3.0 + 0.0j, 3.0 + 0.0j, radius=2.0)
    np.testing.assert_allclose(sorted([z0.real, z1.real]), [-2.0, 2.0])
    np.testing.assert_allclose([z0.imag, z1.imag], [0.0, 0.0])

    z2, z3 = gf.line_circle_intersection(-3.0 + 3.0j, 3.0 + 3.0j, radius=2.0)
    assert z2 is None
    assert z3 is None


def test_line_line_intersection_for_crossing_and_parallel_lines():
    z = gf.line_line_intersection(0.0 + 0.0j, 2.0 + 2.0j, 1.0 + 0.0j, 1.0 + 3.0j)
    np.testing.assert_allclose(z, 1.0 + 1.0j)

    z_parallel = gf.line_line_intersection(
        0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j
    )
    assert z_parallel is None


def test_line_disc_interval_hit_and_no_hit_cases():
    interval = gf.line_disc_interval(
        x0=np.array([0.0, 0.0, 0.0]),
        d=np.array([1.0, 0.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        radius=2.0,
    )
    assert interval == (-2.0, 2.0)

    no_hit = gf.line_disc_interval(
        x0=np.array([3.0, 0.0, 0.0]),
        d=np.array([0.0, 1.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        radius=2.0,
    )
    assert no_hit is None


def test_shorten_line_preserves_center_and_scales_length():
    endpoints = np.array([0.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex128)
    shortened = gf.shorten_line(endpoints, se_factor=0.25)

    np.testing.assert_allclose(np.mean(shortened), np.mean(endpoints))
    np.testing.assert_allclose(np.abs(shortened[1] - shortened[0]), 1.0)


def test_strike_dip_to_normal_returns_expected_for_horizontal_plane():
    normal = gf.convert_strike_dip_to_normal(140.0, 0.0)
    np.testing.assert_allclose(normal, np.array([0.0, 0.0, -1.0]), atol=1e-12)


def test_normal_to_strike_dip_returns_expected_for_vertical_normal():
    strike, dip = gf.convert_normal_to_strike_dip(np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(strike, 0.0, atol=1e-12)
    np.testing.assert_allclose(dip, 90.0, atol=1e-12)


def test_trend_plunge_to_normal_is_unit_length():
    normal = gf.convert_trend_plunge_to_normal(45.0, 20.0)
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0)


def test_fracture_intersection_orthogonal_discs_and_parallel_planes():
    frac0 = _FractureStub(
        center=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        x_vector=[1.0, 0.0, 0.0],
        y_vector=[0.0, 1.0, 0.0],
        radius=1.0,
    )
    frac1 = _FractureStub(
        center=[0.0, 0.0, 0.0],
        normal=[0.0, 1.0, 0.0],
        x_vector=[1.0, 0.0, 0.0],
        y_vector=[0.0, 0.0, 1.0],
        radius=1.0,
    )
    endpoints0, endpoints1 = gf.fracture_intersection(frac0, frac1)

    assert endpoints0 is not None and endpoints1 is not None
    np.testing.assert_allclose(np.sort(endpoints0.real), [-1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(np.sort(endpoints1.real), [-1.0, 1.0], atol=1e-10)

    frac2 = _FractureStub(
        center=[0.0, 0.0, 1.0],
        normal=[0.0, 0.0, 1.0],
        x_vector=[1.0, 0.0, 0.0],
        y_vector=[0.0, 1.0, 0.0],
        radius=1.0,
    )
    no0, no1 = gf.fracture_intersection(frac0, frac2)
    assert no0 is None and no1 is None


def test_get_fracture_intersections_raises_if_unsorted_when_tree_provided():
    fractures = [
        _PairFracture(center=[0.0, 0.0, 0.0], radius=1.0),
        _PairFracture(center=[0.5, 0.0, 0.0], radius=2.0),
    ]
    tree = KDTree(np.array([fr.center for fr in fractures]))

    with pytest.raises(ValueError, match="Fractures must be sorted by radius"):
        gf.get_fracture_intersections(fractures, se_factor=0.9, tree=tree)


def test_check_connectivity_returns_disconnected_fractures():
    f0 = _FractureStub(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    )
    f1 = _FractureStub(
        [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    )
    f2 = _FractureStub(
        [2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    )

    boundary = ConstantHeadLine.__new__(ConstantHeadLine)
    intersection = Intersection.__new__(Intersection)
    intersection.frac0 = f0
    intersection.frac1 = f1

    f0.elements = [boundary, intersection]
    f1.elements = [intersection]
    f2.elements = []

    all_connected, remove_list = gf.check_connectivity([f0, f1, f2])

    assert not all_connected
    assert remove_list == [f2]


def test_build_indptr_prefix_sum():
    counts = np.array([2, 0, 3, 1], dtype=np.int32)

    indptr, total = gf.build_indptr(counts)

    np.testing.assert_array_equal(
        indptr,
        np.array([0, 2, 2, 5, 6], dtype=np.int32),
    )
    assert total == 6
