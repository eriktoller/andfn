import numpy as np
import pytest
from legacy_hpc_reference import (
    connectivity_from_hpc,
    dtype_work,
    dtype_z_arrays,
    element_dtype_hpc,
    fracture_dtype_hpc,
    fracture_metadata_from_hpc,
)
from scipy.sparse import csc_matrix

from andfn.geometry_functions import count_fracture_adjacency_from_fractures
from andfn.hpc import hpc_error_solve as error_solver
from andfn.hpc import hpc_fracture as hf
from andfn.hpc import hpc_math_functions as mf
from andfn.hpc import hpc_solve as hs


def _network():
    fractures = np.zeros(4, dtype=fracture_dtype_hpc)
    fractures["_id"] = np.arange(4)
    fractures["t"] = [2, 4, 3, 5]
    fractures["radius"] = 4
    fractures["constant"] = [10, 20, 30, 40]
    fractures["x_vector"][:, 0] = 1
    fractures["y_vector"][:, 1] = 1
    elements = np.zeros(9, dtype=element_dtype_hpc)
    elements["_id"] = np.arange(9)
    elements["_type"] = [1, 0, 2, 3, 4, 5, 1, 2, 1]
    elements["frac0"] = [0, 0, 0, 0, 0, 0, 1, 1, 3]
    elements["frac1"] = -1
    elements["frac1"][1] = 1
    elements["radius"] = [4, 0, 0.1, 0, 0.2, 0, 4, 0.1, 4]
    elements["center"] = 0.2 + 0.1j
    elements["endpoints0"] = [-0.8 - 0.3j, 0.9 - 0.3j]
    elements["endpoints1"] = [-0.6 + 0.2j, 0.7 + 0.2j]
    elements["ncoef"] = 3
    elements["nint"] = 6
    elements["phi"] = 15
    elements["q"] = np.arange(9) / 10
    elements["coef"][:, :3] = [0, 0.2 + 0.1j, -0.02 + 0.03j]
    for i, ids in enumerate(([5, 0, 3, 1, 4, 2], [7, 1, 6], [], [8])):
        fractures["nelements"][i] = len(ids)
        fractures["elements"][i, : len(ids)] = ids
    return fractures, elements


def _reference_matrix(fractures, elements, discharge, n, z):
    """Independent Python assembly of the pre-migration row conventions."""
    rows, cols, values = [], [], []
    positions = {int(e["_id"]): i for i, e in enumerate(discharge)}
    nd = discharge.size
    for row, e in enumerate(discharge):
        sides = [(int(e["frac0"]), 1.0, "z0")]
        if e["_type"] == 0:
            sides.append((int(e["frac1"]), -1.0, "z1"))
        for fid, sign, zname in sides:
            f = fractures[fid]
            for eid in f["elements"][: f["nelements"]]:
                other = elements[eid]
                if other["_type"] not in (0, 2, 3):
                    continue
                if other["_id"] == e["_id"] + 99 * 99 * 99:
                    continue
                value = hs.get_discharge_term(
                    other, z[zname][row, :n], fid, f["radius"], e["_id"]
                )
                if e["_type"] == 0:
                    value = sign * value / f["t"]
                rows.append(row)
                cols.append(positions[int(eid)])
                values.append(value)
            rows.append(row)
            cols.append(nd + fid)
            values.append(sign / f["t"] if e["_type"] == 0 else 1.0)
    for fid, f in enumerate(fractures):
        for eid in f["elements"][: f["nelements"]]:
            e = elements[eid]
            if e["_type"] in (0, 2, 3):
                rows.append(nd + fid)
                cols.append(positions[int(eid)])
                values.append(
                    1.0 if e["_type"] != 0 or e["frac0"] == f["_id"] else -1.0
                )
    return csc_matrix((values, (rows, cols)), shape=(nd + fractures.size,) * 2)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_csr_order_degrees_empty_single_shared_and_payload(index_dtype):
    fractures, elements = _network()
    connectivity = connectivity_from_hpc(
        fractures, elements.size, index_dtype=index_dtype
    )
    assert connectivity.ids.dtype == index_dtype
    assert connectivity.indptr.dtype == np.int64
    np.testing.assert_array_equal(connectivity.indptr, [0, 6, 9, 9, 10])
    np.testing.assert_array_equal(connectivity.ids, [5, 0, 3, 1, 4, 2, 7, 1, 6, 8])
    for i in range(fractures.size):
        np.testing.assert_array_equal(
            connectivity.element_ids(i),
            fractures["elements"][i, : fractures["nelements"][i]],
        )
    assert connectivity.nbytes == 10 * np.dtype(index_dtype).itemsize + 5 * 8
    assert connectivity.nbytes < fractures["elements"].nbytes
    # Appending empty fractures grows offsets, not padded IDs.
    empty = np.zeros(100, dtype=fracture_dtype_hpc)
    assert connectivity_from_hpc(empty, 0).nbytes == 101 * 8


@pytest.mark.parametrize("invalid", [-1, 9, 2**40])
def test_invalid_ids_rejected_before_narrowing(invalid):
    fractures, elements = _network()
    fractures["elements"][0, 0] = invalid
    with pytest.raises(ValueError):
        connectivity_from_hpc(fractures, elements.size, index_dtype=np.int32)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_potential_derivative_and_heads_match_legacy(index_dtype):
    fractures, elements = _network()
    connectivity = connectivity_from_hpc(
        fractures, elements.size, index_dtype=index_dtype
    )
    metadata = fracture_metadata_from_hpc(fractures)
    z = np.array([0.3 + 0.7j, -0.2 + 0.6j])
    for i in range(fractures.size):
        for point in z:
            for exclude in (-1, 1, 3):
                actual = hf.calc_omega_csr(
                    i,
                    point,
                    metadata,
                    elements,
                    connectivity.ids,
                    connectivity.indptr,
                    exclude,
                )
                expected = hf.calc_omega(fractures[i], point, elements, exclude)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
                np.testing.assert_allclose(
                    hf.calc_w_csr(
                        i,
                        point,
                        metadata,
                        elements,
                        connectivity.ids,
                        connectivity.indptr,
                        exclude,
                    ),
                    hf.calc_w(fractures[i], point, elements, exclude),
                    rtol=1e-13,
                    atol=1e-12,
                )
        actual_array = np.empty(z.size, dtype=np.complex128)
        expected_array = np.empty_like(actual_array)
        hf.calc_omega_array(
            metadata[i], actual_array, z, elements, -1, connectivity.element_ids(i)
        )
        hf.calc_omega_array(fractures[i], expected_array, z, elements)
        np.testing.assert_allclose(actual_array, expected_array, rtol=1e-13, atol=1e-12)
    expected_heads, expected_points = hf.get_heads(fractures, elements, z)
    actual_heads, actual_points = hf.get_heads_csr(
        metadata, elements, connectivity.ids, connectivity.indptr, z
    )
    np.testing.assert_allclose(actual_heads, expected_heads, rtol=1e-13, atol=1e-12)
    np.testing.assert_array_equal(actual_points, expected_points)
    selection = np.array([3, 0, 1, 0], dtype=np.int64)
    selected_heads, selected_points = hf.get_heads_csr(
        metadata, elements, connectivity.ids, connectivity.indptr, z, selection
    )
    np.testing.assert_array_equal(selected_heads, actual_heads[selection])
    np.testing.assert_array_equal(selected_points, actual_points[selection])
    assert hf.calc_omega_csr.nopython_signatures
    assert hf.calc_w_csr.nopython_signatures
    assert hf.get_heads_csr.nopython_signatures


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_discharge_order_matrix_structure_and_values(index_dtype):
    fractures, elements = _network()
    connectivity = connectivity_from_hpc(
        fractures, elements.size, index_dtype=index_dtype
    )
    metadata = fracture_metadata_from_hpc(fractures)
    indices = hs.get_discharge_indices(elements)
    legacy = hs.get_discharge_elements(elements)
    np.testing.assert_array_equal(indices, [1, 2, 3, 7])
    np.testing.assert_array_equal(indices, legacy["_id"])
    assert indices.dtype == np.int64
    assert indices.nbytes == 4 * 8
    assert legacy.nbytes == 4 * element_dtype_hpc.itemsize
    n = 8
    z_legacy = np.zeros(indices.size, dtype=dtype_z_arrays)
    z = np.zeros_like(z_legacy)
    hs.get_z_int_array(z_legacy, legacy, n)
    hs.get_z_int_array(z, elements, n, indices)
    np.testing.assert_array_equal(z, z_legacy)
    expected_counts = hs.count_discharge_nnz(fractures, elements, legacy)
    counts = hs.count_discharge_nnz(
        metadata, elements, elements, connectivity.ids, connectivity.indptr, indices
    )
    np.testing.assert_array_equal(counts, expected_counts)
    # The unchanged error-solver entry uses the legacy count/fill interfaces.
    expected = error_solver.build_discharge_matrix(
        fractures, elements, legacy, n, z_legacy
    )
    reference = _reference_matrix(fractures, elements, legacy, n, z_legacy)
    actual = hs.build_discharge_matrix_csr(
        metadata, elements, connectivity.ids, connectivity.indptr, indices, n, z
    )
    assert expected.shape == actual.shape == (8, 8)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-13, atol=1e-12)
    np.testing.assert_array_equal(actual.indptr, reference.indptr)
    np.testing.assert_array_equal(actual.indices, reference.indices)
    np.testing.assert_allclose(actual.data, reference.data, rtol=1e-13, atol=1e-12)
    expected_head = np.zeros(8)
    actual_head = np.zeros(8)
    hs.build_head_matrix(fractures, elements, legacy, n, expected_head, z_legacy)
    hs.build_head_matrix(
        metadata,
        elements,
        elements,
        n,
        actual_head,
        z,
        connectivity.ids,
        connectivity.indptr,
        indices,
    )
    np.testing.assert_allclose(actual_head, expected_head, rtol=1e-13, atol=1e-12)
    assert hs.get_discharge_indices.nopython_signatures
    assert hs.count_discharge_nnz.nopython_signatures
    assert hs.fill_discharge_matrix.nopython_signatures
    assert hs.build_head_matrix.nopython_signatures


def test_indexed_discharge_updates_original_records():
    fractures, elements = _network()
    indices = hs.get_discharge_indices(elements)
    legacy = hs.get_discharge_elements(elements)
    np.testing.assert_array_equal(
        hs.get_old_discharges(elements, fractures, legacy),
        hs.get_old_discharges(elements, fractures, elements, indices),
    )
    values = np.arange(indices.size + fractures.size, dtype=float)
    before = elements["q"].copy()
    hs.post_matrix_solve(fractures, elements, elements, values, values, indices)
    np.testing.assert_array_equal(elements["q"][indices], values[: indices.size])
    np.testing.assert_array_equal(
        elements["q"][[0, 4, 5, 6, 8]], before[[0, 4, 5, 6, 8]]
    )
    np.testing.assert_array_equal(fractures["constant"], values[indices.size :])


def test_empty_matrix_and_invalid_order():
    fractures = np.zeros(0, dtype=fracture_dtype_hpc)
    elements = np.zeros(0, dtype=element_dtype_hpc)
    connectivity = connectivity_from_hpc(fractures, 0)
    indices = hs.get_discharge_indices(elements)
    matrix = hs.build_discharge_matrix_csr(
        fracture_metadata_from_hpc(fractures),
        elements,
        connectivity.ids,
        connectivity.indptr,
        indices,
        1,
        np.zeros(0, dtype=dtype_z_arrays),
    )
    assert matrix.shape == (0, 0)
    assert matrix.nnz == 0
    fractures, elements = _network()
    connectivity = connectivity_from_hpc(fractures, elements.size)
    with pytest.raises(ValueError, match="order"):
        hs.build_discharge_matrix_csr(
            fractures,
            elements,
            connectivity.ids,
            connectivity.indptr,
            hs.get_discharge_indices(elements)[::-1].copy(),
            1,
            np.zeros(elements.size, dtype=dtype_z_arrays),
        )


def test_branch_events_preserved_without_neighbor_record_gather():
    fractures, elements = _network()
    connectivity = connectivity_from_hpc(fractures, elements.size, index_dtype=np.int32)
    metadata = fracture_metadata_from_hpc(fractures)
    # Event arrays are deliberately independent of connectivity degree.
    work = np.zeros(2, dtype=dtype_work)
    points = 4 * np.exp(1j * np.arange(16) * 2 * np.pi / 16)
    mf.find_branch_cuts(elements[0], points, fractures, elements, work[0], 16)
    mf.find_branch_cuts(
        elements[0],
        points,
        metadata,
        elements,
        work[1],
        16,
        connectivity.ids,
        connectivity.indptr,
    )
    count = work[0]["len_discharge_element"]
    assert count > 0
    assert work[1]["len_discharge_element"] == count
    for field in ("sign_array", "discharge_element", "element_pos"):
        np.testing.assert_array_equal(work[0][field][:count], work[1][field][:count])
    np.testing.assert_array_equal(
        count_fracture_adjacency_from_fractures(fractures, elements), [1, 1, 0, 0]
    )
