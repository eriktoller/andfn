"""Runtime quadrature preserves selected-row coordinates and solver arithmetic."""

import numpy as np
import pytest
from legacy_hpc_reference import connectivity_from_hpc, dtype_z_arrays
from test_hpc_coefficients import network

from andfn.hpc import hpc_solve as hs
from andfn.hpc.hpc_quadrature import allocate_quadrature, validate_quadrature


@pytest.mark.parametrize("rows", [0, 1, 128])
@pytest.mark.parametrize("count", [1, 8, 1024])
def test_allocation_payload_contiguity_and_reuse(rows, count):
    values = allocate_quadrature(rows, count)
    assert values.shape == (2, rows, count)
    assert values.dtype == np.complex128
    assert values.flags.c_contiguous
    assert values.nbytes == rows * count * 16 * 2
    validate_quadrature(values, rows, count)
    _, elements = network()
    ids = np.resize(hs.get_discharge_indices(elements), rows)
    hs.get_z_int_array(values, elements, count, ids)
    expected = values.copy()
    hs.get_z_int_array(values, elements, count, ids)
    np.testing.assert_array_equal(values, expected)
    changed = allocate_quadrature(rows, count + 1)
    hs.get_z_int_array(changed, elements, count + 1, ids)
    assert changed.shape == (2, rows, count + 1)


@pytest.mark.parametrize(
    "rows,count",
    [
        (-1, 8),
        (1, 0),
        (1, -1),
        (1, 2.5),
        (True, 8),
        (1, False),
        (1, np.nan),
        (2**63, 1),
        (1, 2**63),
        (2**60, 16),
        (0, 2**63),
        (1, "8"),
    ],
)
def test_invalid_or_overflowing_dimensions_are_rejected(rows, count):
    with pytest.raises((ValueError, TypeError)):
        allocate_quadrature(rows, count)


@pytest.mark.parametrize("count", [1, 8, 32])
def test_coordinate_order_and_both_fracture_sides_exactly_match_legacy(count):
    _, elements = network()
    ids = np.array([3, 0, 2, 0], dtype=np.int64)
    dense = allocate_quadrature(ids.size, count)
    legacy = np.zeros(ids.size, dtype=dtype_z_arrays)
    hs.get_z_int_array(dense, elements, count, ids)
    hs.get_z_int_array(legacy, elements, count, ids)
    for side, name in enumerate(("z0", "z1")):
        np.testing.assert_array_equal(dense[side], legacy[name][:, :count])
        assert dense[side].flags.c_contiguous
    assert hs.get_z_int_array.nopython_signatures


def test_unused_maximum_does_not_change_dense_allocation(monkeypatch):
    from andfn import element

    allocations = []
    for maximum in (200, 500):
        monkeypatch.setattr(element, "MAX_NCOEF", maximum, raising=False)
        monkeypatch.setattr(hs, "MAX_NCOEF", maximum, raising=False)
        monkeypatch.setattr(
            hs,
            "dtype_z_arrays",
            dtype_z_arrays,
            raising=False,
        )
        values = allocate_quadrature(17, 8)
        allocations.append((values.shape, values.nbytes))
    assert allocations == [((2, 17, 8), 4352)] * 2


def test_matrix_rhs_and_mirror_terms_match_exactly():
    fractures, elements = network()
    # This fixture includes direct and mirrored discharge contributions.
    connectivity = connectivity_from_hpc(fractures, elements.size)
    ids = hs.get_discharge_indices(elements)
    count = 8
    dense = allocate_quadrature(ids.size, count)
    legacy = np.zeros(elements.size, dtype=dtype_z_arrays)
    for z in (dense, legacy):
        hs.get_z_int_array(z, elements, count, ids)
    matrices, heads = [], []
    for z in (dense, legacy):
        matrix = hs.build_discharge_matrix_csr(
            fractures, elements, connectivity.ids, connectivity.indptr, ids, count, z
        )
        matrices.append(matrix)
        output = np.zeros(ids.size + fractures.size)
        hs.build_head_matrix(
            fractures,
            elements,
            elements,
            count,
            output,
            z,
            connectivity.ids,
            connectivity.indptr,
            ids,
        )
        heads.append(output)
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(
            getattr(matrices[0], name), getattr(matrices[1], name)
        )
    np.testing.assert_array_equal(heads[0], heads[1])


def test_compact_solve_and_boundary_never_allocate_padded_quadrature(monkeypatch):
    from test_hpc_adaptive import run_adaptive

    zeros = np.zeros
    allocator = hs.allocate_quadrature
    producer = hs.get_z_int_array
    allocations, populations = [], []

    def checked(*args, **kwargs):
        dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
        if dtype == dtype_z_arrays:
            raise AssertionError("compact main solver allocated padded quadrature")
        return zeros(*args, **kwargs)

    def allocated(rows, count):
        allocations.append((rows, count))
        return allocator(rows, count)

    def populated(values, *args):
        populations.append(values.shape)
        return producer(values, *args)

    monkeypatch.setattr(hs.np, "zeros", checked)
    monkeypatch.setattr(hs, "allocate_quadrature", allocated)
    monkeypatch.setattr(hs, "get_z_int_array", populated)
    history, owner = run_adaptive()
    elements = owner.state.storage.elements
    selected = hs.get_discharge_indices(elements).size
    assert allocations == [(selected, 8)] + [(elements.size, 8)] * len(history)
    assert populations == [(2, rows, count) for rows, count in allocations]


def test_dense_validation_rejects_strides_and_wrong_shape():
    values = allocate_quadrature(3, 8)
    for bad in (values[:, :, ::2], values.astype(np.complex64), values[:, :2]):
        with pytest.raises(ValueError):
            validate_quadrature(bad, 3, 8)


@pytest.mark.parametrize("count", [0, -1, 8.0, True, np.nan, 2**63])
def test_dense_validation_rejects_invalid_requested_count(count):
    with pytest.raises((ValueError, TypeError)):
        validate_quadrature(allocate_quadrature(3, 8), 3, count)


def test_nonfinite_geometry_preserves_legacy_propagation():
    _, elements = network()
    ids = hs.get_discharge_indices(elements)
    elements["endpoints0"][ids[0], 0] = complex(np.nan, 0)
    dense = allocate_quadrature(ids.size, 8)
    padded = np.zeros(elements.size, dtype=dtype_z_arrays)
    for values in (dense, padded):
        hs.get_z_int_array(values, elements, 8, ids)
    for side, name in enumerate(("z0", "z1")):
        np.testing.assert_array_equal(dense[side], padded[name][: ids.size, :8])
    assert not np.isfinite(dense[0, 0]).all()
