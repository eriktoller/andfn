import numpy as np
import pytest

from andfn.storage import (
    Connectivity,
    FixedOrderStorage,
    _layout,
    element_metadata_dtype,
    growth_capacities,
)


def storage_for(lengths, capacities):
    lengths = np.asarray(lengths, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    starts, size = _layout(capacities)
    elements = np.zeros(lengths.size, dtype=element_metadata_dtype)
    elements["_id"] = np.arange(lengths.size)
    elements["_type"] = 1
    elements["ncoef"] = lengths
    elements["nint"] = 2
    return FixedOrderStorage(
        elements,
        np.zeros(size, dtype=np.complex128),
        np.zeros(size, dtype=np.complex128),
        np.zeros(size, dtype=np.complex128),
        starts,
        capacities,
    )


def test_coefficient_capacity_tracks_logical_lengths_without_global_padding():
    storage = storage_for([3, 0, 9], [4, 0, 12])
    storage.validate()
    assert storage.coef_data.size == 16
    assert storage.coef_data.size == storage.coef_capacity.sum()
    assert np.all(storage.elements["ncoef"] <= storage.coef_capacity)


def test_growth_policy_retains_capacity_and_grows_only_required_segments():
    np.testing.assert_array_equal(growth_capacities([0, 10, 7]), [0, 12, 8])
    np.testing.assert_array_equal(
        growth_capacities([11, 15, 101], [12, 12, 100]), [12, 16, 128]
    )
    with pytest.raises(ValueError):
        growth_capacities([1], quantum=0)
    with pytest.raises(ValueError):
        growth_capacities([1], growth_factor=0.5)


def test_storage_validation_rejects_invalid_lengths_and_segment_layouts():
    storage = storage_for([3, 2], [4, 4])
    storage.elements["ncoef"][0] = 5
    with pytest.raises(ValueError, match="capacity"):
        storage.validate()

    storage = storage_for([3, 2], [4, 4])
    storage.coef_start[1] = 0
    with pytest.raises(ValueError, match="partition"):
        storage.validate()


def test_csr_connectivity_payload_depends_on_actual_adjacency_count():
    connectivity = Connectivity(
        np.array([3, 1, 0], dtype=np.int32),
        np.array([0, 2, 3], dtype=np.int64),
    )
    connectivity.validate(2, 4)
    np.testing.assert_array_equal(connectivity.element_ids(0), [3, 1])
    np.testing.assert_array_equal(connectivity.element_ids(1), [0])
    assert connectivity.nbytes == 3 * 4 + 3 * 8
    with pytest.raises(IndexError):
        connectivity.element_ids(2)


def test_csr_validation_rejects_invalid_offsets_and_ids():
    for connectivity in (
        Connectivity(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 2, 1], dtype=np.int64),
        ),
        Connectivity(
            np.array([0, 4], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ),
    ):
        with pytest.raises(ValueError):
            connectivity.validate(2, 3)
