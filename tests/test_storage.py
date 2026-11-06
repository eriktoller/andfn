import numba as nb
import numpy as np
import pytest

from andfn.element import element_dtype_hpc, fracture_dtype_hpc
from andfn.hpc import hpc_bounding_circle, hpc_math_functions
from andfn.storage import Connectivity, from_hpc, growth_capacities, to_hpc


def _legacy():
    elements = np.zeros(3, dtype=element_dtype_hpc)
    fractures = np.zeros(3, dtype=fracture_dtype_hpc)
    elements["_id"] = np.arange(3)
    elements["_type"] = 1
    elements["frac0"] = [0, 1, 1]
    elements["frac1"] = -1
    elements["radius"] = 2
    elements["ncoef"] = [3, 7, 0]
    elements["nint"] = [11, 5, 0]
    fractures["_id"] = np.arange(3)
    fractures["t"] = [1, 2, 3]
    fractures["nelements"] = [1, 3, 0]
    fractures["elements"][0, 0] = 0
    fractures["elements"][1, :3] = [2, 1, 0]
    for i in range(3):
        nc, ni = elements["ncoef"][i], elements["nint"][i]
        elements["coef"][i, :nc] = np.arange(nc) + 1j * (i + 1)
        elements["old_coef"][i, :nc] = -elements["coef"][i, :nc]
        elements["thetas"][i, :ni] = np.arange(ni) / 10
        elements["dpsi_corr"][i, :ni] = np.arange(ni) / 100
    return elements, fractures


def test_roundtrip_preserves_all_metadata_and_logical_values():
    elements, fractures = _legacy()
    storage = from_hpc(elements, fractures, index_dtype=np.int32)
    storage.validate()
    actual_e, actual_f = to_hpc(storage)
    for name in elements.dtype.names:
        np.testing.assert_array_equal(actual_e[name], elements[name])
    for name in fractures.dtype.names:
        np.testing.assert_array_equal(actual_f[name], fractures[name])
    np.testing.assert_array_equal(storage.coef_capacity, [4, 8, 0])
    np.testing.assert_array_equal(storage.integration_capacity, [12, 8, 0])
    np.testing.assert_array_equal(storage.connectivity.indptr, [0, 1, 4, 4])
    np.testing.assert_array_equal(storage.connectivity.ids, [0, 2, 1, 0])
    assert storage.connectivity.ids.dtype == np.int32
    assert np.all(storage.candidate_coef_data == 0)
    assert not np.shares_memory(storage.coef_data, elements)
    assert not np.shares_memory(actual_e, storage.elements)


def test_unused_padding_is_not_persisted():
    elements, fractures = _legacy()
    elements["coef"][0, 3:] = 123
    elements["old_coef"][0, 3:] = 456
    elements["thetas"][0, 11:] = 789
    elements["dpsi_corr"][0, 11:] = 987
    fractures["elements"][0, 1:] = -100
    actual_e, actual_f = to_hpc(from_hpc(elements, fractures))
    assert np.all(actual_e["coef"][0, 3:] == 0)
    assert np.all(actual_e["old_coef"][0, 3:] == 0)
    assert np.all(actual_e["thetas"][0, 11:] == 0)
    assert np.all(actual_e["dpsi_corr"][0, 11:] == 0)
    assert np.all(actual_f["elements"][0, 1:] == 0)


def test_coefficient_views_mutate_only_the_target_segment():
    storage = from_hpc(*_legacy())
    before = storage.coefficients(1).copy()
    storage.coefficients(0)[1] = 99 + 12j
    assert storage.coef_data[1] == 99 + 12j
    np.testing.assert_array_equal(storage.coefficients(1), before)
    assert storage.coefficients(2).size == 0
    for invalid in (-1, 3):
        with pytest.raises(IndexError):
            storage.coefficients(invalid)


def test_logical_diagnostic_accessors_return_writable_views():
    elements, fractures = _legacy()
    storage = from_hpc(elements, fractures)
    for accessor, field, buffer in (
        (storage.old_coefficients, "old_coef", storage.old_coef_data),
        (storage.candidate_coefficients, None, storage.candidate_coef_data),
        (storage.theta_values, "thetas", storage.theta_data),
        (storage.dpsi_corrections, "dpsi_corr", storage.dpsi_corr_data),
    ):
        view = accessor(0)
        expected_length = 3 if "coefficients" in accessor.__name__ else 11
        assert view.size == expected_length
        assert np.shares_memory(view, buffer)
        if field:
            np.testing.assert_array_equal(view, elements[field][0, :expected_length])
        view[0] = 17
        assert buffer[0] == 17
        assert accessor(2).size == 0
        with pytest.raises(IndexError):
            accessor(-1)
    view = storage.fracture_element_ids(1)
    np.testing.assert_array_equal(view, [2, 1, 0])
    assert np.shares_memory(view, storage.connectivity.ids)
    assert storage.fracture_element_ids(2).size == 0
    for i in (-1, 3):
        with pytest.raises(IndexError):
            storage.fracture_element_ids(i)


def test_views_become_stale_after_repacking():
    storage = from_hpc(*_legacy())
    old_view = storage.coefficients(0)
    prefix = old_view.copy()
    storage.expand([9, 7, 0], [11, 5, 0])
    assert not np.shares_memory(old_view, storage.coef_data)
    new_view = storage.coefficients(0)
    assert np.shares_memory(new_view, storage.coef_data)
    np.testing.assert_array_equal(new_view[: prefix.size], prefix)
    old_view[:] = -999
    np.testing.assert_array_equal(new_view[: prefix.size], prefix)


@pytest.mark.parametrize(
    "nc,ni,kwargs",
    [
        ([2, 7, 0], [11, 5, 0], {}),
        ([9, 7, 0], [10, 5, 0], {}),
        ([9, 7, 0], [11, 5], {}),
        ([9, 7, 0], [11, 5, 0], {"growth_factor": 0.5}),
        ([9, 7, 0], [np.iinfo(np.int64).max, 5, 0], {}),
    ],
)
def test_invalid_expansion_is_transactional(nc, ni, kwargs):
    storage = from_hpc(*_legacy())
    storage.candidate_coef_data[:] = 7 + 4j
    buffers = {
        name: value
        for name, value in vars(storage).items()
        if isinstance(value, np.ndarray)
    }
    buffers["connectivity.ids"] = storage.connectivity.ids
    buffers["connectivity.indptr"] = storage.connectivity.indptr
    snapshots = {name: value.copy() for name, value in buffers.items()}
    connectivity = storage.connectivity
    with pytest.raises(ValueError):
        storage.expand(nc, ni, **kwargs)
    assert storage.connectivity is connectivity
    for name, original in buffers.items():
        if name.startswith("connectivity."):
            actual = getattr(storage.connectivity, name.split(".")[1])
        else:
            actual = getattr(storage, name)
        assert actual is original
        np.testing.assert_array_equal(actual, snapshots[name])
    storage.validate()


def test_expansion_within_capacity_reuses_buffers_and_clears_exposed_slack():
    storage = from_hpc(*_legacy())
    data = storage.coef_data
    theta = storage.theta_data
    storage.coef_data[3] = 999
    storage.candidate_coef_data[3] = 999
    storage.theta_data[11] = 999
    prefix = storage.coefficients(0).copy()
    storage.expand([4, 7, 0], [12, 5, 0])
    assert storage.coef_data is data
    assert storage.theta_data is theta
    np.testing.assert_array_equal(storage.coefficients(0)[:3], prefix)
    assert storage.coefficients(0)[3] == 0
    assert storage.candidate_coef_data[3] == 0
    assert storage.theta_data[11] == 0
    storage.validate()


def test_batch_repacking_preserves_every_buffer_and_zeroes_tails():
    elements, fractures = _legacy()
    storage = from_hpc(elements, fractures)
    storage.candidate_coef_data[:] = storage.coef_data * 2
    old_data = storage.coef_data
    storage.expand([9, 12, 2], [13, 9, 1])
    assert storage.coef_data is not old_data
    storage.validate()
    for i in range(3):
        nc, ni = elements["ncoef"][i], elements["nint"][i]
        c, s = storage.coef_start[i], storage.integration_start[i]
        np.testing.assert_array_equal(
            storage.coefficients(i)[:nc], elements["coef"][i, :nc]
        )
        np.testing.assert_array_equal(
            storage.old_coef_data[c : c + nc], elements["old_coef"][i, :nc]
        )
        np.testing.assert_array_equal(
            storage.candidate_coef_data[c : c + nc], 2 * elements["coef"][i, :nc]
        )
        assert np.all(storage.coef_data[c + nc : c + storage.coef_capacity[i]] == 0)
        np.testing.assert_array_equal(
            storage.theta_data[s : s + ni], elements["thetas"][i, :ni]
        )
        np.testing.assert_array_equal(
            storage.dpsi_corr_data[s : s + ni], elements["dpsi_corr"][i, :ni]
        )
        assert np.all(
            storage.theta_data[s + ni : s + storage.integration_capacity[i]] == 0
        )


def test_coefficient_and_integration_families_repack_independently():
    storage = from_hpc(*_legacy())
    coefficients = storage.coef_data
    theta = storage.theta_data
    storage.expand([3, 7, 0], [20, 5, 0])
    assert storage.coef_data is coefficients
    assert storage.theta_data is not theta
    theta = storage.theta_data
    storage.expand([9, 7, 0], [20, 5, 0])
    assert storage.coef_data is not coefficients
    assert storage.theta_data is theta
    storage.validate()


def test_noop_expansion_does_not_replace_data_buffers():
    storage = from_hpc(*_legacy())
    coefficients = storage.coef_data
    theta = storage.theta_data
    storage.expand([3, 7, 0], [11, 5, 0])
    assert storage.coef_data is coefficients
    assert storage.theta_data is theta


def test_growth_policy_and_invalid_requests_are_explicit():
    np.testing.assert_array_equal(growth_capacities([0, 10, 7]), [0, 12, 8])
    np.testing.assert_array_equal(
        growth_capacities([11, 15, 101], [12, 12, 100]), [12, 16, 128]
    )
    for kwargs in ({"quantum": 0}, {"growth_factor": 0.5}, {"growth_factor": np.inf}):
        with pytest.raises(ValueError):
            growth_capacities([1], **kwargs)
    for values in ([-1], [1.5], [np.iinfo(np.uint64).max]):
        with pytest.raises(ValueError):
            growth_capacities(values)
    with pytest.raises(ValueError, match="int64"):
        growth_capacities(np.array([np.iinfo(np.int64).max]))
    storage = from_hpc(*_legacy())
    data = storage.coef_data
    for nc, ni in (([2, 7, 0], [11, 5, 0]), ([3, 7], [11, 5, 0])):
        with pytest.raises(ValueError):
            storage.expand(nc, ni)
    assert storage.coef_data is data
    np.testing.assert_array_equal(storage.elements["ncoef"], [3, 7, 0])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.coef_start.__setitem__(1, 0),
        lambda s: s.coef_capacity.__setitem__(0, 2),
        lambda s: s.integration_start.__setitem__(0, 1),
        lambda s: s.elements["nint"].__setitem__(0, -1),
        lambda s: s.elements["_id"].__setitem__(1, 99),
        lambda s: s.elements["frac0"].__setitem__(0, 99),
        lambda s: s.fractures["_id"].__setitem__(0, 99),
        lambda s: s.connectivity.ids.__setitem__(0, 99),
        lambda s: s.connectivity.indptr.__setitem__(1, 5),
        lambda s: setattr(s, "candidate_coef_data", s.coef_data),
        lambda s: setattr(s, "theta_data", s.theta_data[::2]),
        lambda s: setattr(s, "coef_start", s.coef_start.astype(np.int32)),
    ],
)
def test_validation_rejects_corrupted_storage(mutate):
    storage = from_hpc(*_legacy())
    mutate(storage)
    with pytest.raises(ValueError):
        storage.validate()


@pytest.mark.parametrize(
    "field,value",
    [("ncoef", -1), ("ncoef", 100000), ("nint", 100000), ("frac0", 99)],
)
def test_conversion_rejects_invalid_legacy_elements(field, value):
    elements, fractures = _legacy()
    elements[field][0] = value
    with pytest.raises(ValueError):
        from_hpc(elements, fractures)


def test_conversion_rejects_invalid_connectivity_and_index_types():
    elements, fractures = _legacy()
    fractures["nelements"][0] = 100000
    with pytest.raises(ValueError):
        from_hpc(elements, fractures)
    elements, fractures = _legacy()
    fractures["elements"][0, 0] = -1
    with pytest.raises(ValueError):
        from_hpc(elements, fractures, index_dtype=np.int32)
    with pytest.raises(ValueError):
        from_hpc(elements, fractures, index_dtype=np.uint32)
    with pytest.raises(ValueError):
        Connectivity(np.array([0], dtype=float), np.array([0, 1])).validate(1, 1)


def test_empty_dfn_and_legacy_capacity_limits():
    storage = from_hpc(
        np.zeros(0, dtype=element_dtype_hpc),
        np.zeros(0, dtype=fracture_dtype_hpc),
    )
    storage.expand(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    e, f = to_hpc(storage)
    assert e.size == f.size == 0
    storage = from_hpc(*_legacy())
    storage.expand([element_dtype_hpc["coef"].shape[0] + 1, 7, 0], [11, 5, 0])
    with pytest.raises(ValueError, match="legacy coefficient"):
        to_hpc(storage)
    storage = from_hpc(*_legacy())
    storage.expand([3, 7, 0], [element_dtype_hpc["thetas"].shape[0] + 1, 5, 0])
    with pytest.raises(ValueError, match="legacy integration"):
        to_hpc(storage)


def test_flat_arrays_compile_and_roundtrip_preserves_kernel_results():
    elements, fractures = _legacy()
    storage = from_hpc(elements, fractures)
    compiled_series = nb.njit(
        lambda data, start, count, chi: hpc_math_functions.taylor_series(
            chi, data[start : start + count]
        )
    )
    actual = compiled_series(storage.coef_data, storage.coef_start[0], 3, 0.5j)
    expected = hpc_math_functions.taylor_series(0.5j, elements["coef"][0, :3])
    assert actual == expected
    assert compiled_series.nopython_signatures
    roundtrip, _ = to_hpc(storage)
    for z in (0.5j, 1 + 0.5j):
        assert hpc_bounding_circle.calc_omega(
            elements[0], z
        ) == hpc_bounding_circle.calc_omega(roundtrip[0], z)
        assert hpc_bounding_circle.calc_w(elements[0], z) == hpc_bounding_circle.calc_w(
            roundtrip[0], z
        )


def test_coefficient_payload_scales_with_actual_capacity():
    # A modest real allocation; the million-element legacy case is analytical.
    ne = 1000
    elements = np.zeros(ne, dtype=element_dtype_hpc)
    elements["_id"] = np.arange(ne)
    elements["frac0"] = elements["frac1"] = -1
    elements["ncoef"] = 10
    elements["nint"] = 20
    storage = from_hpc(elements, np.zeros(0, dtype=fracture_dtype_hpc))
    payload = sum(
        array.nbytes
        for array in (
            storage.coef_data,
            storage.old_coef_data,
            storage.candidate_coef_data,
        )
    )
    assert payload == 48 * int(storage.coef_capacity.sum())
    assert payload == 48 * 12 * ne
    assert storage.nbytes < elements.nbytes / 5
    assert 48 * 12 * 1_000_000 < 32 * 500 * 1_000_000
