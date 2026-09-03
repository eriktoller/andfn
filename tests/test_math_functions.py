import numpy as np
import pytest
from numba.core.errors import TypingError

from andfn import math_functions as mf


def _manual_asym_expansion(chi, coef):
    return sum(c * chi ** (-n) for n, c in enumerate(coef))


def _manual_asym_expansion_d1(chi, coef):
    return -sum(n * c * chi ** (-n - 1) for n, c in enumerate(coef))


def _manual_taylor_series(chi, coef):
    return sum(c * chi**n for n, c in enumerate(coef))


def _manual_taylor_series_d1(chi, coef):
    return sum(n * c * chi ** (n - 1) for n, c in enumerate(coef) if n > 0)


def test_asym_expansion_scalar_matches_series_definition():
    chi = 1.5 + 0.5j
    coef = np.array([1.0 + 2.0j, -0.4 + 0.2j, 0.3 - 0.1j], dtype=np.complex128)

    result = mf.asym_expansion(chi, coef)
    expected = _manual_asym_expansion(chi, coef)

    assert np.isscalar(result)
    np.testing.assert_allclose(result, expected)


def test_asym_expansion_array_matches_elementwise_evaluation():
    chi = np.array([1.0 + 0.5j, 2.0 - 1.0j, -1.5 + 0.2j], dtype=np.complex128)
    coef = np.array([1.0 + 2.0j, -0.4 + 0.2j, 0.3 - 0.1j], dtype=np.complex128)

    result = mf.asym_expansion(chi, coef)
    expected = np.array(
        [_manual_asym_expansion(c, coef) for c in chi], dtype=np.complex128
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == chi.shape
    np.testing.assert_allclose(result, expected)


def test_asym_expansion_d1_scalar_matches_series_derivative():
    chi = 1.2 - 0.8j
    coef = np.array([1.0 + 0.5j, 0.3 - 0.1j, -0.2 + 0.4j], dtype=np.complex128)

    result = mf.asym_expansion_d1(chi, coef)
    expected = _manual_asym_expansion_d1(chi, coef)

    np.testing.assert_allclose(result, expected)


def test_asym_expansion_d1_array_input_raises_typing_error():
    chi = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    coef = np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)

    with pytest.raises(TypingError):
        mf.asym_expansion_d1(chi, coef)


def test_taylor_series_scalar_matches_series_definition():
    chi = -0.4 + 0.7j
    coef = np.array([0.3 - 0.1j, 1.2 + 0.5j, -0.8 + 0.2j], dtype=np.complex128)

    result = mf.taylor_series(chi, coef)
    expected = _manual_taylor_series(chi, coef)

    assert np.isscalar(result)
    np.testing.assert_allclose(result, expected)


def test_taylor_series_array_matches_elementwise_evaluation():
    chi = np.array([0.2 + 0.1j, -0.5 + 0.3j, 1.0 - 0.4j], dtype=np.complex128)
    coef = np.array([0.3 - 0.1j, 1.2 + 0.5j, -0.8 + 0.2j], dtype=np.complex128)

    result = mf.taylor_series(chi, coef)
    expected = np.array(
        [_manual_taylor_series(c, coef) for c in chi], dtype=np.complex128
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == chi.shape
    np.testing.assert_allclose(result, expected)


def test_taylor_series_d1_scalar_matches_series_derivative():
    chi = -0.2 + 0.9j
    coef = np.array(
        [0.3 - 0.1j, 1.2 + 0.5j, -0.8 + 0.2j, 0.1 - 0.3j], dtype=np.complex128
    )

    result = mf.taylor_series_d1(chi, coef)
    expected = _manual_taylor_series_d1(chi, coef)

    np.testing.assert_allclose(result, expected)


def test_taylor_series_d1_array_input_raises_typing_error():
    chi = np.array([0.1 + 0.0j, 0.2 + 0.0j], dtype=np.complex128)
    coef = np.array([1.0 + 0.0j, 0.5 + 0.0j, -0.1 + 0.0j], dtype=np.complex128)

    with pytest.raises(TypingError):
        mf.taylor_series_d1(chi, coef)


def test_well_chi_scalar_and_array_match_closed_form():
    q = 2.5
    chi_scalar = 1.7 - 0.4j
    chi_array = np.array([1.7 - 0.4j, 0.8 + 0.6j, 2.0 + 0.0j], dtype=np.complex128)

    scalar_result = mf.well_chi(chi_scalar, q)
    array_result = mf.well_chi(chi_array, q)

    scalar_expected = q / (2 * np.pi) * np.log(chi_scalar)
    array_expected = q / (2 * np.pi) * np.log(chi_array)

    np.testing.assert_allclose(scalar_result, scalar_expected)
    assert isinstance(array_result, np.ndarray)
    assert array_result.shape == chi_array.shape
    np.testing.assert_allclose(array_result, array_expected)
