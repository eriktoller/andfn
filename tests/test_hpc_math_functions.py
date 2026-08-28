import numpy as np
import pytest

from andfn.hpc import hpc_math_functions as hmf


def _manual_asym(chi, coef):
    return sum(coef[n] * chi ** (-n) for n in range(len(coef)))


def _manual_taylor(chi, coef):
    return sum(coef[n] * chi**n for n in range(len(coef)))


def test_series_and_derivatives_match_manual_values():
    chi = 1.2 + 0.4j
    coef = np.array([1.0 + 0.0j, 2.0 - 1.0j, -0.5 + 0.5j], dtype=np.complex128)

    assert hmf.asym_expansion(chi, coef) == pytest.approx(_manual_asym(chi, coef))
    assert hmf.taylor_series(chi, coef) == pytest.approx(_manual_taylor(chi, coef))

    eps = 1e-7
    num_d1_taylor = (
        _manual_taylor(chi + eps, coef) - _manual_taylor(chi - eps, coef)
    ) / (2 * eps)
    assert hmf.taylor_series_d1(chi, coef) == pytest.approx(
        num_d1_taylor, rel=1e-5, abs=1e-6
    )


def test_array_accumulators_and_well_chi():
    chi = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    coef = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    omega_a = np.zeros(2, dtype=np.complex128)
    hmf.asym_expansion_array(omega_a, chi, coef)
    expected_a = np.array([_manual_asym(chi[0], coef), _manual_asym(chi[1], coef)])
    assert np.allclose(omega_a, expected_a)

    omega_t = np.zeros(2, dtype=np.complex128)
    hmf.taylor_series_array(omega_t, chi, coef)
    expected_t = np.array([_manual_taylor(chi[0], coef), _manual_taylor(chi[1], coef)])
    assert np.allclose(omega_t, expected_t)

    q = 3.0
    assert hmf.well_chi(2.0 + 0.0j, q) == pytest.approx(
        q / (2 * np.pi) * np.log(2.0 + 0.0j)
    )
    omega_w = np.zeros(2, dtype=np.complex128)
    hmf.well_chi_array(omega_w, chi, q)
    assert np.allclose(omega_w, q / (2 * np.pi) * np.log(chi))


def test_theta_and_exp_helpers_and_formatters():
    thetas = np.zeros(4, dtype=np.float64)
    hmf.calc_thetas(4, 1, thetas)
    assert np.allclose(thetas, np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]))

    thetas2 = np.zeros(4, dtype=np.float64)
    hmf.calc_thetas(4, 0, thetas2)
    assert np.allclose(thetas2[0], np.pi / 8)

    exp_arr = np.zeros(4, dtype=np.complex128)
    hmf.fill_exp_array(4, thetas, exp_arr, 1)
    assert np.allclose(exp_arr, np.exp(1j * thetas))

    coef_ref = np.zeros(6, dtype=np.complex128)
    coef_ref[:5] = 1.0
    coef = coef_ref.copy()
    assert hmf.calc_error(coef, coef_ref) == pytest.approx(0.0)
    assert hmf.calc_coef_error(coef, coef_ref) == pytest.approx(0.0)

    assert hmf.cut_trail("1234000") == "1234"
    assert hmf.float2str(0.0) == "0.0"
    assert hmf.float2str(np.nan) == "nan"
