"""
tests/test_transformations.py
------------------------------
Unit tests for adstock and saturation functions.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmm.transformations import (
    geometric_adstock,
    delayed_adstock,
    hill_saturation,
    logistic_saturation,
    power_saturation,
    transform_spend,
    adstock_matrix,
)


# ── Adstock ────────────────────────────────────────────────────────────────────

class TestGeometricAdstock:
    def test_zero_spend_gives_zero(self):
        x = np.zeros(10)
        assert np.allclose(geometric_adstock(x, lam=0.5), 0)

    def test_impulse_decays_geometrically(self):
        x = np.zeros(10)
        x[0] = 1.0
        a = geometric_adstock(x, lam=0.5)
        expected = 0.5 ** np.arange(10)
        assert np.allclose(a, expected, atol=1e-10)

    def test_lam_zero_is_no_carryover(self):
        x = np.array([1.0, 2.0, 3.0])
        a = geometric_adstock(x, lam=0.0)
        assert np.allclose(a, x)

    def test_output_shape(self):
        x = np.ones(52)
        assert geometric_adstock(x, lam=0.3).shape == (52,)

    def test_non_negative(self):
        x = np.abs(np.random.randn(50))
        a = geometric_adstock(x, lam=0.7)
        assert (a >= 0).all()

    def test_matrix_form_matches_recursive(self):
        """Adstock via matrix multiplication must equal recursive formula."""
        T = 20
        x = np.random.rand(T)
        lam = 0.4
        a_recursive = geometric_adstock(x, lam=lam)
        L = adstock_matrix(T, lam)
        a_matrix = L @ x
        assert np.allclose(a_recursive, a_matrix, atol=1e-8)


class TestDelayedAdstock:
    def test_peak_at_theta(self):
        """For an impulse input, the maximum should be near lag theta."""
        x = np.zeros(30)
        x[0] = 1.0
        for theta in [0, 2, 4]:
            a = delayed_adstock(x, lam=0.5, theta=theta)
            assert a.argmax() == theta, f"Expected peak at {theta}, got {a.argmax()}"

    def test_output_shape(self):
        x = np.ones(52)
        assert delayed_adstock(x, lam=0.5, theta=2).shape == (52,)


# ── Saturation ─────────────────────────────────────────────────────────────────

class TestHillSaturation:
    def test_zero_input(self):
        assert hill_saturation(np.array([0.0]), K=0.5, n=1.0)[0] == 0.0

    def test_half_saturation_at_K(self):
        K = 0.4
        result = hill_saturation(np.array([K]), K=K, n=1.0)[0]
        assert abs(result - 0.5) < 1e-10

    def test_upper_bound_approx_one(self):
        large = np.array([1e6])
        assert hill_saturation(large, K=0.5, n=1.0)[0] > 0.9999

    def test_monotone_increasing(self):
        a = np.linspace(0, 2, 100)
        s = hill_saturation(a, K=0.5, n=1.5)
        assert (np.diff(s) >= 0).all()

    def test_n_greater_one_is_sigmoidal(self):
        """With n>1, there is an inflection point."""
        a = np.linspace(0, 1, 1000)
        s = hill_saturation(a, K=0.5, n=2.0)
        second_deriv = np.diff(np.diff(s))
        # Should change sign
        assert (second_deriv > 0).any() and (second_deriv < 0).any()

    def test_output_shape(self):
        a = np.ones((10, 5))
        # hill should broadcast
        s = hill_saturation(a, K=0.5, n=1.0)
        assert s.shape == (10, 5)


class TestLogisticSaturation:
    def test_inflection_at_x0(self):
        x0 = 0.5
        L = 1.0
        result = logistic_saturation(np.array([x0]), L=L, k=10, x0=x0)[0]
        assert abs(result - L / 2) < 1e-4

    def test_bounded_by_L(self):
        a = np.linspace(0, 100, 200)
        s = logistic_saturation(a, L=5.0, k=1.0, x0=2.0)
        assert (s <= 5.0 + 1e-10).all()
        assert (s >= 0).all()


class TestPowerSaturation:
    def test_alpha_one_is_linear(self):
        a = np.array([1.0, 2.0, 3.0])
        assert np.allclose(power_saturation(a, alpha=1.0), a)

    def test_concavity(self):
        a = np.linspace(0.01, 2, 100)
        s = power_saturation(a, alpha=0.5)
        second_deriv = np.diff(np.diff(s))
        assert (second_deriv < 0).all()


# ── Combined pipeline ──────────────────────────────────────────────────────────

class TestTransformSpend:
    def test_output_in_zero_one(self):
        x = np.abs(np.random.randn(52)) * 10_000
        x_star = transform_spend(x, lam=0.5, K=0.4, n=1.2)
        assert x_star.min() >= 0
        assert x_star.max() <= 1.0 + 1e-10

    def test_zero_spend_gives_zero(self):
        x = np.zeros(52)
        x_star = transform_spend(x, lam=0.5, K=0.4, n=1.0)
        assert np.allclose(x_star, 0)

    def test_shape_preserved(self):
        x = np.ones(100)
        assert transform_spend(x, lam=0.3, K=0.5, n=1.0).shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
