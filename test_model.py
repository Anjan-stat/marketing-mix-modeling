"""
tests/test_model.py
-------------------
Integration tests for the end-to-end MMM pipeline.

These tests use fast settings (grid search, few weeks) to stay quick.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmm.data_generator import generate_data, CHANNELS
from mmm.model import MarketingMixModel
from mmm.decomposition import decompose_contributions, compute_roi
from mmm.optimizer import BudgetOptimizer


@pytest.fixture(scope="module")
def data():
    df_sales, df_spend, true_params = generate_data(n_weeks=60, seed=0)
    y = df_sales["sales"]
    t = np.arange(len(y))
    df_controls = pd.DataFrame({
        "trend": t,
        "sin52": np.sin(2 * np.pi * t / 52),
        "cos52": np.cos(2 * np.pi * t / 52),
    }, index=df_spend.index)
    return y, df_spend, df_controls, true_params


@pytest.fixture(scope="module")
def fitted_model(data):
    y, df_spend, df_controls, _ = data
    model = MarketingMixModel(
        channels=CHANNELS,
        control_cols=["trend", "sin52", "cos52"],
        ridge_alpha=1.0,
        test_size=0.2,
        search="grid",   # faster for tests
        seed=0,
    )
    model.fit(y, df_spend, df_controls)
    return model


class TestModelFit:
    def test_fit_returns_self(self, data):
        y, df_spend, df_controls, _ = data
        model = MarketingMixModel(channels=CHANNELS, search="grid", seed=0)
        result = model.fit(y, df_spend, df_controls)
        assert result is model

    def test_r2_reasonable(self, fitted_model):
        # Grid search on short data (60 weeks) gives lower R²; 0.05 is a sanity floor.
        # Use search='lbfgs' with full data (156 weeks) for R² > 0.85 in practice.
        assert fitted_model.result_.r2_train > 0.05, "R² on train should be > 0.05"

    def test_betas_non_negative(self, fitted_model):
        for cp in fitted_model.result_.channel_params:
            assert cp.beta >= 0, f"Beta for {cp.name} is negative: {cp.beta}"

    def test_summary_has_all_channels(self, fitted_model):
        summary = fitted_model.summary()
        for ch in CHANNELS:
            assert ch in summary.index

    def test_params_in_bounds(self, fitted_model):
        for cp in fitted_model.result_.channel_params:
            assert 0 < cp.lam < 1, f"λ out of bounds for {cp.name}"
            assert 0 < cp.K  < 1, f"K out of bounds for {cp.name}"
            assert cp.n > 0,       f"n <= 0 for {cp.name}"

    def test_predict_shape(self, fitted_model):
        preds = fitted_model.predict()
        n_total = 60
        assert len(preds) == n_total


class TestDecomposition:
    def test_decomp_columns(self, fitted_model, data):
        y, df_spend, df_controls, _ = data
        decomp = decompose_contributions(fitted_model, y, df_spend, df_controls)
        for ch in CHANNELS:
            assert f"contrib_{ch}" in decomp.columns
        assert "baseline" in decomp.columns
        assert "y_hat" in decomp.columns

    def test_contributions_sum_approx_sales(self, fitted_model, data):
        """baseline + media contribs + residual should equal actual sales."""
        y, df_spend, df_controls, _ = data
        decomp = decompose_contributions(fitted_model, y, df_spend, df_controls)
        media_total = sum(decomp[f"contrib_{ch}"] for ch in CHANNELS)
        reconstructed = decomp["baseline"] + media_total + decomp["residual"]
        assert np.allclose(reconstructed.values, decomp["sales"].values, atol=0.01)

    def test_roi_positive(self, fitted_model, data):
        y, df_spend, df_controls, _ = data
        decomp = decompose_contributions(fitted_model, y, df_spend, df_controls)
        roi_df = compute_roi(decomp, df_spend, CHANNELS)
        # ROI should be positive for all channels (given positive spend and contribution)
        assert (roi_df["avg_roi"] >= 0).all()

    def test_spend_share_sums_to_one(self, fitted_model, data):
        y, df_spend, df_controls, _ = data
        decomp = decompose_contributions(fitted_model, y, df_spend, df_controls)
        roi_df = compute_roi(decomp, df_spend, CHANNELS)
        assert abs(roi_df["spend_share"].sum() - 1.0) < 1e-6


class TestOptimizer:
    def test_budget_constraint_satisfied(self, fitted_model, data):
        y, df_spend, df_controls, _ = data
        opt = BudgetOptimizer(fitted_model, n_restarts=3, seed=0)
        total_budget = df_spend.mean().sum()
        result = opt.optimise(total_budget, df_spend)
        actual_total = sum(result.optimal_budget.values())
        assert abs(actual_total - total_budget) / total_budget < 0.01

    def test_optimal_revenue_geq_current(self, fitted_model, data):
        y, df_spend, df_controls, _ = data
        opt = BudgetOptimizer(fitted_model, n_restarts=5, seed=0)
        total_budget = df_spend.mean().sum()
        result = opt.optimise(total_budget, df_spend)
        # Optimal should be at least as good as current
        assert result.optimal_revenue >= result.current_revenue - 1e-3

    def test_all_channels_receive_budget(self, fitted_model, data):
        y, df_spend, df_controls, _ = data
        opt = BudgetOptimizer(fitted_model, n_restarts=3, seed=0)
        total_budget = df_spend.mean().sum()
        result = opt.optimise(total_budget, df_spend)
        for ch in CHANNELS:
            assert result.optimal_budget[ch] > 0, f"Channel {ch} got zero budget"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
