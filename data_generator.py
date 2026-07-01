"""
data_generator.py
-----------------
Generates synthetic weekly sales data that follows the MMM data-generating
process (DGP) exactly, so we can verify model recovery.

True DGP
--------
  y_t = baseline_t + sum_m beta_m * hill(adstock(x_mt, lambda_m), K_m, n_m) + eps_t

where baseline_t includes trend + annual seasonality + holiday spikes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ── True parameter values used to generate synthetic data ─────────────────────

TRUE_PARAMS: Dict[str, dict] = {
    "tv": {
        "lambda": 0.70,   # high carryover — TV lingers
        "peak_lag": 1,
        "K": 0.55,        # half-saturation (normalised spend)
        "n": 1.5,         # Hill shape
        "beta": 12_000,   # incremental revenue per unit of transformed spend
        "weekly_budget_mean": 380_000,
        "weekly_budget_std": 60_000,
    },
    "radio": {
        "lambda": 0.40,
        "peak_lag": 0,
        "K": 0.45,
        "n": 1.2,
        "beta": 9_500,
        "weekly_budget_mean": 120_000,
        "weekly_budget_std": 25_000,
    },
    "digital_display": {
        "lambda": 0.30,
        "peak_lag": 0,
        "K": 0.35,
        "n": 1.0,
        "beta": 8_000,
        "weekly_budget_mean": 180_000,
        "weekly_budget_std": 40_000,
    },
    "paid_search": {
        "lambda": 0.10,   # very short carryover — intent-driven
        "peak_lag": 0,
        "K": 0.25,
        "n": 0.8,         # sub-linear — diminishes quickly
        "beta": 18_000,
        "weekly_budget_mean": 200_000,
        "weekly_budget_std": 35_000,
    },
    "social_media": {
        "lambda": 0.45,
        "peak_lag": 1,
        "K": 0.40,
        "n": 1.1,
        "beta": 10_000,
        "weekly_budget_mean": 120_000,
        "weekly_budget_std": 30_000,
    },
}

CHANNELS = list(TRUE_PARAMS.keys())


# ── Helper transformations (mirrors transformations.py for DGP use) ────────────

def _adstock(x: np.ndarray, lam: float, peak_lag: int = 0) -> np.ndarray:
    """Delayed geometric adstock."""
    T = len(x)
    a = np.zeros(T)
    for t in range(T):
        delayed = x[t - peak_lag] if t >= peak_lag else 0.0
        a[t] = delayed + lam * (a[t - 1] if t > 0 else 0.0)
    return a


def _hill(a: np.ndarray, K: float, n: float) -> np.ndarray:
    """Hill saturation: a^n / (K^n + a^n)."""
    return a**n / (K**n + a**n)


# ── Data generator ─────────────────────────────────────────────────────────────

def generate_data(
    n_weeks: int = 156,       # 3 years
    start_date: str = "2021-01-04",
    noise_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate synthetic MMM dataset.

    Returns
    -------
    df_sales : DataFrame with columns [date, sales, trend, seasonality, ...]
    df_spend : DataFrame with spend columns per channel
    true_params : dict of true parameter values
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    # ── Baseline: trend + seasonality + holiday spikes ────────────────────────
    trend = 500_000 + 800 * t                                    # slight upward trend
    seasonality = 80_000 * np.sin(2 * np.pi * t / 52 - np.pi / 2)  # annual cycle
    holiday_weeks = [48, 49, 50, 51, 52, 100, 101, 150, 151]        # Christmas, Easter, etc.
    holiday = np.zeros(n_weeks)
    for w in holiday_weeks:
        if w < n_weeks:
            holiday[w] = 60_000
    baseline = trend + seasonality + holiday                     # organic baseline

    # ── Media spend (with budget pulses / flights) ────────────────────────────
    spend_raw: Dict[str, np.ndarray] = {}
    for ch, p in TRUE_PARAMS.items():
        mu, sigma = p["weekly_budget_mean"], p["weekly_budget_std"]
        raw = rng.normal(mu, sigma, size=n_weeks).clip(0)
        # zero out some weeks to simulate flighting
        flight_off = rng.choice(n_weeks, size=n_weeks // 6, replace=False)
        raw[flight_off] = 0.0
        spend_raw[ch] = raw

    # ── Normalise spend to [0, 1] per channel (for stable saturation params) ──
    spend_norm: Dict[str, np.ndarray] = {
        ch: arr / arr.max() for ch, arr in spend_raw.items()
    }

    # ── Apply adstock + saturation and accumulate media contribution ──────────
    media_contrib: Dict[str, np.ndarray] = {}
    for ch, p in TRUE_PARAMS.items():
        a = _adstock(spend_norm[ch], lam=p["lambda"], peak_lag=p["peak_lag"])
        s = _hill(a, K=p["K"], n=p["n"])
        media_contrib[ch] = p["beta"] * s

    total_media = sum(media_contrib.values())

    # ── Add noise ─────────────────────────────────────────────────────────────
    signal = baseline + total_media
    noise = rng.normal(0, noise_frac * signal.mean(), size=n_weeks)
    sales = signal + noise

    # ── Assemble DataFrames ───────────────────────────────────────────────────
    df_spend = pd.DataFrame(spend_raw, index=dates)
    df_spend.index.name = "date"

    df_sales = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "baseline": baseline,
        "trend": trend,
        "seasonality": seasonality,
        "holiday": holiday,
        **{f"contrib_{ch}": media_contrib[ch] for ch in CHANNELS},
    })
    df_sales.set_index("date", inplace=True)

    return df_sales, df_spend, TRUE_PARAMS


if __name__ == "__main__":
    df_sales, df_spend, params = generate_data()
    print(df_sales[["sales", "baseline"]].describe())
    print("\nSpend totals ($M):")
    print((df_spend.sum() / 1e6).round(2))
    df_sales.to_csv("data/synthetic_sales.csv")
    df_spend.to_csv("data/synthetic_spend.csv")
    print("\nData saved to data/")
