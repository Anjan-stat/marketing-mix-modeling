"""
decomposition.py
----------------
Channel contribution decomposition and ROI analysis.

For a fitted MMM, each week's sales are decomposed as:

    y_t = baseline_t + sum_m contrib_{m,t} + residual_t

where:

    baseline_t    = intercept + control contributions
    contrib_{m,t} = beta_m * s(a_{m,t})   (media contribution)
    residual_t    = y_t - y_hat_t

Revenue ROI for channel m:

    ROI_m = sum_t contrib_{m,t} / sum_t x_{m,t}

Incremental ROI (marginal at current spend level):

    mROI_m = d(contrib_m) / d(x_m) |_{x=x_bar}
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model import MarketingMixModel, ChannelParams
from .transformations import geometric_adstock, hill_saturation, marginal_roi


def decompose_contributions(
    model: MarketingMixModel,
    y: pd.Series,
    X_spend: pd.DataFrame,
    X_controls: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Decompose fitted sales into baseline + per-channel contributions.

    Returns
    -------
    pd.DataFrame with columns:
        date, sales, y_hat, baseline, residual,
        contrib_<channel> for each channel,
        pct_<channel> (percentage share of total)
    """
    model._check_fitted()
    result = model.result_
    channels = model.channels
    y_vals = y.values.astype(float)
    T = len(y_vals)

    # ── Compute per-channel contributions ──────────────────────────────────────
    contributions: Dict[str, np.ndarray] = {}
    for cp in result.channel_params:
        x_norm = X_spend[cp.name].values / model._spend_max[cp.name]
        a = geometric_adstock(x_norm, lam=cp.lam)
        if a.max() > 0:
            a /= a.max()
        sat = hill_saturation(a, K=cp.K, n=cp.n)
        contributions[cp.name] = cp.beta * sat

    # ── Baseline: intercept + control variable contributions ───────────────────
    baseline = np.full(T, result.intercept)
    if X_controls is not None and result.control_coefs:
        for col, coef in result.control_coefs.items():
            if col in X_controls.columns:
                vals = X_controls[col].values.astype(float)
                norm_vals = (vals - vals.mean()) / (vals.std() + 1e-8)
                baseline += coef * norm_vals

    # ── Total predicted ────────────────────────────────────────────────────────
    total_media = sum(contributions.values())
    y_hat = baseline + total_media
    residual = y_vals - y_hat

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame({"date": y.index, "sales": y_vals, "y_hat": y_hat,
                       "baseline": baseline, "residual": residual})
    for ch, c in contributions.items():
        df[f"contrib_{ch}"] = c
    df.set_index("date", inplace=True)

    # Percentage decomposition
    total_positive = df[[f"contrib_{ch}" for ch in channels]].clip(lower=0).sum(axis=1) \
                     + df["baseline"].clip(lower=0)
    for ch in channels:
        df[f"pct_{ch}"] = (df[f"contrib_{ch}"].clip(lower=0) / total_positive).fillna(0)
    df["pct_baseline"] = (df["baseline"].clip(lower=0) / total_positive).fillna(0)

    return df


def compute_roi(
    decomp: pd.DataFrame,
    X_spend: pd.DataFrame,
    channels: List[str],
    price_per_unit: float = 1.0,
) -> pd.DataFrame:
    """
    Compute average and marginal ROI per channel.

    Parameters
    ----------
    decomp : output of decompose_contributions
    X_spend : raw spend DataFrame
    channels : list of channel names
    price_per_unit : revenue per unit of sales (if y is in units, not $)

    Returns
    -------
    pd.DataFrame with columns:
        channel, total_spend, total_contribution, avg_roi,
        spend_share, contribution_share
    """
    rows = []
    for ch in channels:
        total_spend = X_spend[ch].sum()
        total_contrib = decomp[f"contrib_{ch}"].sum() * price_per_unit
        avg_roi = total_contrib / total_spend if total_spend > 0 else np.nan
        rows.append({
            "channel": ch,
            "total_spend": total_spend,
            "total_contribution": total_contrib,
            "avg_roi": avg_roi,
        })
    df = pd.DataFrame(rows).set_index("channel")
    df["spend_share"] = df["total_spend"] / df["total_spend"].sum()
    df["contribution_share"] = df["total_contribution"] / df["total_contribution"].sum()
    return df.sort_values("avg_roi", ascending=False)


def response_curves(
    model: MarketingMixModel,
    X_spend: pd.DataFrame,
    n_points: int = 100,
    scale_factor: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """
    Compute S-shaped response curves for each channel.

    For each channel, sweep spend from 0 to scale_factor * current_max
    and compute the resulting contribution holding other channels constant.

    Returns
    -------
    dict mapping channel name → DataFrame(spend, contribution, marginal_roi)
    """
    model._check_fitted()
    curves = {}
    for cp in model.result_.channel_params:
        x_max = X_spend[cp.name].max()
        spend_range = np.linspace(0, scale_factor * x_max, n_points)
        # Normalise to the same scale used in training
        x_norm = spend_range / model._spend_max[cp.name]
        # Single-period adstock (no carryover in a one-shot sweep)
        sat = hill_saturation(x_norm, K=cp.K, n=cp.n)
        contribution = cp.beta * sat
        mroi = np.gradient(contribution, spend_range)
        curves[cp.name] = pd.DataFrame({
            "spend": spend_range,
            "contribution": contribution,
            "marginal_roi": mroi,
        })
    return curves


def waterfall_summary(decomp: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    """
    Average weekly contribution waterfall (for visualisation).
    """
    avg = {"Baseline": decomp["baseline"].mean()}
    for ch in channels:
        avg[ch.replace("_", " ").title()] = decomp[f"contrib_{ch}"].mean()
    avg["Total Sales"] = decomp["sales"].mean()
    return pd.Series(avg).rename("avg_weekly_contribution")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .data_generator import generate_data, CHANNELS
    from .model import MarketingMixModel

    df_sales, df_spend, _ = generate_data(seed=42)
    y = df_sales["sales"]
    df_controls = pd.DataFrame({
        "trend": np.arange(len(y)),
        "sin52": np.sin(2 * np.pi * np.arange(len(y)) / 52),
        "cos52": np.cos(2 * np.pi * np.arange(len(y)) / 52),
    }, index=df_spend.index)

    model = MarketingMixModel(channels=CHANNELS, control_cols=["trend","sin52","cos52"])
    model.fit(y, df_spend, df_controls)

    decomp = decompose_contributions(model, y, df_spend, df_controls)
    roi_df  = compute_roi(decomp, df_spend, CHANNELS)
    print(roi_df[["total_spend","total_contribution","avg_roi","spend_share","contribution_share"]])
