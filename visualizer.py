"""
visualizer.py
-------------
All MMM visualisations in one place.

Plots
-----
1.  plot_sales_decomposition   — stacked area chart (waterfall)
2.  plot_actual_vs_predicted   — time series fit quality
3.  plot_adstock_curves        — impulse response per channel
4.  plot_saturation_curves     — response curves with current spend markers
5.  plot_roi_bars              — ROI comparison bar chart
6.  plot_contribution_pie      — revenue attribution pie
7.  plot_budget_optimisation   — before/after budget allocation
8.  plot_efficient_frontier    — budget vs revenue curve
9.  plot_parameter_recovery    — true vs estimated parameters (for synthetic data)
10. plot_residuals             — residual diagnostics
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .transformations import geometric_adstock, hill_saturation
from .model import MarketingMixModel
from .decomposition import decompose_contributions, compute_roi, response_curves
from .optimizer import OptimResult

# ── Palette ────────────────────────────────────────────────────────────────────
CHANNEL_COLORS = {
    "tv":             "#2196F3",
    "radio":          "#FF9800",
    "digital_display":"#4CAF50",
    "paid_search":    "#9C27B0",
    "social_media":   "#F44336",
    "baseline":       "#78909C",
}
DEFAULT_FIGSIZE = (14, 5)
sns.set_theme(style="whitegrid", palette="muted")


def _ch_label(ch: str) -> str:
    return ch.replace("_", " ").title()


# ── 1. Sales Decomposition ─────────────────────────────────────────────────────

def plot_sales_decomposition(
    decomp: pd.DataFrame,
    channels: List[str],
    title: str = "Sales Decomposition",
    figsize=(16, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    stack_cols = ["baseline"] + [f"contrib_{ch}" for ch in channels]
    stack_labels = ["Baseline"] + [_ch_label(ch) for ch in channels]
    colors = [CHANNEL_COLORS.get("baseline", "#78909C")] + \
             [CHANNEL_COLORS.get(ch, "#999") for ch in channels]

    vals = decomp[stack_cols].clip(lower=0).values
    ax.stackplot(decomp.index, vals.T, labels=stack_labels, colors=colors, alpha=0.85)
    ax.plot(decomp.index, decomp["sales"], color="black", lw=1.5, label="Actual Sales")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ── 2. Actual vs Predicted ─────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    decomp: pd.DataFrame,
    n_train: int,
    figsize=DEFAULT_FIGSIZE,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Time series
    ax1.plot(decomp.index, decomp["sales"], label="Actual", color="#333", lw=1.5)
    ax1.plot(decomp.index, decomp["y_hat"], label="Predicted", color="#1976D2",
             lw=1.5, linestyle="--")
    ax1.axvline(decomp.index[n_train], color="red", lw=1, ls=":", label="Train/Test split")
    ax1.set_title("Actual vs Predicted Sales", fontweight="bold")
    ax1.set_ylabel("Sales ($)")
    ax1.legend()
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))

    # Scatter
    ax2.scatter(decomp["sales"], decomp["y_hat"], alpha=0.5, s=20, color="#1976D2")
    lims = [min(decomp["sales"].min(), decomp["y_hat"].min()),
            max(decomp["sales"].max(), decomp["y_hat"].max())]
    ax2.plot(lims, lims, "r--", lw=1.5, label="y = ŷ")
    r2 = np.corrcoef(decomp["sales"], decomp["y_hat"])[0, 1] ** 2
    ax2.set_title(f"Predicted vs Actual  (R²={r2:.3f})", fontweight="bold")
    ax2.set_xlabel("Actual Sales ($)")
    ax2.set_ylabel("Predicted Sales ($)")
    ax2.legend()

    fig.tight_layout()
    return fig


# ── 3. Adstock Curves ──────────────────────────────────────────────────────────

def plot_adstock_curves(
    channel_params,
    figsize=(14, 4),
) -> plt.Figure:
    T = 20
    impulse = np.zeros(T)
    impulse[0] = 1.0
    fig, ax = plt.subplots(figsize=figsize)
    for cp in channel_params:
        a = geometric_adstock(impulse, lam=cp.lam)
        ax.plot(a, marker="o", ms=4, label=f"{_ch_label(cp.name)} (λ={cp.lam:.2f})",
                color=CHANNEL_COLORS.get(cp.name, "#999"))
    ax.set_title("Adstock Impulse Response (unit spend at t=0)", fontweight="bold")
    ax.set_xlabel("Weeks after exposure")
    ax.set_ylabel("Carryover weight")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── 4. Saturation Curves ───────────────────────────────────────────────────────

def plot_saturation_curves(
    model: MarketingMixModel,
    X_spend: pd.DataFrame,
    figsize=(16, 10),
) -> plt.Figure:
    channels = model.channels
    n = len(channels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    curves = response_curves(model, X_spend, n_points=200, scale_factor=1.8)

    for i, cp in enumerate(model.result_.channel_params):
        ax = axes[i]
        df_curve = curves[cp.name]
        color = CHANNEL_COLORS.get(cp.name, "#999")

        ax.plot(df_curve["spend"] / 1e3, df_curve["contribution"] / 1e3,
                color=color, lw=2)
        ax.fill_between(df_curve["spend"] / 1e3, df_curve["contribution"] / 1e3,
                         alpha=0.15, color=color)

        # Mark current average spend
        cur_spend = X_spend[cp.name].mean()
        cur_sat = hill_saturation(np.array([cur_spend / model._spend_max[cp.name]]),
                                   K=cp.K, n=cp.n)[0]
        cur_contrib = cp.beta * cur_sat
        ax.axvline(cur_spend / 1e3, color="grey", lw=1, ls="--")
        ax.axhline(cur_contrib / 1e3, color="grey", lw=1, ls="--")
        ax.scatter([cur_spend / 1e3], [cur_contrib / 1e3], color=color, s=80, zorder=5)

        ax.set_title(_ch_label(cp.name), fontweight="bold")
        ax.set_xlabel("Spend ($K/week)")
        ax.set_ylabel("Contribution ($K/week)")
        ax.text(0.97, 0.05, f"K={cp.K:.2f}  n={cp.n:.2f}",
                transform=ax.transAxes, ha="right", fontsize=8, color="grey")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Media Saturation Response Curves  (● = current average spend)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


# ── 5. ROI Bar Chart ───────────────────────────────────────────────────────────

def plot_roi_bars(
    roi_df: pd.DataFrame,
    figsize=(12, 5),
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    channels = roi_df.index.tolist()
    colors = [CHANNEL_COLORS.get(ch, "#999") for ch in channels]

    # ROI
    bars = ax1.barh(channels, roi_df["avg_roi"], color=colors, edgecolor="white")
    ax1.bar_label(bars, fmt="{:.2f}x", padding=3, fontsize=9)
    ax1.set_title("Average ROI by Channel", fontweight="bold")
    ax1.set_xlabel("Revenue ROI ($ per $ spent)")
    ax1.invert_yaxis()

    # Spend vs Contribution share
    x = np.arange(len(channels))
    w = 0.35
    ax2.barh(x - w/2, roi_df["spend_share"] * 100, w, color=colors, alpha=0.7, label="Spend Share")
    ax2.barh(x + w/2, roi_df["contribution_share"] * 100, w, color=colors, label="Contribution Share")
    ax2.set_yticks(x)
    ax2.set_yticklabels([_ch_label(ch) for ch in channels])
    ax2.invert_yaxis()
    ax2.set_title("Spend vs Revenue Contribution Share", fontweight="bold")
    ax2.set_xlabel("Share (%)")
    ax2.legend()

    fig.tight_layout()
    return fig


# ── 6. Contribution Pie ────────────────────────────────────────────────────────

def plot_contribution_pie(
    decomp: pd.DataFrame,
    channels: List[str],
    figsize=(8, 8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    labels = ["Baseline"] + [_ch_label(ch) for ch in channels]
    colors = [CHANNEL_COLORS.get("baseline")] + [CHANNEL_COLORS.get(ch, "#999") for ch in channels]
    vals = [decomp["baseline"].clip(lower=0).mean()] + \
           [decomp[f"contrib_{ch}"].clip(lower=0).mean() for ch in channels]
    explode = [0.02] * len(vals)
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 10},
    )
    ax.set_title("Average Weekly Revenue Attribution", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


# ── 7. Budget Optimisation ─────────────────────────────────────────────────────

def plot_budget_optimisation(
    result: OptimResult,
    channels: List[str],
    figsize=(13, 5),
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = [CHANNEL_COLORS.get(ch, "#999") for ch in channels]
    cur_vals = [result.current_budget[ch] / 1e3 for ch in channels]
    opt_vals = [result.optimal_budget[ch] / 1e3 for ch in channels]
    x = np.arange(len(channels))
    w = 0.35

    # Grouped bar
    ax1.bar(x - w/2, cur_vals, w, label="Current", color=[c+"99" for c in colors], edgecolor="white")
    ax1.bar(x + w/2, opt_vals, w, label="Optimal", color=colors, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([_ch_label(ch) for ch in channels], rotation=15, ha="right")
    ax1.set_ylabel("Weekly Budget ($K)")
    ax1.set_title("Budget Reallocation", fontweight="bold")
    ax1.legend()

    # Revenue lift
    rev_data = {
        "Current": result.current_revenue / 1e3,
        "Optimal": result.optimal_revenue / 1e3,
    }
    bar_colors = ["#90A4AE", "#4CAF50"]
    bars = ax2.bar(rev_data.keys(), rev_data.values(), color=bar_colors, edgecolor="white", width=0.4)
    ax2.bar_label(bars, fmt="${:,.0f}K", padding=3, fontsize=10)
    ax2.set_title(f"Media Revenue Uplift  (+{result.revenue_lift_pct:.1%})", fontweight="bold")
    ax2.set_ylabel("Estimated Media Revenue ($K)")

    fig.suptitle("Budget Optimisation Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 8. Efficient Frontier ──────────────────────────────────────────────────────

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    figsize=(10, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frontier_df["budget"] / 1e6, frontier_df["optimal_revenue"] / 1e6,
            color="#1976D2", lw=2.5, label="Optimal allocation")
    ax.fill_between(frontier_df["budget"] / 1e6, frontier_df["optimal_revenue"] / 1e6,
                    alpha=0.1, color="#1976D2")
    ax.set_title("Budget–Revenue Efficient Frontier", fontweight="bold", fontsize=13)
    ax.set_xlabel("Total Media Budget ($M)")
    ax.set_ylabel("Optimal Media Revenue ($M)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 9. Parameter Recovery ──────────────────────────────────────────────────────

def plot_parameter_recovery(
    model: MarketingMixModel,
    true_params: dict,
    figsize=(14, 4),
) -> plt.Figure:
    channels = model.channels
    param_names = ["lambda", "K", "n"]
    true_key_map = {"lambda": "lambda", "K": "K", "n": "n"}
    est_key_map  = {"lambda": "lam",    "K": "K", "n": "n"}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, p in zip(axes, param_names):
        true_vals = [true_params[ch][true_key_map[p]] for ch in channels]
        est_vals  = [getattr(cp, est_key_map[p]) for cp in model.result_.channel_params]
        ax.scatter(true_vals, est_vals, s=80,
                   color=[CHANNEL_COLORS.get(ch, "#999") for ch in channels])
        for ch, t, e in zip(channels, true_vals, est_vals):
            ax.annotate(_ch_label(ch), (t, e), textcoords="offset points",
                        xytext=(5, 3), fontsize=7)
        lims = [min(true_vals + est_vals) - 0.05, max(true_vals + est_vals) + 0.05]
        ax.plot(lims, lims, "r--", lw=1, label="y=x")
        ax.set_xlabel(f"True {p}")
        ax.set_ylabel(f"Estimated {p}")
        ax.set_title(f"Parameter Recovery: {p}", fontweight="bold")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── 10. Residual Diagnostics ───────────────────────────────────────────────────

def plot_residuals(
    decomp: pd.DataFrame,
    figsize=(14, 8),
) -> plt.Figure:
    resid = decomp["residual"]
    y_hat = decomp["y_hat"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)

    # Residuals over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(decomp.index, resid, color="#555", lw=1)
    ax1.axhline(0, color="red", lw=1, ls="--")
    ax1.fill_between(decomp.index, resid, alpha=0.2, color="#555")
    ax1.set_title("Residuals Over Time", fontweight="bold")
    ax1.set_ylabel("Residual ($)")

    # Residuals vs Fitted
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_hat, resid, alpha=0.4, s=15, color="#1976D2")
    ax2.axhline(0, color="red", lw=1, ls="--")
    ax2.set_xlabel("Fitted Values ($)")
    ax2.set_ylabel("Residuals ($)")
    ax2.set_title("Residuals vs Fitted", fontweight="bold")

    # Q-Q plot
    ax3 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    ax3.scatter(osm, osr, s=15, alpha=0.5, color="#1976D2")
    ax3.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.set_title(f"Normal Q-Q Plot  (r={r:.3f})", fontweight="bold")

    fig.suptitle("Residual Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
