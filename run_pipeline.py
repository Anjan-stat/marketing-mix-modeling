"""
run_pipeline.py
---------------
End-to-end MMM pipeline:

  1. Generate synthetic data (or load real data)
  2. Fit the Marketing Mix Model
  3. Decompose contributions and compute ROI
  4. Optimise budget allocation
  5. Save all plots to outputs/

Run with:  python -m mmm.run_pipeline
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmm.data_generator import generate_data, CHANNELS, TRUE_PARAMS
from mmm.model import MarketingMixModel
from mmm.decomposition import decompose_contributions, compute_roi
from mmm.optimizer import BudgetOptimizer
from mmm import visualizer as viz

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    t0 = time.time()

    # ── 1. Data ────────────────────────────────────────────────────────────────
    banner("Step 1: Generating synthetic data")
    df_sales, df_spend, true_params = generate_data(n_weeks=156, seed=42)
    df_sales.to_csv("data/synthetic_sales.csv")
    df_spend.to_csv("data/synthetic_spend.csv")

    y = df_sales["sales"]
    print(f"  Weeks: {len(y)}  |  Date range: {y.index[0].date()} → {y.index[-1].date()}")
    print(f"  Avg weekly sales: ${y.mean():,.0f}")
    print(f"  Total spend ($M): {(df_spend.sum()/1e6).to_dict()}")

    # Control variables: linear trend + Fourier seasonality
    t_vec = np.arange(len(y))
    df_controls = pd.DataFrame({
        "trend": t_vec,
        "sin52": np.sin(2 * np.pi * t_vec / 52),
        "cos52": np.cos(2 * np.pi * t_vec / 52),
        "sin26": np.sin(2 * np.pi * t_vec / 26),
        "cos26": np.cos(2 * np.pi * t_vec / 26),
    }, index=df_spend.index)

    # ── 2. Model Fitting ───────────────────────────────────────────────────────
    banner("Step 2: Fitting Marketing Mix Model (Ridge + L-BFGS-B)")
    model = MarketingMixModel(
        channels=CHANNELS,
        control_cols=["trend", "sin52", "cos52", "sin26", "cos26"],
        ridge_alpha=0.1,
        test_size=0.2,
        search="lbfgs",
        seed=42,
    )
    model.fit(y, df_spend, df_controls)

    print("\nFitted parameters:")
    print(model.summary().to_string())

    # ── 3. Contribution Decomposition ──────────────────────────────────────────
    banner("Step 3: Decomposing contributions & computing ROI")
    decomp = decompose_contributions(model, y, df_spend, df_controls)
    roi_df = compute_roi(decomp, df_spend, CHANNELS)
    print("\nROI Table:")
    print(roi_df[["total_spend", "total_contribution", "avg_roi",
                   "spend_share", "contribution_share"]].to_string())

    # ── 4. Budget Optimisation ─────────────────────────────────────────────────
    banner("Step 4: Optimising budget allocation")
    optimizer = BudgetOptimizer(model, n_restarts=15, seed=0)
    total_budget = df_spend.mean().sum()
    opt_result = optimizer.optimise(total_budget, df_spend)
    table = optimizer.summary_table(opt_result)
    print(table.to_string())

    # ── 5. Plots ───────────────────────────────────────────────────────────────
    banner("Step 5: Generating plots")
    n_train = model._n_train

    plots = {
        "01_sales_decomposition":
            viz.plot_sales_decomposition(decomp, CHANNELS),
        "02_actual_vs_predicted":
            viz.plot_actual_vs_predicted(decomp, n_train),
        "03_adstock_curves":
            viz.plot_adstock_curves(model.result_.channel_params),
        "04_saturation_curves":
            viz.plot_saturation_curves(model, df_spend),
        "05_roi_bars":
            viz.plot_roi_bars(roi_df),
        "06_contribution_pie":
            viz.plot_contribution_pie(decomp, CHANNELS),
        "07_budget_optimisation":
            viz.plot_budget_optimisation(opt_result, CHANNELS),
        "08_parameter_recovery":
            viz.plot_parameter_recovery(model, true_params),
        "09_residuals":
            viz.plot_residuals(decomp),
    }

    for name, fig in plots.items():
        path = f"outputs/{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    banner("Pipeline complete")
    print(f"  Time elapsed: {time.time() - t0:.1f}s")
    print(f"  Model R² (test): {model.result_.r2_test:.3f}")
    print(f"  Model MAPE (test): {model.result_.mape_test:.2%}")
    print(f"  Revenue lift from optimisation: +{opt_result.revenue_lift_pct:.1%}")
    print(f"\n  All outputs saved to outputs/")


if __name__ == "__main__":
    main()
