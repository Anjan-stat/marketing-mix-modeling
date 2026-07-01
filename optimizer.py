"""
optimizer.py
------------
Budget optimisation for Marketing Mix Models.

Problem formulation
-------------------
Given a total budget B and fitted MMM response curves, solve:

    max_{b >= 0}  sum_m  beta_m * hill(b_m / b_max_m, K_m, n_m)

    subject to:
        sum_m b_m = B                      (budget constraint)
        b_m >= b_min_m  for all m          (lower bounds per channel)
        b_m <= b_max_m  for all m          (upper bounds per channel)

This is a concave maximisation problem (hill is concave when n <= 1,
and approximately concave for moderate n in the operating range),
solved efficiently with SLSQP.

For n > 1 the objective can be non-concave (S-shaped), so we use
multiple random restarts to find a good local optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from .model import MarketingMixModel, ChannelParams
from .transformations import hill_saturation


@dataclass
class OptimResult:
    """Result of budget optimisation."""
    current_budget: Dict[str, float]
    optimal_budget: Dict[str, float]
    current_revenue: float
    optimal_revenue: float
    revenue_lift: float          # absolute
    revenue_lift_pct: float      # percentage
    converged: bool
    n_restarts_used: int


class BudgetOptimizer:
    """
    Constrained nonlinear budget optimiser.

    Parameters
    ----------
    model : fitted MarketingMixModel
    n_restarts : number of random restarts (more = more robust for non-concave)
    seed : random seed
    """

    def __init__(
        self,
        model: MarketingMixModel,
        n_restarts: int = 20,
        seed: int = 0,
    ):
        self.model = model
        self.n_restarts = n_restarts
        self.seed = seed
        model._check_fitted()
        self._ch_params: List[ChannelParams] = model.result_.channel_params

    # ── Main optimisation ──────────────────────────────────────────────────────

    def optimise(
        self,
        total_budget: float,
        X_spend: pd.DataFrame,
        min_budget_pct: float = 0.05,   # each channel must get ≥5% of its current spend
        max_budget_pct: float = 3.0,    # each channel can receive ≤300% of current spend
        fixed_channels: Optional[Dict[str, float]] = None,
    ) -> OptimResult:
        """
        Find the spend allocation that maximises total media revenue.

        Parameters
        ----------
        total_budget : total budget ($) to allocate
        X_spend : historical spend (used for current-budget reference & bounds)
        min_budget_pct : minimum spend fraction relative to current average
        max_budget_pct : maximum spend fraction relative to current average
        fixed_channels : dict of {channel: fixed_spend} to hold constant

        Returns
        -------
        OptimResult
        """
        channels = self.model.channels
        n = len(channels)
        fixed_channels = fixed_channels or {}

        # Current average weekly spend
        current_avg = {ch: X_spend[ch].mean() for ch in channels}
        current_total = sum(current_avg.values())
        # Scale current_avg to match total_budget (apples-to-apples)
        scale = total_budget / current_total
        current_scaled = {ch: v * scale for ch, v in current_avg.items()}

        # Channel bounds
        lb = np.array([
            fixed_channels.get(ch, min_budget_pct * current_scaled[ch])
            for ch in channels
        ])
        ub = np.array([
            fixed_channels.get(ch, max_budget_pct * current_scaled[ch])
            for ch in channels
        ])

        # Adjust total_budget for fixed channels
        fixed_total = sum(fixed_channels.values())
        free_budget = total_budget - fixed_total
        free_mask = np.array([ch not in fixed_channels for ch in channels])

        # Normalise factors for each channel (spend_max from training)
        spend_max = np.array([self.model._spend_max[ch] for ch in channels])
        betas = np.array([cp.beta for cp in self._ch_params])
        Ks    = np.array([cp.K   for cp in self._ch_params])
        ns    = np.array([cp.n   for cp in self._ch_params])

        def revenue(b: np.ndarray) -> float:
            """Total media contribution for spend vector b."""
            b_norm = b / spend_max
            sat = hill_saturation(b_norm, K=Ks, n=ns)   # vectorised
            return float(np.dot(betas, sat))

        def neg_revenue(b):
            return -revenue(b)

        def neg_revenue_grad(b):
            """Analytical gradient of -revenue w.r.t. b."""
            b_norm = b / spend_max
            # d/db [beta * a^n / (K^n + a^n)]  where a = b/bmax
            Kn = Ks ** ns
            bn = b_norm ** ns
            dsat_da = ns * Kn * b_norm**(ns - 1) / (Kn + bn)**2
            da_db = 1.0 / spend_max
            return -betas * dsat_da * da_db

        # Equality constraint: free channels sum to free_budget
        # (fixed channels already at their fixed value)
        constraints = [
            {
                "type": "eq",
                "fun": lambda b: np.sum(b[free_mask]) - free_budget,
                "jac": lambda b: free_mask.astype(float),
            }
        ]

        best_val = np.inf
        best_b = None
        rng = np.random.default_rng(self.seed)
        converged = False

        for restart in range(self.n_restarts):
            # Random initial point (Dirichlet allocation)
            b0 = np.array([
                fixed_channels.get(ch, current_scaled[ch])
                for ch in channels
            ], dtype=float)
            if restart > 0:
                # Perturb free channels
                alpha = rng.dirichlet(np.ones(free_mask.sum())) * free_budget
                b0[free_mask] = alpha

            # Clip to bounds
            b0 = np.clip(b0, lb, ub)

            res = minimize(
                neg_revenue,
                b0,
                jac=neg_revenue_grad,
                method="SLSQP",
                bounds=list(zip(lb, ub)),
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_b = res.x.copy()
                converged = res.success

        # Build result
        optimal_budget = {ch: float(best_b[i]) for i, ch in enumerate(channels)}
        current_rev = revenue(np.array([current_scaled[ch] for ch in channels]))
        optimal_rev = -best_val

        return OptimResult(
            current_budget=current_scaled,
            optimal_budget=optimal_budget,
            current_revenue=current_rev,
            optimal_revenue=optimal_rev,
            revenue_lift=optimal_rev - current_rev,
            revenue_lift_pct=(optimal_rev - current_rev) / current_rev,
            converged=converged,
            n_restarts_used=self.n_restarts,
        )

    # ── Efficient frontier ─────────────────────────────────────────────────────

    def efficient_frontier(
        self,
        X_spend: pd.DataFrame,
        budget_range: Optional[np.ndarray] = None,
        n_points: int = 30,
    ) -> pd.DataFrame:
        """
        Compute the media revenue-vs-budget efficient frontier.

        Sweeps total budget from 50% to 200% of current spend.
        At each budget level, runs the full optimiser.

        Returns
        -------
        pd.DataFrame with columns: budget, optimal_revenue, current_revenue
        """
        current_total = sum(X_spend[ch].mean() for ch in self.model.channels)
        if budget_range is None:
            budget_range = np.linspace(0.5 * current_total, 2.0 * current_total, n_points)

        rows = []
        for B in budget_range:
            opt = self.optimise(B, X_spend)
            rows.append({
                "budget": B,
                "optimal_revenue": opt.optimal_revenue,
                "current_revenue": opt.current_revenue,
            })
            print(f"  Budget ${B/1e6:.2f}M → Revenue ${opt.optimal_revenue/1e6:.2f}M")
        return pd.DataFrame(rows)

    # ── Convenience ───────────────────────────────────────────────────────────

    def summary_table(self, result: OptimResult) -> pd.DataFrame:
        """Print a before/after comparison table."""
        channels = self.model.channels
        rows = []
        for ch in channels:
            cur = result.current_budget[ch]
            opt = result.optimal_budget[ch]
            rows.append({
                "channel": ch,
                "current_budget": round(cur),
                "optimal_budget": round(opt),
                "change_pct": f"{(opt - cur) / cur:+.1%}",
                "current_share": f"{cur / sum(result.current_budget.values()):.1%}",
                "optimal_share": f"{opt / sum(result.optimal_budget.values()):.1%}",
            })
        df = pd.DataFrame(rows).set_index("channel")
        print(f"\n{'='*60}")
        print(f"  Budget Optimisation Result")
        print(f"  Total Budget: ${sum(result.optimal_budget.values())/1e6:.2f}M")
        print(f"  Current Revenue: ${result.current_revenue/1e6:.2f}M")
        print(f"  Optimal Revenue: ${result.optimal_revenue/1e6:.2f}M")
        print(f"  Lift: +{result.revenue_lift_pct:.1%}")
        print(f"  Converged: {result.converged}")
        print(f"{'='*60}")
        return df


if __name__ == "__main__":
    from .data_generator import generate_data, CHANNELS
    from .model import MarketingMixModel
    import numpy as np

    df_sales, df_spend, _ = generate_data(seed=42)
    y = df_sales["sales"]
    df_controls = pd.DataFrame({
        "trend": np.arange(len(y)),
        "sin52": np.sin(2 * np.pi * np.arange(len(y)) / 52),
        "cos52": np.cos(2 * np.pi * np.arange(len(y)) / 52),
    }, index=df_spend.index)

    model = MarketingMixModel(channels=CHANNELS, control_cols=["trend","sin52","cos52"])
    model.fit(y, df_spend, df_controls)

    opt = BudgetOptimizer(model, n_restarts=10)
    total_budget = df_spend.mean().sum() * 1000  # annualised
    result = opt.optimise(total_budget, df_spend)
    table = opt.summary_table(result)
    print(table)
