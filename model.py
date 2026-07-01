"""
model.py
--------
Frequentist Marketing Mix Model.

Estimation strategy
-------------------
1. Outer loop  — optimise adstock (λ) and saturation (K, n) hyper-parameters
                 via L-BFGS-B on held-out MAPE (or grid search).
2. Inner loop  — given transformed features X*, fit a Ridge regression with
                 non-negativity constraint on media coefficients.

The non-negativity constraint on β ensures economic interpretability:
media spend cannot have a negative effect on sales in expectation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .transformations import transform_spend, geometric_adstock, hill_saturation

warnings.filterwarnings("ignore")


# ── Parameter containers ───────────────────────────────────────────────────────

@dataclass
class ChannelParams:
    """Fitted transformation parameters for one channel."""
    name: str
    lam: float        # adstock decay
    K: float          # Hill half-saturation
    n: float          # Hill shape
    beta: float       # regression coefficient


@dataclass
class MMFitResult:
    """Container for a fitted model."""
    channel_params: List[ChannelParams]
    intercept: float
    control_coefs: Dict[str, float]
    r2_train: float
    r2_test: float
    mape_test: float
    X_transformed: pd.DataFrame   # transformed media features


# ── Model class ────────────────────────────────────────────────────────────────

class MarketingMixModel:
    """
    Frequentist MMM: optimise adstock + saturation hyper-parameters,
    then fit Ridge regression with non-negative media coefficients.

    Parameters
    ----------
    channels : list of channel names (columns in X_spend)
    control_cols : list of control variable columns in X_controls
    ridge_alpha : Ridge regularisation strength
    test_size : fraction of weeks held out for evaluation
    search : 'grid' or 'lbfgs'  (hyper-param search method)
    seed : random seed
    """

    PARAM_BOUNDS = {
        "lam": (0.01, 0.95),
        "K":   (0.05, 0.95),
        "n":   (0.3,  3.0),
    }

    def __init__(
        self,
        channels: List[str],
        control_cols: Optional[List[str]] = None,
        ridge_alpha: float = 1.0,
        test_size: float = 0.2,
        search: str = "lbfgs",
        seed: int = 42,
    ):
        self.channels = channels
        self.control_cols = control_cols or []
        self.ridge_alpha = ridge_alpha
        self.test_size = test_size
        self.search = search
        self.seed = seed
        self.result_: Optional[MMFitResult] = None
        self._scaler = StandardScaler()

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(
        self,
        y: pd.Series,
        X_spend: pd.DataFrame,
        X_controls: Optional[pd.DataFrame] = None,
    ) -> "MarketingMixModel":
        """
        Fit the MMM.

        Parameters
        ----------
        y : target sales series (index = dates)
        X_spend : media spend DataFrame, columns = channels
        X_controls : optional control variable DataFrame (trend, seasonality…)
        """
        self._y = y.values.astype(float)
        self._X_spend = {ch: X_spend[ch].values.astype(float) for ch in self.channels}
        self._X_controls = X_controls

        # Normalise spend per channel to [0, 1]
        self._spend_max = {ch: v.max() for ch, v in self._X_spend.items()}
        self._X_spend_norm = {
            ch: v / self._spend_max[ch] for ch, v in self._X_spend.items()
        }

        # Train / test split (time-based)
        T = len(self._y)
        self._n_test = int(T * self.test_size)
        self._n_train = T - self._n_test

        # Optimise hyper-parameters
        print(f"Optimising transformation hyper-parameters ({self.search})…")
        best_params = self._optimise_hyperparams()

        # Final fit on all data
        X_star, channel_params = self._build_features(best_params)
        fit_result = self._fit_ridge(X_star, channel_params)
        self.result_ = fit_result
        print(f"\nFit complete  R²_train={fit_result.r2_train:.3f}  "
              f"R²_test={fit_result.r2_test:.3f}  "
              f"MAPE_test={fit_result.mape_test:.2%}")
        return self

    def predict(self, X_star: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return fitted (in-sample) predictions."""
        self._check_fitted()
        if X_star is None:
            X_star = self.result_.X_transformed
        media_cols = [c for c in X_star.columns if c in self.channels]
        ctrl_cols = [c for c in X_star.columns if c not in self.channels]
        X = X_star[media_cols + ctrl_cols].values
        return self._ridge.predict(X)

    @property
    def channel_params(self) -> List[ChannelParams]:
        self._check_fitted()
        return self.result_.channel_params

    # ── Hyper-parameter optimisation ───────────────────────────────────────────

    def _optimise_hyperparams(self) -> Dict[str, Dict[str, float]]:
        """Choose between grid search and L-BFGS-B."""
        if self.search == "grid":
            return self._grid_search()
        return self._lbfgs_search()

    def _lbfgs_search(self) -> Dict[str, Dict[str, float]]:
        """
        Joint L-BFGS-B optimisation over all channel hyper-parameters.
        Minimises held-out MAPE.
        """
        n_ch = len(self.channels)
        # Initial guess: midpoint of bounds
        x0 = []
        bounds = []
        for ch in self.channels:
            x0 += [0.5, 0.4, 1.2]   # lam, K, n
            bounds += [
                self.PARAM_BOUNDS["lam"],
                self.PARAM_BOUNDS["K"],
                self.PARAM_BOUNDS["n"],
            ]

        best_loss = np.inf
        best_params = None

        def objective(x):
            params = self._vec_to_params(x)
            return self._eval_params(params)

        # Multiple restarts
        rng = np.random.default_rng(self.seed)
        for restart in range(5):
            x0_r = rng.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds],
            )
            res = minimize(
                objective,
                x0_r,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = self._vec_to_params(res.x)
            print(f"  Restart {restart+1}/5  loss={res.fun:.4f}")

        return best_params

    def _grid_search(self) -> Dict[str, Dict[str, float]]:
        """Per-channel grid search (faster but less accurate)."""
        best_params = {}
        lam_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        K_grid   = [0.2, 0.4, 0.6, 0.8]
        n_grid   = [0.5, 1.0, 1.5, 2.0]

        for ch in self.channels:
            best_loss, best_p = np.inf, (0.5, 0.4, 1.0)
            for lam, K, n in product(lam_grid, K_grid, n_grid):
                x_star = self._transform_channel(
                    self._X_spend_norm[ch], lam, K, n
                )
                # Simple univariate OLS on train, MAPE on test
                y_tr = self._y[:self._n_train]
                x_tr = x_star[:self._n_train].reshape(-1, 1)
                y_te = self._y[self._n_train:]
                x_te = x_star[self._n_train:].reshape(-1, 1)
                from numpy.linalg import lstsq
                b, _, _, _ = lstsq(x_tr, y_tr, rcond=None)
                pred = x_te @ b
                mape = np.mean(np.abs((y_te - pred) / y_te))
                if mape < best_loss:
                    best_loss, best_p = mape, (lam, K, n)
            best_params[ch] = {"lam": best_p[0], "K": best_p[1], "n": best_p[2]}
        return best_params

    def _vec_to_params(self, x: np.ndarray) -> Dict[str, Dict[str, float]]:
        params = {}
        for i, ch in enumerate(self.channels):
            params[ch] = {
                "lam": float(x[3*i]),
                "K":   float(x[3*i + 1]),
                "n":   float(x[3*i + 2]),
            }
        return params

    def _eval_params(self, params: Dict[str, Dict[str, float]]) -> float:
        """Evaluate MAPE on the held-out test set for given hyper-params."""
        X_star, _ = self._build_features(params)
        X_tr = X_star.iloc[:self._n_train].values
        X_te = X_star.iloc[self._n_train:].values
        y_tr = self._y[:self._n_train]
        y_te = self._y[self._n_train:]
        ridge = Ridge(alpha=self.ridge_alpha, positive=True, fit_intercept=True)
        ridge.fit(X_tr, y_tr)
        pred = ridge.predict(X_te)
        mape = np.mean(np.abs((y_te - pred) / y_te))
        return mape

    # ── Feature building ───────────────────────────────────────────────────────

    def _build_features(
        self, params: Dict[str, Dict[str, float]]
    ) -> Tuple[pd.DataFrame, List[ChannelParams]]:
        """Transform raw spend into model features X*."""
        T = len(self._y)
        cols = {}
        for ch in self.channels:
            p = params[ch]
            cols[ch] = self._transform_channel(
                self._X_spend_norm[ch], p["lam"], p["K"], p["n"]
            )
        # Add control variables
        if self._X_controls is not None:
            for col in self.control_cols:
                if col in self._X_controls.columns:
                    vals = self._X_controls[col].values.astype(float)
                    # Standardise controls
                    cols[col] = (vals - vals.mean()) / (vals.std() + 1e-8)

        X_star = pd.DataFrame(cols)

        # Build placeholder ChannelParams (betas assigned after Ridge fit)
        ch_params = [
            ChannelParams(ch, params[ch]["lam"], params[ch]["K"], params[ch]["n"], 0.0)
            for ch in self.channels
        ]
        return X_star, ch_params

    @staticmethod
    def _transform_channel(
        x_norm: np.ndarray, lam: float, K: float, n: float
    ) -> np.ndarray:
        a = geometric_adstock(x_norm, lam=lam)
        if a.max() > 0:
            a /= a.max()
        return hill_saturation(a, K=K, n=n)

    # ── Ridge regression ───────────────────────────────────────────────────────

    def _fit_ridge(
        self,
        X_star: pd.DataFrame,
        ch_params: List[ChannelParams],
    ) -> MMFitResult:
        """Fit Ridge with positive=True on full data; evaluate on test split."""
        X = X_star.values
        y = self._y

        ridge = Ridge(alpha=self.ridge_alpha, positive=True, fit_intercept=True)
        ridge.fit(X, y)
        self._ridge = ridge

        y_pred_all = ridge.predict(X)
        y_pred_test = ridge.predict(X[self._n_train:])
        y_test = y[self._n_train:]

        r2_train = r2_score(y[:self._n_train], ridge.predict(X[:self._n_train]))
        r2_test  = r2_score(y_test, y_pred_test)
        mape_test = float(np.mean(np.abs((y_test - y_pred_test) / y_test)))

        # Assign fitted betas to channel params
        coef = ridge.coef_
        for i, cp in enumerate(ch_params):
            cp.beta = float(coef[i])

        ctrl_coefs = {}
        for j, col in enumerate(self.control_cols):
            ctrl_coefs[col] = float(coef[len(self.channels) + j])

        return MMFitResult(
            channel_params=ch_params,
            intercept=float(ridge.intercept_),
            control_coefs=ctrl_coefs,
            r2_train=r2_train,
            r2_test=r2_test,
            mape_test=mape_test,
            X_transformed=X_star,
        )

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of fitted parameters."""
        self._check_fitted()
        rows = []
        for cp in self.result_.channel_params:
            rows.append({
                "channel": cp.name,
                "beta": round(cp.beta, 2),
                "lambda (adstock)": round(cp.lam, 3),
                "K (half-sat)": round(cp.K, 3),
                "n (shape)": round(cp.n, 3),
            })
        return pd.DataFrame(rows).set_index("channel")

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")


if __name__ == "__main__":
    from .data_generator import generate_data, CHANNELS

    df_sales, df_spend, _ = generate_data(seed=42)
    y = df_sales["sales"]
    # Add simple controls
    df_controls = pd.DataFrame({
        "trend": np.arange(len(y)),
        "sin52": np.sin(2 * np.pi * np.arange(len(y)) / 52),
        "cos52": np.cos(2 * np.pi * np.arange(len(y)) / 52),
    }, index=df_spend.index)

    model = MarketingMixModel(
        channels=CHANNELS,
        control_cols=["trend", "sin52", "cos52"],
        ridge_alpha=0.5,
        search="lbfgs",
    )
    model.fit(y, df_spend, df_controls)
    print("\nParameter Summary:")
    print(model.summary())
