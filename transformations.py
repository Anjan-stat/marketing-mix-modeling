"""
transformations.py
------------------
Adstock (carryover) and saturation functions.

Statistical Background
----------------------
Adstock:    a_t = x_t + λ * a_{t-1}        (geometric decay IIR filter)
            Extended with a peak-lag θ for delayed response.

Saturation: Hill   — s(a) = a^n / (K^n + a^n)
            Logistic — s(a) = L / (1 + exp(-k(a - x0)))

Both transformations are differentiable, enabling gradient-based
optimisation of their hyper-parameters.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── Adstock Transformations ────────────────────────────────────────────────────

def geometric_adstock(
    x: np.ndarray,
    lam: float,
    normalise: bool = False,
) -> np.ndarray:
    """
    Geometric (Koyck) adstock with instantaneous peak.

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw media spend time series.
    lam : float in [0, 1)
        Carryover / decay rate. Higher → longer memory.
    normalise : bool
        If True, divide by (1 / (1 - lam)) so the steady-state
        gain equals the raw spend.

    Returns
    -------
    a : np.ndarray, shape (T,)
        Adstocked series.

    Notes
    -----
    Impulse response: h[k] = λ^k  (for k = 0, 1, 2, ...)
    Cumulative impulse response weight sum: 1 / (1 - λ)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    a = np.zeros(T)
    a[0] = x[0]
    for t in range(1, T):
        a[t] = x[t] + lam * a[t - 1]
    if normalise:
        a /= (1.0 / (1.0 - lam))
    return a


def delayed_adstock(
    x: np.ndarray,
    lam: float,
    theta: int = 0,
    L: int = 13,
) -> np.ndarray:
    """
    Delayed adstock where the peak response occurs at lag θ.

    The weight for lag k is proportional to λ^{(k - θ)^2},
    i.e. a Gaussian kernel centred at θ, applied to the spend history.

    Parameters
    ----------
    x : array-like, shape (T,)
    lam : float in (0, 1)
        Controls the spread of weights around the peak.
    theta : int >= 0
        Lag at which the response peaks.
    L : int
        Maximum lag to consider (truncation horizon).

    Returns
    -------
    a : np.ndarray, shape (T,)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    # Build normalised weight vector
    lags = np.arange(L + 1)
    weights = lam ** ((lags - theta) ** 2)
    weights /= weights.sum()
    # Convolve (causal: only past values)
    a = np.zeros(T)
    for t in range(T):
        for k, w in enumerate(weights):
            if t - k >= 0:
                a[t] += w * x[t - k]
    return a


def adstock_matrix(T: int, lam: float) -> np.ndarray:
    """
    Return the T×T lower-triangular Toeplitz adstock matrix L_λ
    such that a = L_λ @ x.

    Useful for analytical derivations and vectorised computation.
    """
    L = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            L[i, j] = lam ** (i - j)
    return L


# ── Saturation Transformations ─────────────────────────────────────────────────

def hill_saturation(
    a: np.ndarray,
    K: float,
    n: float,
) -> np.ndarray:
    """
    Hill (Michaelis–Menten) saturation function.

    s(a) = a^n / (K^n + a^n)

    Properties
    ----------
    - s(0) = 0, lim_{a→∞} s(a) = 1  (bounded in [0, 1])
    - s(K) = 0.5  ← K is the half-saturation point
    - n > 1 : sigmoidal (S-curve); n = 1 : concave; n < 1 : very concave

    Parameters
    ----------
    a : array-like
        Adstocked spend (any non-negative values).
    K : float > 0
        Half-saturation constant.
    n : float > 0
        Hill coefficient (shape).
    """
    a = np.asarray(a, dtype=float)
    Kn = K ** n
    return a**n / (Kn + a**n)


def logistic_saturation(
    a: np.ndarray,
    L: float = 1.0,
    k: float = 1.0,
    x0: float = 0.5,
) -> np.ndarray:
    """
    Logistic (sigmoid) saturation function.

    s(a) = L / (1 + exp(-k * (a - x0)))

    Parameters
    ----------
    a : array-like
    L : float
        Carrying capacity (upper asymptote).
    k : float
        Growth rate / steepness.
    x0 : float
        Inflection point (spend at maximum marginal return).
    """
    a = np.asarray(a, dtype=float)
    return L / (1.0 + np.exp(-k * (a - x0)))


def power_saturation(a: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple power (Cobb-Douglas) saturation: s(a) = a^α, α ∈ (0, 1].

    Concave everywhere; no upper asymptote. Often used as a quick proxy.
    """
    a = np.asarray(a, dtype=float)
    return a ** alpha


# ── Combined pipeline ──────────────────────────────────────────────────────────

def transform_spend(
    x: np.ndarray,
    lam: float,
    K: float,
    n: float,
    theta: int = 0,
    use_delayed: bool = False,
    normalise_adstock: bool = True,
) -> np.ndarray:
    """
    Full transformation pipeline: adstock → Hill saturation.

    Parameters
    ----------
    x : raw spend, shape (T,)
    lam : adstock decay rate
    K : Hill half-saturation
    n : Hill shape
    theta : peak lag (used only if use_delayed=True)
    use_delayed : use delayed vs geometric adstock
    normalise_adstock : normalise adstock output to [0, 1]

    Returns
    -------
    x_star : transformed spend, shape (T,), values in [0, 1]
    """
    # Step 1 — Adstock
    if use_delayed and theta > 0:
        a = delayed_adstock(x, lam=lam, theta=theta)
    else:
        a = geometric_adstock(x, lam=lam)

    # Step 2 — Normalise adstocked series to [0, 1] for stable saturation params
    if normalise_adstock and a.max() > 0:
        a = a / a.max()

    # Step 3 — Saturation
    x_star = hill_saturation(a, K=K, n=n)
    return x_star


# ── Response curve helpers ─────────────────────────────────────────────────────

def marginal_roi(
    a: np.ndarray,
    K: float,
    n: float,
    beta: float,
) -> np.ndarray:
    """
    Marginal ROI = d/da [beta * hill(a)] = beta * d(hill)/da

    d(hill)/da = n * K^n * a^{n-1} / (K^n + a^n)^2
    """
    a = np.asarray(a, dtype=float)
    Kn = K ** n
    an = a ** n
    return beta * n * Kn * a ** (n - 1) / (Kn + an) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Demonstrate adstock decay for different λ values
    T = 52
    impulse = np.zeros(T)
    impulse[4] = 1.0  # single spend spike at week 4

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Adstock impulse response
    ax = axes[0]
    for lam in [0.1, 0.4, 0.7, 0.9]:
        ax.plot(geometric_adstock(impulse, lam), label=f"λ={lam}")
    ax.set_title("Adstock: Impulse Response for Different λ")
    ax.set_xlabel("Week")
    ax.set_ylabel("Adstocked Value")
    ax.legend()
    ax.grid(alpha=0.3)

    # Hill saturation curves
    ax = axes[1]
    a_range = np.linspace(0, 1, 200)
    for K, n, label in [(0.2, 1.0, "K=0.2, n=1"), (0.5, 1.5, "K=0.5, n=1.5"),
                         (0.5, 0.8, "K=0.5, n=0.8"), (0.8, 2.0, "K=0.8, n=2")]:
        ax.plot(a_range, hill_saturation(a_range, K, n), label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="50% saturation")
    ax.set_title("Hill Saturation Curves")
    ax.set_xlabel("Normalised Adstocked Spend")
    ax.set_ylabel("Saturation (fraction of max effect)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/transformations_demo.png", dpi=150)
    print("Saved outputs/transformations_demo.png")
