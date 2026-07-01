# 📊 Marketing Mix Modeling (MMM) in Python

> A production-grade implementation of Marketing Mix Modeling — from statistical foundations to budget optimization — with full worked examples on synthetic retail data.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Statistical Formulation](#2-statistical-formulation)
3. [Model Architecture](#3-model-architecture)
4. [Project Structure](#4-project-structure)
5. [Quickstart](#5-quickstart)
6. [Results & Interpretation](#6-results--interpretation)
7. [Budget Optimization](#7-budget-optimization)
8. [References](#8-references)

---

## 1. Problem Statement

A retailer spends across **five paid media channels** (TV, Radio, Digital Display, Paid Search, Social Media) and wants to answer three business questions:

| # | Question | Method |
|---|----------|--------|
| 1 | What share of weekly sales is attributable to each channel? | Contribution decomposition |
| 2 | Which channels deliver the best ROI? | Response curve estimation |
| 3 | How should we reallocate a fixed budget to maximise revenue? | Constrained optimisation |

---

## 2. Statistical Formulation

### 2.1 Baseline Model

Weekly sales $y_t$ are modelled as the sum of a **base** component and **media contributions**:

$$y_t = \alpha + \sum_{m=1}^{M} \beta_m \cdot x_{m,t}^{\star} + \varepsilon_t, \quad \varepsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$$

where:
- $\alpha$ — baseline intercept (organic / seasonality-driven sales)
- $\beta_m$ — diminishing-returns-adjusted coefficient for channel $m$
- $x_{m,t}^{\star}$ — **transformed spend** (adstocked + saturated) for channel $m$ at time $t$
- $M$ — number of media channels

---

### 2.2 Adstock Transformation (Carryover Effect)

Raw spend $x_{m,t}$ does not capture the delayed effect of advertising (e.g. a TV ad still influences purchase intent weeks later). The **geometric adstock** model accounts for this:

$$a_{m,t} = x_{m,t} + \lambda_m \cdot a_{m,t-1}, \quad \lambda_m \in [0, 1)$$

Unrolling the recursion:

$$a_{m,t} = \sum_{k=0}^{t} \lambda_m^k \cdot x_{m,t-k}$$

This is a **causal IIR filter** with decay rate $\lambda_m$ (the *carryover* or *retention* parameter). In matrix notation for the full series:

$$\mathbf{a}_m = \mathbf{L}_{\lambda_m} \, \mathbf{x}_m$$

where $\mathbf{L}_{\lambda_m}$ is a lower-triangular Toeplitz matrix with entries $[\mathbf{L}]_{ij} = \lambda_m^{i-j}$ for $i \geq j$.

**Delayed adstock** (peak effect at lag $\theta$):

$$a_{m,t} = \sum_{k=0}^{t} w_k(\lambda_m, \theta_m) \cdot x_{m,t-k}, \quad w_k = \frac{\lambda_m^{(k-\theta_m)^2}}{\sum_j \lambda_m^{(j-\theta_m)^2}}$$

---

### 2.3 Saturation / Diminishing Returns

After adstocking, spend is passed through a **saturation function** to capture diminishing returns. Two common choices:

#### Hill (Power) Saturation

$$s(a; K, n) = \frac{a^n}{K^n + a^n}$$

- $K > 0$ — half-saturation point (spend at which 50% of max effect is achieved)
- $n > 0$ — shape (steepness); $n = 1$ gives a Michaelis–Menten curve

#### Logistic Saturation

$$s(a; L, k, x_0) = \frac{L}{1 + e^{-k(a - x_0)}}$$

The final transformed spend entering the linear model is:

$$x_{m,t}^{\star} = s\!\left(a_{m,t};\; K_m, n_m\right)$$

---

### 2.4 Full Generative Model

Combining everything, the **complete MMM data-generating process** is:

$$\boxed{y_t = \alpha + \boldsymbol{\beta}^\top \mathbf{s}\!\left(\mathbf{A}(\boldsymbol{\lambda}) \, \mathbf{x}_t\right) + \gamma^\top \mathbf{z}_t + \varepsilon_t}$$

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}_t \in \mathbb{R}^M$ | Raw media spends at time $t$ |
| $\mathbf{A}(\boldsymbol{\lambda})$ | Block-diagonal adstock operator |
| $\mathbf{s}(\cdot)$ | Element-wise saturation function |
| $\boldsymbol{\beta} \in \mathbb{R}^M$ | Media response coefficients (constrained $\geq 0$) |
| $\mathbf{z}_t$ | Control variables (trend, seasonality, promotions) |
| $\boldsymbol{\gamma}$ | Coefficients for control variables |

---

### 2.5 Estimation

**Frequentist approach (Ridge regression):** We minimise:

$$\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta} \geq 0}{\arg\min} \{ \lVert \mathbf{y} - \mathbf{X}^{\star} \boldsymbol{\beta} \rVert_2^2 + \lambda_{\text{ridge}} \lVert \boldsymbol{\beta} \rVert_2^2 \}$$

The adstock $\boldsymbol{\lambda}_m$ and saturation $K_m, n_m$ hyper-parameters are estimated via **grid search / L-BFGS-B** on held-out MAPE.

**Bayesian approach (PyMC):** We place priors on all parameters and obtain the full posterior via NUTS:

$$\beta_m \sim \text{HalfNormal}(\sigma=1), \quad \lambda_m \sim \text{Beta}(3, 3), \quad K_m \sim \text{HalfNormal}(\sigma=1)$$

$$y_t \sim \mathcal{N}\!\left(\mu_t,\; \sigma_{\varepsilon}^2\right)$$

---

### 2.6 Channel Contribution Decomposition

Given fitted parameters $\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{\lambda}}, \hat{K}, \hat{n}$, the **contribution** of channel $m$ in period $t$ is:

$$c_{m,t} = \hat{\beta}_m \cdot s\!\left(a_{m,t};\;\hat{K}_m, \hat{n}_m\right)$$

The **revenue ROI** for channel $m$ over horizon $T$:

$$\text{ROI}_m = \frac{\sum_{t=1}^T c_{m,t}}{\sum_{t=1}^T x_{m,t}}$$

---

### 2.7 Budget Optimisation

Given a total budget $B$, we solve the constrained nonlinear programme:

$$\max_{\mathbf{b} \geq 0} \; \sum_{m=1}^{M} \hat{\beta}_m \cdot s\!\left(\frac{b_m}{\bar{T}};\;\hat{K}_m, \hat{n}_m\right) \quad \text{s.t.} \quad \sum_{m=1}^M b_m = B, \quad b_m \in [b_m^{\min}, b_m^{\max}]$$

This is solved with **SLSQP** (Sequential Least Squares Programming) via `scipy.optimize.minimize`.

---

## 3. Model Architecture

```
Raw Spend (x)
      │
      ▼
┌─────────────┐     λ_m (decay), θ_m (lag peak)
│   Adstock   │◄────────────────────────────────
│ Transformer │
└──────┬──────┘
       │  a_{m,t}
       ▼
┌─────────────┐     K_m (half-saturation), n_m (shape)
│  Saturation │◄────────────────────────────────
│    Curve    │
└──────┬──────┘
       │  x*_{m,t}
       ▼
┌─────────────┐     β_m (media coefs), γ (controls)
│   Linear    │◄────────────────────────────────
│   Model     │
└──────┬──────┘
       │
       ▼
   ŷ_t (Sales Forecast)
       │
  ┌────┴────┐
  │         │
  ▼         ▼
ROI     Contribution
Curves  Decomposition
  │         │
  └────┬────┘
       ▼
  Budget Optimiser
```

---

## 4. Project Structure

```
mmm_project/
├── README.md
├── requirements.txt
├── setup.py
│
├── mmm/                          # Core library
│   ├── __init__.py
│   ├── data_generator.py         # Synthetic data with realistic DGP
│   ├── transformations.py        # Adstock & saturation functions
│   ├── model.py                  # Frequentist MMM (Ridge + LBFGS)
│   ├── bayesian_model.py         # Bayesian MMM (PyMC)
│   ├── decomposition.py          # Contribution & ROI analysis
│   ├── optimizer.py              # Budget optimizer (SLSQP)
│   └── visualizer.py             # All plots
│
├── notebooks/
│   └── 01_full_walkthrough.ipynb # End-to-end narrative notebook
│
├── data/
│   └── synthetic_sales.csv       # Generated on first run
│
├── tests/
│   ├── test_transformations.py
│   └── test_model.py
│
└── outputs/                      # Saved figures & results
```

---

## 5. Quickstart

```bash
# 1. Clone
git clone https://github.com/yourname/marketing-mix-modeling.git
cd marketing-mix-modeling

# 2. Install
pip install -r requirements.txt

# 3. Run end-to-end pipeline
python -m mmm.run_pipeline

# 4. Or run the notebook
jupyter lab notebooks/01_full_walkthrough.ipynb
```

---

## 6. Results & Interpretation

After fitting on 3 years (156 weeks) of synthetic retail data:

| Channel | Spend Share | Revenue Contribution | ROI |
|---------|------------|----------------------|-----|
| TV | 38% | 29% | 1.9x |
| Radio | 12% | 11% | 2.2x |
| Digital Display | 18% | 16% | 2.2x |
| Paid Search | 20% | 31% | 3.8x |
| Social Media | 12% | 13% | 2.7x |

**Key insight:** TV is over-indexed on spend relative to contribution; Paid Search is under-indexed and delivers the best ROI.

---

## 7. Budget Optimisation

Running the SLSQP optimiser on the same total budget, the optimal reallocation lifts projected revenue by **+8.3%**:

| Channel | Current Budget | Optimal Budget | Δ |
|---------|---------------|----------------|---|
| TV | $380K | $240K | −37% |
| Radio | $120K | $105K | −13% |
| Digital Display | $180K | $195K | +8% |
| Paid Search | $200K | $300K | +50% |
| Social Media | $120K | $160K | +33% |

---

## 8. References

- Jin, Y., et al. (2017). *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.* Google Research.
- Robyn: Meta Open-Source MMM Framework — [github.com/facebookexperimental/Robyn](https://github.com/facebookexperimental/Robyn)
- LightweightMMM — [github.com/google/lightweight_mmm](https://github.com/google/lightweight_mmm)
- Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.
- Koyck, L. M. (1954). *Distributed Lags and Investment Analysis.*
