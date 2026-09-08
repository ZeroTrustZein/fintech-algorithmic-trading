# System Architecture & Quantitative Mathematics

This document provides a comprehensive technical overview of the architecture, algorithmic formulations, and design decisions behind **`fintech-algorithmic-trading`**.

---

## 1. High-Level System Architecture

`fintech-algorithmic-trading` is designed as a modular, decoupled quantitative trading risk management and backtesting platform. It isolates numerical computation, market simulation, execution logic, real-time risk surveillance, and reporting into dedicated subsystems.

```
                   ┌──────────────────────────────────────────┐
                   │            Market Data Feed              │
                   │  - Streaming OHLCV Candlestick Bars      │
                   │  - Synthetic Level 1 & Level 2 Depth     │
                   │  - Multi-Timeframe Resampling            │
                   └────────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       ┌─────────────────────────┐             ┌─────────────────────────┐
       │ Quantitative Strategies │             │ Core Risk Analytics     │
       │ - TrendFollowingMACD    │             │ - Multi-Method VaR/CVaR │
       │ - BollingerReversion    │             │ - Monte Carlo (GBM/Jump)│
       │ - VolatilityBreakout    │             │ - Extreme Value Theory  │
       │ - StatArbPairsTrading   │             │ - Black-Scholes Greeks  │
       │ - CompositeAggregator   │             │ - Delta-Gamma Hedging   │
       └────────────┬────────────┘             │ - Portfolio Optimizer   │
                    │                          │ - Kelly Position Sizer  │
                    ▼ Signals                  └────────────┬────────────┘
       ┌─────────────────────────┐                          │
       │ Pre-Trade Risk Gate     │                          │
       │ - Notional Ceilings     │                          │
       │ - Gross Leverage Limits │                          │
       │ - Position & Sector Cap │                          │
       │ - Cash Sufficiency      │                          │
       └────────────┬────────────┘                          │
                    │ Approved Orders                       │
                    ▼                                       │
       ┌─────────────────────────┐                          │
       │ Order Management System │                          │
       │ - Slippage Simulator    │                          │
       │ - Execution Fills       │                          │
       │ - Position Lifecycle    │                          │
       └────────────┬────────────┘                          │
                    │ Portfolio State                       │
                    ▼                                       │
       ┌─────────────────────────┐                          │
       │ Real-Time Surveillance  │                          │
       │ - Circuit Breakers      │                          │
       │ - Drawdown Guardrails   │                          │
       │ - Margin Alert Router   │                          │
       │ - Emergency Kill-Switch │                          │
       └────────────┬────────────┘                          │
                    │                                       │
                    ▼                                       ▼
       ┌────────────────────────────────────────────────────────┐
       │               Institutional Reporting                  │
       │  - Markdown Risk Tearsheets & Memos                    │
       │  - JSON / CSV Performance & Equity Curve Exporters     │
       │  - Rich Interactive Command-Line Interface (CLI)       │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Core Quantitative Mathematics & Risk Formulations

### 2.1 Value-at-Risk (VaR) & Expected Shortfall (CVaR)

Value-at-Risk defines the maximum anticipated loss over holding horizon $h$ at confidence level $\alpha$ (e.g., $95\%$ or $99\%$). Let $R$ denote the portfolio return, and $L = -R$ represent portfolio loss.

#### Historical Simulation
Historical VaR computes the empirical quantile directly from the empirical loss distribution:
$$\text{VaR}_{\alpha}(1\text{d}) = \text{Quantile}_{\alpha}(\{L_t\}_{t=1}^T)$$
$$\text{VaR}_{\alpha}(h) = \text{VaR}_{\alpha}(1\text{d}) \times \sqrt{h}$$

Expected Shortfall (Conditional VaR), measuring tail loss beyond the VaR cutoff:
$$\text{CVaR}_{\alpha}(1\text{d}) = \mathbb{E}[L \mid L \ge \text{VaR}_{\alpha}(1\text{d})] = \frac{1}{|T_{\text{tail}}|} \sum_{L_i \in T_{\text{tail}}} L_i$$

#### Parametric Gaussian Model
Assuming returns follow $R \sim \mathcal{N}(\mu, \sigma^2)$:
$$\text{VaR}_{\alpha}(1\text{d}) = \max(0, z_{\alpha} \sigma - \mu)$$
$$\text{CVaR}_{\alpha}(1\text{d}) = \max\left(\text{VaR}_{\alpha}(1\text{d}), -\mu + \sigma \frac{\phi(z_{\alpha})}{1 - \alpha}\right)$$
where $z_{\alpha} = \Phi^{-1}(\alpha)$ is the standard normal inverse cumulative distribution function, and $\phi(\cdot)$ is the standard normal probability density function.

For a multi-asset portfolio with asset weight vector $\mathbf{w}$ and covariance matrix $\mathbf{\Sigma}$:
$$\sigma_{\text{portfolio}} = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$$
$$\text{VaR}_{\alpha}(h) = z_{\alpha} \sigma_{\text{portfolio}} \sqrt{h} \times \text{NAV}$$

#### Cornish-Fisher Expansion
Standard normal distributions understate financial market tail risk due to negative skewness and excess kurtosis. The Cornish-Fisher expansion computes an adjusted quantile $z_{\text{cf}}$ using the skewness $S$ and excess kurtosis $K$:
$$z_{\text{cf}} = z + \frac{z^2 - 1}{6} S + \frac{z^3 - 3z}{24} K - \frac{2z^3 - 5z}{36} S^2$$
$$\text{VaR}_{\alpha}(1\text{d}) = \max(0, z_{\text{cf}} \sigma - \mu)$$

---

### 2.2 Extreme Value Theory (EVT) Peaks-Over-Threshold (POT)

By the Pickands-Balkema-de Haan theorem, conditional exceedances over a sufficiently high threshold $u$ converge asymptotically to the Generalized Pareto Distribution (GPD):
$$G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi} \quad (\xi \ne 0)$$
where:
- $u$: High threshold quantile (e.g. 90th or 95th percentile of historical losses).
- $y = L - u > 0$: Magnitude of loss exceedance.
- $\xi$: Shape parameter (tail index; $\xi > 0$ indicates heavy/fat tails).
- $\beta$: Scale parameter.

Parameters $(\xi, \beta)$ are estimated via maximum likelihood estimation (MLE). The EVT quantile formula for VaR at confidence level $\alpha$ is:
$$\text{VaR}_{\alpha} = u + \frac{\beta}{\xi} \left[ \left( \frac{N}{N_u} (1 - \alpha) \right)^{-\xi} - 1 \right]$$
where $N$ is the total number of observations and $N_u$ is the number of exceedances above $u$.

Expected Shortfall under GPD:
$$\text{CVaR}_{\alpha} = \frac{\text{VaR}_{\alpha} + \beta - \xi u}{1 - \xi}$$

---

### 2.3 Stochastic Monte Carlo Path Simulation

The simulation engine models the temporal evolution of correlated multi-asset portfolios across time steps $\Delta t = \frac{h / 252}{N_{\text{steps}}}$ using Cholesky factor decomposition:
$$\mathbf{\Sigma}_{\text{corr}} = \mathbf{L} \mathbf{L}^T$$
$$\mathbf{Z}_{\text{corr}} = \mathbf{Z}_{\text{uncorr}} \mathbf{L}^T$$

#### 1. Correlated Geometric Brownian Motion (GBM)
$$S_{i}(t + \Delta t) = S_i(t) \exp\left( \left(\mu_i - \frac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i \sqrt{\Delta t} Z_{\text{corr}, i} \right)$$
where drift $\mu_i = r_f + \beta_i \times \text{ERP}$ combines the risk-free rate $r_f$ with the asset beta $\beta_i$ and equity risk premium ($\text{ERP} = 5\%$).

#### 2. Merton Jump-Diffusion Model
Captures sudden discontinuous price shocks and market gaps:
$$S_{i}(t + \Delta t) = S_i(t) \exp\left( \left(\mu_i - \frac{1}{2}\sigma_i^2 - \lambda k\right)\Delta t + \sigma_i \sqrt{\Delta t} Z_{\text{corr}, i} + \sum_{j=1}^{N_J} Y_j \right)$$
where:
- $N_J \sim \text{Poisson}(\lambda \Delta t)$: Random jump arrivals.
- $Y_j \sim \mathcal{N}(\mu_J, \sigma_J^2)$: Log-jump magnitude.
- $k = \exp\left(\mu_J + \frac{1}{2}\sigma_J^2\right) - 1$: Martingale compensator.

#### 3. Ornstein-Uhlenbeck Mean-Reverting Drift
$$S_i(t + \Delta t) = S_i(t) \exp\left( \kappa_i \frac{S_{i}(0) - S_i(t)}{S_i(t)}\Delta t - \frac{1}{2}\sigma_i^2 \Delta t + \sigma_i \sqrt{\Delta t} Z_{\text{corr}, i} \right)$$
where $\kappa$ is the mean-reversion speed pulling the price toward equilibrium $S_i(0)$.

#### Pathwise Drawdown Tracking
Along each simulation trajectory $m \in \{1, \dots, M\}$:
$$\text{Peak}_m(t) = \max_{\tau \le t} V_m(\tau)$$
$$\text{Drawdown}_m(t) = \frac{\text{Peak}_m(t) - V_m(t)}{\text{Peak}_m(t)}$$
$$\text{MaxDrawdown}_m = \max_{t \in [0, T]} \text{Drawdown}_m(t)$$

---

### 2.4 Analytical Option Greeks & Delta-Gamma Hedging

European option pricing with continuous dividend yield $q$ under Black-Scholes:
$$d_1 = \frac{\ln(S / K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$

#### Sensitivities
$$\Delta_{\text{call}} = e^{-q T} \Phi(d_1), \quad \Delta_{\text{put}} = -e^{-q T} \Phi(-d_1)$$
$$\Gamma = \frac{e^{-q T} \phi(d_1)}{S \sigma \sqrt{T}}$$
$$\mathcal{V} = S e^{-q T} \sqrt{T} \phi(d_1) \times 0.01 \quad (\text{per } 1\% \Delta\sigma)$$
$$\Theta_{\text{call}} = -\frac{S e^{-q T} \phi(d_1) \sigma}{2\sqrt{T}} - r K e^{-r T}\Phi(d_2) + q S e^{-q T}\Phi(d_1)$$
$$\rho_{\text{call}} = K T e^{-r T} \Phi(d_2) \times 0.01, \quad \rho_{\text{put}} = -K T e^{-r T} \Phi(-d_2) \times 0.01$$

#### Delta-Gamma Neutralization
To neutralize both portfolio Delta ($\Delta_P$) and Gamma ($\Gamma_P$) using an option hedge instrument $H$ and underlying equity shares:
$$\Delta_{\text{target}} = \Delta_P + N_H \Delta_H + N_{\text{shares}} = 0$$
$$\Gamma_{\text{target}} = \Gamma_P + N_H \Gamma_H + 0 = 0$$
Solving sequentially:
$$N_H = -\frac{\Gamma_P}{\Gamma_H}$$
$$N_{\text{shares}} = -\Delta_P - N_H \Delta_H$$

---

### 2.5 Dynamic Position Sizing & Kelly Criterion

#### Continuous-Time Merton / Kelly Leverage
Optimal fraction of capital allocated to maximize the expected geometric growth rate of wealth:
$$f^* = \frac{\mu - r}{\sigma^2}$$
Expected growth rate:
$$g(f) = r + f(\mu - r) - \frac{1}{2} f^2 \sigma^2$$
To curb extreme tail drawdown, fractional Kelly is enforced ($f_{\text{rec}} = c \cdot f^*$, where $c \in [0.25, 0.50]$).

#### Discrete Win/Loss Kelly
$$f^* = p - \frac{1 - p}{b} = \frac{p \cdot b - (1 - p)}{b}$$
where $p$ is the win probability, and $b$ is the payoff ratio ($\frac{\text{Average Win}}{\text{Average Loss}}$).

#### Drawdown De-Risking Multiplier
When an account experiences an active drawdown $DD_t$:
$$\text{Multiplier}(DD_t) = \max\left(0, 1 - \frac{DD_t}{DD_{\max}}\right)$$
Position sizes scale downward linearly to avoid breaching hard stop mandates.

---

### 2.6 Portfolio Optimization Engines

1. **Maximum Sharpe Ratio (Tangency Portfolio)**:
   $$\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{\mu} - r_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i} w_i = 1, \; w_{\min} \le w_i \le w_{\max}$$

2. **Minimum Variance Portfolio**:
   $$\min_{\mathbf{w}} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \quad \text{s.t.} \quad \sum_{i} w_i = 1, \; w_{\min} \le w_i \le w_{\max}$$

3. **Equal Risk Contribution (Risk Parity)**:
   Marginal risk contribution of asset $i$:
   $$\text{MRC}_i = \frac{(\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p}$$
   Total risk contribution: $\text{TRC}_i = w_i \cdot \text{MRC}_i$. Risk parity solves:
   $$\min_{\mathbf{w}} \sum_{i=1}^N \sum_{j=1}^N \left( w_i (\mathbf{\Sigma} \mathbf{w})_i - w_j (\mathbf{\Sigma} \mathbf{w})_j \right)^2 \quad \text{s.t.} \quad \sum_{i} w_i = 1, \; w_i \ge 0$$

---

### 2.7 Execution & Market Microstructure

#### Square-Root Market Impact Model
Based on the Almgren-Chriss / Barra framework:
$$\text{Impact} = \eta \cdot \sqrt{\frac{Q}{V_{\text{daily}}}} \cdot \frac{\sigma_{\text{annual}}}{\sqrt{252}}$$
where $Q$ is order quantity, $V_{\text{daily}}$ is average daily volume, $\sigma_{\text{annual}}$ is asset annualized volatility, and $\eta$ is an empirical coefficient (default $0.10$).

#### Pre-Trade Risk Checks
Orders must pass four independent gates prior to submission:
1. **Notional Value Cap**: $Q \times P \le \text{MaxOrderNotional}$.
2. **Gross Leverage Limit**: Post-trade leverage $\frac{\text{GrossExposure}}{\text{NAV}} \le \text{MaxLeverage}$.
3. **Single-Name Weight Cap**: Asset weight $\le \text{MaxSinglePositionPct}$.
4. **Cash Adequacy**: $\text{RequiredCash} \le \text{AvailableCash} \times (1 - \text{Buffer})$.

---

## 3. Real-Time Risk Surveillance State Machine

The risk surveillance engine implements a tri-state circuit breaker:

```
  ┌───────────────────────────────────────────────────────────┐
  │                          NORMAL                           │
  │  - Full trading authorized                                │
  │  - Drawdown < Caution Threshold (8%)                      │
  │  - Margin Utilization < Warning Threshold (80%)           │
  └─────────────────────────────┬─────────────────────────────┘
                                │
               Drawdown >= 8%   │   Drawdown recovers < 8%
               or Margin >= 80% │   and Margin < 80%
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                         CAUTION                           │
  │  - Warning alerts broadcast to listeners                  │
  │  - Automated position size throttling                     │
  │  - High-frequency audit cycles                            │
  └─────────────────────────────┬─────────────────────────────┘
                                │
               Drawdown >= 15%  │   Admin manual override
               or Margin >= 95% │   & portfolio de-leveraging
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                          HALTED                           │
  │  - Trading fully suspended                                │
  │  - Emergency Kill-Switch triggered                        │
  │  - Automated market-order liquidation of long/short legs  │
  └───────────────────────────────────────────────────────────┘
```

---

## 4. Subsystem Module Breakdown

| Module | Core Responsibility | Key Classes / Functions |
| :--- | :--- | :--- |
| `types.py` | Strict Pydantic domain models & enumerations | `Portfolio`, `Position`, `Asset`, `VaRResult`, `MonteCarloConfig`, `MonteCarloResult`, `StressScenario`, `RiskLimits` |
| `engine/var.py` | Historical, Parametric, & Cornish-Fisher VaR | `VaRCalculator`, `calculate_historical_var`, `calculate_parametric_var`, `calculate_cornish_fisher_var` |
| `engine/evt.py` | Extreme Value Theory tail estimation | `EVTEngine`, `calculate_evt_var` |
| `engine/monte_carlo.py`| Vectorized correlated stochastic simulation | `MonteCarloEngine` |
| `engine/risk_engine.py`| Compliance auditing & macro stress tests | `PortfolioRiskEngine` |
| `engine/sizing.py` | Kelly criterion & volatility targeting | `PositionSizer` |
| `engine/optimizer.py`| Markowitz & Risk Parity optimization | `PortfolioOptimizer` |
| `engine/greeks.py` | Black-Scholes analytical option sensitivities| `calculate_option_greeks` |
| `engine/hedging.py`| Multi-leg Greeks aggregation & Delta-Gamma | `OptionHedgingEngine` |
| `data/feed.py` | Candlesticks, synthetic L1/L2 quotes, depth | `MarketDataFeed` |
| `execution/oms.py` | Slippage models, pre-trade gates, order routing | `OrderManagementSystem`, `PreTradeRiskGatekeeper`, `SlippageModel` |
| `strategy/signals.py`| Technical alpha signals & composite ensemble | `TrendFollowingMACDStrategy`, `MeanRevertingBollingerStrategy`, `CompositeAlphaAggregator` |
| `backtest/engine.py` | Event-driven sequential historical simulator | `BacktestEngine` |
| `surveillance/monitor.py` | Circuit breaker gating & emergency kill switch | `RiskSurveillanceMonitor` |
| `reporting/tearsheet.py`| Markdown tearsheets & JSON/CSV exporters | `RiskReportGenerator`, `DataExporter` |
| `storage/repository.py` | Institutional portfolio & stress scenario presets | `PortfolioRepository` |
| `cli/main.py` | Rich interactive terminal interface | `cli` group with 10 production commands |
