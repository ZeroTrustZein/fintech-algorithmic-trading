# FinTech Algorithmic Trading Risk Engine & Monte Carlo VaR Analyzer

A high-performance quantitative risk management engine and multi-asset Monte Carlo Value-at-Risk (VaR) analyzer for institutional algorithmic trading strategies.

## Features

- **Multi-Method Value-at-Risk (VaR) & CVaR / Expected Shortfall**:
  - Historical Simulation VaR
  - Parametric (Variance-Covariance) Gaussian VaR
  - Cornish-Fisher Expansion VaR (accounting for skewness and kurtosis / fat-tail risk)
  - Extreme Value Theory (EVT) and Tail-Risk Expected Shortfall (CVaR)
- **Multi-Asset Monte Carlo Simulation Engine**:
  - Correlated Geometric Brownian Motion (GBM) via Cholesky decomposition
  - Merton Jump-Diffusion model for sudden market shocks and flash crash regimes
  - Pathwise drawdown and Max Drawdown distribution tracking
  - Full percentile and quantile distribution analysis ($P_1, P_5, P_{50}, P_{95}, P_{99}$)
- **Real-Time Trading Risk Engine & Policy Verification**:
  - Leverage, margin utilization, and liquidity thresholds
  - Single-position and sector concentration limit checks
  - Dynamic volatility-targeted position sizing & Kelly criterion bounds
- **Analytical Option Greeks & Derivatives Sensitivities**:
  - Black-Scholes analytical $\Delta, \Gamma, \mathcal{V}, \Theta, \rho$
  - Portfolio delta/gamma hedge sizing
- **Crisis Stress Testing**:
  - Historical crisis scenarios (2008 Lehman shock, 2020 Covid liquidity freeze, 2022 Rates repricing)
  - Custom factor shock matrix & scenario sensitivity
- **Modern CLI**:
  - Rich-formatted tables, colorized risk thresholds, and JSON export capabilities

## Project Structure

```
fintech-algorithmic-trading/
├── src/
│   └── fintech_algorithmic_trading/
│       ├── types.py            # Pydantic domain models & enumerations
│       ├── engine/
│       │   ├── var.py          # VaR & Expected Shortfall calculators
│       │   ├── monte_carlo.py  # GBM & Jump-Diffusion correlated simulator
│       │   ├── risk_engine.py  # Real-time risk limits & stress tests
│       │   └── greeks.py       # Black-Scholes & option sensitivities
│       ├── storage/
│       │   └── repository.py   # Portfolio profiles & market data loaders
│       └── cli/
│           └── main.py         # Command-line interface with Rich
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── test_scaffold.py        # Comprehensive test suite
├── pyproject.toml
├── setup.py
└── README.md
```

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Run CLI
fintech-algorithmic-trading --help
fintech-algorithmic-trading list-portfolios
fintech-algorithmic-trading analyze-var --portfolio global-macro
fintech-algorithmic-trading simulate --portfolio tech-momentum --sims 2000
fintech-algorithmic-trading stress-test --portfolio multi-asset-balanced
fintech-algorithmic-trading check-limits --portfolio global-macro

# Run test suite
pytest
```
