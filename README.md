# FinTech Algorithmic Trading Risk Engine & Monte Carlo VaR Analyzer

[![CI](https://github.com/zein/fintech-algorithmic-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/zein/fintech-algorithmic-trading/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Test Coverage: 98%](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](https://github.com/zein/fintech-algorithmic-trading)

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
- **Order Management & Execution Simulation (OMS)**:
  - Pre-trade risk gatekeeper (notional limits, gross leverage ceilings, position concentration, cash adequacy)
  - Fixed basis points and square-root market impact slippage modeling
  - TWAP and VWAP multi-slice order routing simulation
- **Quantitative Alpha Strategy Pipelines**:
  - Dual EMA / MACD momentum trend-following
  - Bollinger Band mean-reversion with z-score normalization
  - Donchian channel volatility breakout with ATR expansion
  - Cointegrated Statistical Arbitrage pairs trading engine
  - Multi-strategy ensemble composite aggregator
- **Event-Driven Backtesting Engine**:
  - Sequential historical simulation with no lookahead bias
  - Full institutional KPI scorecard: Sharpe, Sortino, Calmar, Max Drawdown duration, Win Rate, Profit Factor, Expectancy
  - Rolling 95% Historical VaR tracking along equity path
- **Real-Time Risk Surveillance & Circuit Breaker**:
  - Drawdown threshold surveillance (CAUTION state at 8%, HALTED at 15%)
  - Automated emergency kill-switch for rapid portfolio de-leveraging
  - Margin utilization warning and critical liquidation alerts
- **Crisis Stress Testing**:
  - Historical crisis scenarios (2008 Lehman shock, 2020 Covid liquidity freeze, 2022 Rates repricing)
  - Custom factor shock matrix & scenario sensitivity
- **Institutional Reporting & Export**:
  - Full Markdown tearsheets with capital summary, position ledger, VaR matrix, Monte Carlo percentiles, and compliance audit
  - JSON and CSV export utilities for positions, backtest trade logs, and equity curves
- **Modern CLI**:
  - Rich-formatted tables, colorized risk thresholds, and JSON export capabilities

## Project Structure

```
fintech-algorithmic-trading/
├── src/
│   └── fintech_algorithmic_trading/
│       ├── types.py            # Pydantic domain models & enumerations
│       ├── engine/             # Quantitative mathematics & core engines
│       │   ├── var.py          # VaR & Expected Shortfall calculators
│       │   ├── evt.py          # Extreme Value Theory (POT / GPD) analyzer
│       │   ├── monte_carlo.py  # GBM & Jump-Diffusion correlated simulator
│       │   ├── risk_engine.py  # Real-time risk limits & stress tests
│       │   ├── sizing.py       # Kelly criterion & volatility targeting
│       │   ├── optimizer.py    # Markowitz & Risk Parity optimizers
│       │   ├── greeks.py       # Black-Scholes & option sensitivities
│       │   └── hedging.py      # Delta-Gamma hedging system
│       ├── data/               # Market data feeds & L2 order book depth
│       ├── execution/          # OMS, slippage, & pre-trade risk gating
│       ├── strategy/           # Alpha signals, MACD, Bollinger, StatArb
│       ├── backtest/           # Event-driven backtesting & KPI attribution
│       ├── surveillance/       # Real-time circuit breakers & kill-switch
│       ├── reporting/          # Tearsheet generator & CSV/JSON export
│       ├── storage/            # Portfolio profiles & market data loaders
│       └── cli/                # Command-line interface with Rich
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_scaffold.py        # Scaffold & basic engine tests
│   ├── test_core_engines.py    # Core math & optimization test suite
│   └── test_subsystems.py      # Subsystems integration test suite
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
fintech-algorithmic-trading backtest --symbol SPY --strategy composite --bars 252
fintech-algorithmic-trading surveillance --portfolio tech-momentum
fintech-algorithmic-trading orderbook --symbol NVDA --levels 5
fintech-algorithmic-trading report --portfolio crypto-fx-alpha

# Run test suite
pytest
```
