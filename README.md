# FinTech Algorithmic Trading Risk Engine & Monte Carlo VaR Analyzer

[![CI](https://github.com/zein/fintech-algorithmic-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/zein/fintech-algorithmic-trading/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Test Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/zein/fintech-algorithmic-trading)

A high-performance quantitative risk management engine, multi-asset Monte Carlo Value-at-Risk (VaR) analyzer, and event-driven backtesting platform built for institutional algorithmic trading strategies.

---

## Documentation Suite

Detailed technical guides and reference manuals are available in the [`docs/`](docs/) directory:

- 📐 **[System Architecture & Quantitative Formulations](docs/ARCHITECTURE.md)**: Deep mathematical formulations (VaR, EVT, GBM/Jump-Diffusion/Mean-Reversion, Black-Scholes, Delta-Gamma Neutralization, Kelly Criterion, Markowitz & Risk Parity) and state machine designs.
- 📚 **[Python API Reference](docs/API_REFERENCE.md)**: Comprehensive reference for all domain models, functions, signatures, parameters, and exceptions across all subsystems.
- 🚀 **[User & CLI Guide](docs/USAGE_GUIDE.md)**: Complete command-line manual, option explanations, downstream JSON recipes, and end-to-end Python workflows.

---

## Key Features

- **Multi-Method Value-at-Risk (VaR) & CVaR / Expected Shortfall**:
  - Historical Simulation VaR (empirical quantile scaling).
  - Parametric Gaussian VaR (with multi-asset covariance matrix evaluation).
  - Cornish-Fisher Expansion VaR (adjusting for skewness and excess kurtosis fat tails).
  - Extreme Value Theory (EVT) Peaks-Over-Threshold (POT) using Generalized Pareto Distribution (GPD).
- **Multi-Asset Stochastic Monte Carlo Engine**:
  - Correlated Geometric Brownian Motion (GBM) via Cholesky factor decomposition.
  - Merton Jump-Diffusion model with Poisson shocks and martingale compensator.
  - Ornstein-Uhlenbeck mean-reverting drift simulation.
  - Pathwise high-water mark drawdown and percentile tracking ($P_1, P_5, P_{50}, P_{95}, P_{99}$).
- **Analytical Option Greeks & Derivatives Sensitivities**:
  - Exact Black-Scholes formulas for European Call and Put options.
  - First and second-order Greeks: $\Delta, \Gamma, \mathcal{V}, \Theta, \rho$.
  - Multi-leg portfolio Greeks aggregation and automated Delta-Gamma neutralization solver.
- **Dynamic Position Sizing & Portfolio Optimization**:
  - Continuous Merton/Kelly optimal leverage fraction and growth rate bounds.
  - Discrete win/loss Kelly criterion with fraction capping.
  - Volatility-targeted inverse-variance position allocation.
  - Markowitz Modern Portfolio Theory (Max Sharpe, Min Variance) and Equal Risk Contribution (Risk Parity).
  - Active drawdown de-risking multiplier.
- **Order Management & Execution Simulation (OMS)**:
  - Pre-trade risk gatekeeper (notional limits, gross leverage ceilings, single-name concentration, cash sufficiency).
  - Fixed basis points and square-root market impact slippage modeling.
  - TWAP and VWAP multi-slice order routing simulation.
- **Quantitative Alpha Strategy Pipelines**:
  - Dual EMA / MACD momentum trend-following.
  - Bollinger Band mean-reversion with z-score normalization.
  - Donchian channel volatility breakout with ATR expansion.
  - Cointegrated Statistical Arbitrage pairs trading engine.
  - Multi-strategy ensemble composite aggregator.
- **Event-Driven Backtesting & Attribution**:
  - Sequential historical simulation with zero lookahead bias.
  - Full institutional KPI scorecard: Sharpe, Sortino, Calmar, Max Drawdown duration, Win Rate, Profit Factor, Expectancy.
  - Rolling 95% Historical VaR tracking along equity path.
- **Real-Time Risk Surveillance & Circuit Breaker**:
  - Tri-state circuit breaker (`NORMAL` $\to$ `CAUTION` $\to$ `HALTED`).
  - Automated emergency kill-switch for immediate portfolio de-leveraging.
  - Real-time margin utilization warning and critical alerts.
- **Crisis Stress Testing & Institutional Reporting**:
  - Historical crisis scenarios (2008 Lehman shock, 2020 Covid liquidity freeze, 2022 Rates repricing, Tech Rout, Crypto Winter).
  - Publication-ready Markdown tearsheets and JSON/CSV export utilities.

---

## System Architecture

```
                   ┌──────────────────────────────────────────┐
                   │            Market Data Feed              │
                   │  - Streaming OHLCV Candlestick Bars      │
                   │  - Synthetic Level 1 & Level 2 Depth     │
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

## Project Structure

```
fintech-algorithmic-trading/
├── docs/
│   ├── ARCHITECTURE.md         # System design, data flow, mathematical risk formulas
│   ├── API_REFERENCE.md        # Comprehensive Python API documentation
│   └── USAGE_GUIDE.md          # CLI user manual & programmatic Python workflows
├── src/
│   └── fintech_algorithmic_trading/
│       ├── types.py            # Pydantic domain models & enumerations
│       ├── engine/             # Quantitative mathematics & core engines
│       │   ├── var.py          # VaR & Expected Shortfall calculators
│       │   ├── evt.py          # Extreme Value Theory (POT / GPD) analyzer
│       │   ├── monte_carlo.py  # GBM, Jump-Diffusion, Mean-Reversion simulator
│       │   ├── risk_engine.py  # Real-time risk limits & stress tests
│       │   ├── sizing.py       # Kelly criterion & volatility targeting
│       │   ├── optimizer.py    # Markowitz & Risk Parity optimizers
│       │   ├── greeks.py       # Black-Scholes & option sensitivities
│       │   └── hedging.py      # Delta-Gamma hedging system
│       ├── data/               # Market data feeds & L2 order book depth
│       │   └── feed.py
│       ├── execution/          # OMS, slippage, & pre-trade risk gating
│       │   └── oms.py
│       ├── strategy/           # Alpha signals, MACD, Bollinger, StatArb
│       │   └── signals.py
│       ├── backtest/           # Event-driven backtesting & KPI attribution
│       │   └── engine.py
│       ├── surveillance/       # Real-time circuit breakers & kill-switch
│       │   └── monitor.py
│       ├── reporting/          # Tearsheet generator & CSV/JSON export
│       │   └── tearsheet.py
│       ├── storage/            # Portfolio profiles & market data loaders
│       │   └── repository.py
│       └── cli/                # Command-line interface with Rich
│           └── main.py
├── tests/
│   ├── conftest.py             # Shared fixtures & test helpers
│   ├── test_scaffold.py        # Baseline engine tests
│   ├── test_core_engines.py    # Math, optimization, and Greek calculations
│   ├── test_subsystems.py      # Subsystem integration tests
│   ├── test_edge_cases.py      # Numerical edge cases & boundary validations
│   └── test_cli_comprehensive.py # Full CLI invocation test suite
├── pyproject.toml
├── setup.py
├── pytest.ini
├── ruff.toml
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/zein/fintech-algorithmic-trading.git
cd fintech-algorithmic-trading

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### CLI Command Cheatsheet

| Command | Description | Example Usage |
| :--- | :--- | :--- |
| `list-portfolios` | List institutional benchmark portfolios | `fintech-algorithmic-trading list-portfolios` |
| `analyze-var` | Compute Historical, Parametric, & Cornish-Fisher VaR | `fintech-algorithmic-trading analyze-var -p global-macro -c 0.95` |
| `simulate` | Correlated Monte Carlo simulation | `fintech-algorithmic-trading simulate -p tech-momentum --sims 2000` |
| `stress-test` | Macro crisis stress testing | `fintech-algorithmic-trading stress-test -p multi-asset-balanced` |
| `check-limits` | Audit portfolio against risk policy mandate | `fintech-algorithmic-trading check-limits -p global-macro` |
| `greeks` | Compute analytical Black-Scholes option Greeks | `fintech-algorithmic-trading greeks -t CALL -s 500 -k 505 -e 0.25 -v 0.2` |
| `backtest` | Run event-driven historical simulation | `fintech-algorithmic-trading backtest -s SPY --strategy composite` |
| `surveillance` | Real-time circuit breaker & drawdown surveillance | `fintech-algorithmic-trading surveillance -p tech-momentum` |
| `orderbook` | Inspect Level 2 order book market depth | `fintech-algorithmic-trading orderbook -s NVDA --levels 5` |
| `report` | Generate institutional Markdown tearsheet | `fintech-algorithmic-trading report -p crypto-fx-alpha` |

---

## Python API Quickstart

```python
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.engine.var import VaRCalculator
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.engine.risk_engine import PortfolioRiskEngine

# 1. Retrieve benchmark portfolio
portfolio = PortfolioRepository.get_portfolio("global-macro")

# 2. Multi-Method VaR & CVaR Analysis
returns = PortfolioRepository.generate_synthetic_returns(n_days=750, vol_daily=0.012)
var_results = VaRCalculator.evaluate_all(
    returns=returns,
    portfolio_value=portfolio.net_asset_value,
    confidence_level=0.95,
)
for method, result in var_results.items():
    print(f"{method.value}: VaR = ${result.var_amount:,.2f} ({result.var_pct*100:.2f}%)")

# 3. Correlated Monte Carlo Path Simulation
mc_engine = MonteCarloEngine()
mc_result = mc_engine.simulate_portfolio(portfolio)
print(f"Mean Terminal Value: ${mc_result.mean_terminal_value:,.2f}")
print(f"P95 Terminal Value:  ${mc_result.p95_terminal_value:,.2f}")

# 4. Institutional Limit Mandate Compliance Audit
risk_engine = PortfolioRiskEngine()
compliance = risk_engine.check_compliance(portfolio)
print(f"Compliance Passed: {compliance.passed} (Violations: {len(compliance.violations)})")
```

---

## Testing & Quality Assurance

The codebase is rigorously validated with 100% test coverage, strict static typing via Mypy, and Ruff linting:

```bash
# Run pytest test suite
python -m pytest

# Run Ruff linter and code formatting checks
python -m ruff check .

# Run Mypy static type checker
python -m mypy src tests
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
