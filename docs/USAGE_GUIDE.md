# User Guide & Practical Workflows

A practical guide for quantitative analysts, portfolio risk officers, and algorithmic traders using **`fintech-algorithmic-trading`**.

---

## 1. Installation

### From Source
```bash
git clone https://github.com/zein/fintech-algorithmic-trading.git
cd fintech-algorithmic-trading

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows PowerShell/CMD
# source venv/bin/activate  # Linux / macOS

# Install package and dependencies
pip install -e .
```

### Verification
Ensure the CLI binary is available:
```bash
fintech-algorithmic-trading --help
```

---

## 2. Command-Line Interface (CLI) Guide

The CLI features interactive Rich tables, terminal coloring, and full `--json-out` support for CI/CD pipelines and downstream scripts.

### 2.1 Listing Portfolios
Inspect pre-configured benchmark portfolios, their Net Asset Values, gross exposure, position counts, and leverage:
```bash
fintech-algorithmic-trading list-portfolios
```
Output as JSON:
```bash
fintech-algorithmic-trading list-portfolios --json-out
```

---

### 2.2 Multi-Method Value-at-Risk Analysis
Evaluate Historical, Parametric Gaussian, Cornish-Fisher, and Extreme Value Theory (EVT) VaR & CVaR:
```bash
fintech-algorithmic-trading analyze-var --portfolio global-macro --confidence 0.95 --horizon 1
```
For a 10-day 99% holding period:
```bash
fintech-algorithmic-trading analyze-var -p tech-momentum -c 0.99 -h 10
```

---

### 2.3 Stochastic Monte Carlo Path Simulation
Simulate multi-asset correlated trajectories across market regimes:
```bash
# Standard Geometric Brownian Motion
fintech-algorithmic-trading simulate --portfolio tech-momentum --sims 2000 --days 21 --drift GBM

# Merton Jump-Diffusion (liquidity shocks & flash crashes)
fintech-algorithmic-trading simulate --portfolio crypto-fx-alpha --sims 5000 --days 30 --drift JUMP_DIFFUSION

# Ornstein-Uhlenbeck Mean-Reverting Drift
fintech-algorithmic-trading simulate --portfolio multi-asset-balanced --sims 3000 --days 63 --drift MEAN_REVERTING
```

---

### 2.4 Crisis Stress Testing
Stress-test portfolios against historical macroeconomic shocks:
```bash
# Test all historical crisis scenarios (2008 Lehman, 2020 Covid, 2022 Rates Shock)
fintech-algorithmic-trading stress-test --portfolio multi-asset-balanced

# Target a specific shock scenario
fintech-algorithmic-trading stress-test --portfolio global-macro --scenario lehman-2008
fintech-algorithmic-trading stress-test --portfolio crypto-fx-alpha --scenario crypto-winter
```

---

### 2.5 Risk Policy Limits Audit
Validate a portfolio against institutional mandates (leverage, single-name concentration, sector exposure, cash buffer, and maximum 1-day 95% VaR):
```bash
fintech-algorithmic-trading check-limits --portfolio global-macro
```

---

### 2.6 Analytical Option Greeks Calculator
Calculate theoretical contract price and Black-Scholes sensitivities ($\Delta, \Gamma, \mathcal{V}, \Theta, \rho$):
```bash
# 3-month Out-of-the-Money Call Option
fintech-algorithmic-trading greeks --type CALL --spot 500.0 --strike 510.0 --expiry 0.25 --vol 0.22 --rate 0.045

# At-the-Money Put Option
fintech-algorithmic-trading greeks --type PUT --spot 100.0 --strike 100.0 --expiry 0.50 --vol 0.35 --rate 0.04
```

---

### 2.7 Quantitative Alpha Strategy Backtesting
Execute sequential historical simulation with realistic commissions and slippage:
```bash
# Dual EMA / MACD Trend Following
fintech-algorithmic-trading backtest --symbol SPY --strategy macd --bars 252

# Bollinger Bands Mean Reversion
fintech-algorithmic-trading backtest --symbol AAPL --strategy bollinger --bars 300

# Donchian Volatility Breakout
fintech-algorithmic-trading backtest --symbol NVDA --strategy breakout --bars 252

# Multi-Strategy Composite Ensemble
fintech-algorithmic-trading backtest --symbol QQQ --strategy composite --bars 500 --capital 2000000
```

---

### 2.8 Real-Time Risk Surveillance
Audit live portfolio drawdown, margin utilization, and circuit breaker status:
```bash
fintech-algorithmic-trading surveillance --portfolio tech-momentum
```

---

### 2.9 Order Book Market Depth
Query simulated Level 2 bid/ask liquidity depth:
```bash
fintech-algorithmic-trading orderbook --symbol NVDA --levels 5
```

---

### 2.10 Institutional Risk Tearsheet Generation
Generate comprehensive Markdown risk memos containing capital summary, position ledger, VaR matrix, and compliance audit:
```bash
# Print tearsheet directly to stdout
fintech-algorithmic-trading report --portfolio crypto-fx-alpha

# Export tearsheet to a Markdown file
fintech-algorithmic-trading report --portfolio global-macro --output tearsheet-global-macro.md
```

---

### 2.11 PostgreSQL Schema & Validation Script (`sql-schema`)
Generate complete PostgreSQL schema DDL or transformation scripts for database initialization:
```bash
# Display staging, audit, and clean table DDL
fintech-algorithmic-trading sql-schema

# Export schema DDL to file
fintech-algorithmic-trading sql-schema --output schema.sql

# Display SQL data cleaning, validation, and quarantine transformation script
fintech-algorithmic-trading sql-schema --validate-script

# Export validation transformation script to file
fintech-algorithmic-trading sql-schema --validate-script --output clean_validate.sql
```

---

### 2.12 Raw Export Cleaning & Validation (`clean-export`)
Clean, deduplicate, and validate raw trades, positions, or equity curve exports:
```bash
# Clean default synthetic trade journal
fintech-algorithmic-trading clean-export --type trades

# Clean raw trade export from CSV file
fintech-algorithmic-trading clean-export --type trades --file raw_trades.csv

# Clean portfolio positions and save clean records to CSV
fintech-algorithmic-trading clean-export --type positions --portfolio tech-momentum --output clean_positions.csv

# Clean equity curve and emit validation summary as JSON
fintech-algorithmic-trading clean-export --type equity --file raw_equity.csv --json-out
```

---

### 2.13 SQL Ingestion & COPY Generation (`sql-generate`)
Generate high-throughput PostgreSQL staging ingestion commands:
```bash
# Generate bulk SQL INSERT statements for trades
fintech-algorithmic-trading sql-generate --type trades --file raw_trades.csv

# Generate high-performance PostgreSQL COPY command
fintech-algorithmic-trading sql-generate --type trades --file /var/data/raw_trades.csv --copy-command

# Generate positions staging INSERTs from portfolio preset
fintech-algorithmic-trading sql-generate --type positions --portfolio global-macro --output stage_positions.sql
```

---

## 3. Python Workflows & Integration Examples

### 3.1 Multi-Method VaR & CVaR Computation
```python
import numpy as np
from fintech_algorithmic_trading.engine.var import VaRCalculator
from fintech_algorithmic_trading.storage.repository import PortfolioRepository

# Load benchmark portfolio
portfolio = PortfolioRepository.get_portfolio("global-macro")

# Generate synthetic returns calibrated to portfolio volatility
returns = PortfolioRepository.generate_synthetic_returns(n_days=750, vol_daily=0.012)

# Compute all VaR methodologies
results = VaRCalculator.evaluate_all(
    returns=returns,
    portfolio_value=portfolio.net_asset_value,
    confidence_level=0.95,
    horizon_days=1,
)

for method, res in results.items():
    print(
        f"{method.value:20s}: VaR = ${res.var_amount:,.2f} ({res.var_pct * 100:.2f}%), CVaR = ${res.cvar_amount:,.2f}"
    )
```

---

### 3.2 Correlated Monte Carlo Simulation
```python
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.types import MonteCarloConfig, DriftModel
from fintech_algorithmic_trading.storage.repository import PortfolioRepository

portfolio = PortfolioRepository.get_portfolio("tech-momentum")

# Configure Jump-Diffusion model
config = MonteCarloConfig(
    n_simulations=5000,
    horizon_days=21,
    time_steps=21,
    drift_model=DriftModel.JUMP_DIFFUSION,
    jump_intensity=0.15,
    jump_mean=-0.06,
    jump_std=0.10,
    random_seed=42,
)

engine = MonteCarloEngine(config=config)
result = engine.simulate_portfolio(portfolio)

print(f"Mean Terminal Value: ${result.mean_terminal_value:,.2f}")
print(
    f"95% Simulated VaR:   ${result.simulated_var_95_amount:,.2f} ({result.simulated_var_95_pct * 100:.2f}%)"
)
print(f"Expected Max DD:     {result.max_drawdown_mean_pct * 100:.2f}%")
```

---

### 3.3 Dynamic Delta-Gamma Dynamic Hedging
```python
from fintech_algorithmic_trading.types import OptionContract, OptionType
from fintech_algorithmic_trading.engine.hedging import OptionHedgingEngine

# Define current option positions
contracts = [
    OptionContract(
        symbol="SPY-CALL-550",
        underlying_symbol="SPY",
        option_type=OptionType.CALL,
        spot_price=540.0,
        strike_price=550.0,
        time_to_expiry_years=0.15,
        volatility=0.18,
        quantity=50.0,
    ),
    OptionContract(
        symbol="SPY-PUT-530",
        underlying_symbol="SPY",
        option_type=OptionType.PUT,
        spot_price=540.0,
        strike_price=530.0,
        time_to_expiry_years=0.15,
        volatility=0.20,
        quantity=-30.0,
    ),
]

# Calculate portfolio Greeks
portfolio_greeks = OptionHedgingEngine.calculate_portfolio_greeks(
    contracts=contracts,
    underlying_shares=1500.0,
)
print(f"Net Delta: {portfolio_greeks.net_delta:.2f}, Net Gamma: {portfolio_greeks.net_gamma:.6f}")

# Target Delta-neutrality
delta_hedge = OptionHedgingEngine.calculate_delta_hedge(
    current_net_delta=portfolio_greeks.net_delta,
    target_delta=0.0,
    underlying_symbol="SPY",
)
print(f"Shares needed to neutralize Delta: {delta_hedge.underlying_shares_needed:.1f}")
```

---

### 3.4 Modern Portfolio Optimization (Markowitz & Risk Parity)
```python
import numpy as np
from fintech_algorithmic_trading.engine.optimizer import PortfolioOptimizer
from fintech_algorithmic_trading.types import OptimizationObjective

symbols = ["SPY", "QQQ", "TLT", "GLD"]
expected_returns = np.array([0.10, 0.14, 0.04, 0.07])
covariance_matrix = np.array(
    [
        [0.0256, 0.0280, -0.0050, 0.0020],
        [0.0280, 0.0484, -0.0070, 0.0030],
        [-0.0050, -0.0070, 0.0196, 0.0040],
        [0.0020, 0.0030, 0.0040, 0.0225],
    ]
)

optimizer = PortfolioOptimizer(risk_free_rate=0.045)

# 1. Max Sharpe Ratio
res_sharpe = optimizer.optimize(
    symbols=symbols,
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    objective=OptimizationObjective.MAX_SHARPE,
)
print("Max Sharpe Weights:", res_sharpe.weights)
print("Sharpe Ratio:", res_sharpe.sharpe_ratio)

# 2. Risk Parity (Equal Risk Contribution)
res_rp = optimizer.optimize(
    symbols=symbols,
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    objective=OptimizationObjective.RISK_PARITY,
)
print("Risk Parity Weights:", res_rp.weights)
print("Risk Contributions:", res_rp.risk_contributions)
```

---

### 3.5 Event-Driven Backtesting
```python
from fintech_algorithmic_trading.backtest import BacktestEngine
from fintech_algorithmic_trading.strategy import TrendFollowingMACDStrategy
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.types import BacktestConfig

bars = PortfolioRepository.generate_synthetic_bars("SPY", n_bars=300, start_price=500.0)
strategy = TrendFollowingMACDStrategy()

config = BacktestConfig(
    initial_capital=1_000_000.0,
    commission_bps=5.0,
    slippage_bps=2.5,
)

engine = BacktestEngine(config=config)
result = engine.run(symbol="SPY", bars=bars, strategy=strategy)

print(f"Total Return:     {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio:     {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown:     {result.metrics.max_drawdown_pct * 100:.2f}%")
print(f"Total Trades:     {result.metrics.total_trades}")
print(f"Win Rate:         {result.metrics.win_rate * 100:.1f}%")
```

---

### 3.6 Cleaning and Validating Backtest Trades with PostgreSQL Pipeline
Run an event-driven backtest, export raw trades, and validate them through `PostgreSQLExportPipeline`:
```python
from fintech_algorithmic_trading.backtest import BacktestEngine
from fintech_algorithmic_trading.strategy import TrendFollowingMACDStrategy
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.reporting.tearsheet import DataExporter
from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline
from fintech_algorithmic_trading.types import ExportType

# 1. Execute backtest
bars = PortfolioRepository.generate_synthetic_bars("QQQ", n_bars=350, start_price=400.0)
engine = BacktestEngine()
result = engine.run("QQQ", bars, TrendFollowingMACDStrategy())

# 2. Export & Clean trades directly
summary, clean_trades = DataExporter.export_and_clean_trades(result.trades)
print(f"Total Raw Trades:  {summary.total_raw_records}")
print(f"Valid Clean Trades:{summary.valid_clean_records}")
print(f"Quarantined:       {summary.quarantined_records}")
print(f"Duplicates:        {summary.duplicate_records}")

# 3. Generate PostgreSQL Staging DDL & Ingest Commands
insert_sql = PostgreSQLExportPipeline.generate_staging_insert_sql(
    ExportType.TRADES, [t.model_dump() for t in clean_trades]
)
print(f"Generated {len(clean_trades)} row staging INSERT statement.")
```

---

### 3.7 End-to-End PostgreSQL Ingestion & SQL Transformations
Executing database schema migration, staging load, and transformation with SQL on PostgreSQL:
```bash
# 1. Initialize PostgreSQL schema
psql -h localhost -U postgres -d trading_db -c "$(fintech-algorithmic-trading sql-schema)"

# 2. Stage raw export via COPY command
psql -h localhost -U postgres -d trading_db -c "\copy raw_trades_export (trade_id, symbol, side, quantity, entry_price, exit_price, entry_time, exit_time, pnl, return_pct, commission, holding_period_bars) FROM 'raw_trades.csv' WITH (FORMAT csv, HEADER true);"

# 3. Execute SQL cleaning, window deduplication, and quarantine logic
psql -h localhost -U postgres -d trading_db -c "$(fintech-algorithmic-trading sql-schema --validate-script)"

# 4. Inspect quarantine errors
psql -h localhost -U postgres -d trading_db -c "SELECT rule_name, COUNT(*) FROM export_validation_errors GROUP BY rule_name;"

# 5. Query clean production trades
psql -h localhost -U postgres -d trading_db -c "SELECT symbol, COUNT(*), SUM(pnl) FROM clean_trades GROUP BY symbol;"
```

---

### 3.8 Domain Object Roundtrip Reconstruction
Reconstruct native `Portfolio` or `List[BacktestTrade]` objects from validated database records:
```python
from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline
from fintech_algorithmic_trading.types import ExportType

raw_positions = [
    {
        "portfolio_id": "prod-desk-1",
        "symbol": "NVDA",
        "name": "NVIDIA Corp",
        "asset_class": "EQUITY",
        "sector": "Technology",
        "side": "LONG",
        "quantity": 500.0,
        "entry_price": 120.0,
        "current_price": 135.0,
        "market_value": 67500.0,
        "unrealized_pnl": 7500.0,
        "unrealized_pnl_pct": 0.125,
    }
]

# Clean positions
summary, clean_positions = PostgreSQLExportPipeline.process_positions(raw_positions)

# Reconstruct live domain Portfolio object
portfolio = PostgreSQLExportPipeline.reconstruct_portfolio(
    portfolio_id="prod-desk-1",
    clean_positions=clean_positions,
    initial_cash=50_000.0,
)

print(f"Reconstructed Portfolio: {portfolio.id}")
print(f"Net Asset Value:        ${portfolio.net_asset_value:,.2f}")
print(f"Leverage:                {portfolio.leverage:.2f}x")
print(f"Positions Count:         {len(portfolio.positions)}")
```
