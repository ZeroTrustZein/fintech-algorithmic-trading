# Python API Reference

Comprehensive reference for public modules, classes, mathematical functions, and data structures in **`fintech-algorithmic-trading`**.

---

## Table of Contents

- [1. Domain Models (`types`)](#1-domain-models-types)
- [2. Value-at-Risk Engine (`engine.var`)](#2-value-at-risk-engine-enginevar)
- [3. Extreme Value Theory (`engine.evt`)](#3-extreme-value-theory-engineevt)
- [4. Monte Carlo Simulator (`engine.monte_carlo`)](#4-monte-carlo-simulator-enginemonte_carlo)
- [5. Risk Compliance & Stress Testing (`engine.risk_engine`)](#5-risk-compliance--stress-testing-enginerisk_engine)
- [6. Position Sizing & Kelly Criterion (`engine.sizing`)](#6-position-sizing--kelly-criterion-enginesizing)
- [7. Portfolio Optimization (`engine.optimizer`)](#7-portfolio-optimization-engineoptimizer)
- [8. Option Greeks & Sensitivities (`engine.greeks`)](#8-option-greeks--sensitivities-enginegreeks)
- [9. Dynamic Hedging Engine (`engine.hedging`)](#9-dynamic-hedging-engine-enginehedging)
- [10. Market Data Feed (`data.feed`)](#10-market-data-feed-datafeed)
- [11. Order Management System (`execution.oms`)](#11-order-management-system-executionoms)
- [12. Quantitative Alpha Strategies (`strategy.signals`)](#12-quantitative-alpha-strategies-strategysignals)
- [13. Backtest Engine (`backtest.engine`)](#13-backtest-engine-backtestengine)
- [14. Risk Surveillance & Circuit Breakers (`surveillance.monitor`)](#14-risk-surveillance--circuit-breakers-surveillancemonitor)
- [15. Reporting & Export (`reporting.tearsheet`)](#15-reporting--export-reportingtearsheet)
- [16. Repository & Presets (`storage.repository`)](#16-repository--presets-storagerepository)

---

## 1. Domain Models (`types`)

Package: `fintech_algorithmic_trading.types`

### Core Enumerations

- **`AssetClass`**: `EQUITY`, `FIXED_INCOME`, `COMMODITY`, `FX`, `CRYPTO`, `DERIVATIVE`.
- **`OrderSide`**: `BUY`, `SELL`.
- **`PositionType`**: `LONG`, `SHORT`.
- **`VaRMethod`**: `HISTORICAL`, `PARAMETRIC`, `CORNISH_FISHER`, `MONTE_CARLO`, `EXTREME_VALUE_THEORY`.
- **`DriftModel`**: `GBM`, `JUMP_DIFFUSION`, `MEAN_REVERTING`.
- **`OptionType`**: `CALL`, `PUT`.
- **`OptimizationObjective`**: `MAX_SHARPE`, `MIN_VARIANCE`, `RISK_PARITY`, `EQUAL_WEIGHT`.
- **`KellyCriterionMode`**: `CONTINUOUS`, `DISCRETE`.
- **`OrderType`**: `MARKET`, `LIMIT`, `STOP_LOSS`, `TWAP`, `VWAP`.
- **`OrderStatus`**: `PENDING`, `FILLED`, `PARTIALLY_FILLED`, `REJECTED`, `CANCELLED`.
- **`TimeInForce`**: `GTC`, `IOC`, `FOK`, `DAY`.
- **`SignalDirection`**: `LONG`, `SHORT`, `FLAT`.
- **`AlertSeverity`**: `INFO`, `WARNING`, `CRITICAL`, `EMERGENCY_KILL_SWITCH`.
- **`CircuitBreakerState`**: `NORMAL`, `CAUTION`, `HALTED`.

### Primary Pydantic Models

#### `Asset`
Tradable financial instrument representation.
- `symbol: str` — Ticker symbol.
- `name: str` — Full descriptive security name.
- `asset_class: AssetClass` — Asset category.
- `sector: Optional[str] = "General"` — Industry classification.
- `currency: str = "USD"` — Denomination currency.
- `current_price: float` (gt 0) — Current market spot price.
- `volatility_annual: float` (ge 0) — Annualized return standard deviation.
- `beta_to_market: float = 1.0` — Beta relative to benchmark.
- `dividend_yield: float = 0.0` — Annual dividend or yield rate.

#### `Position`
Open investment position.
- `asset: Asset` — Target instrument.
- `quantity: float` (gt 0) — Units held.
- `entry_price: float` (ge 0) — Cost basis per share.
- `current_price: float` (gt 0) — Latest mark price.
- `side: PositionType = PositionType.LONG` — Directional bias.
- **Computed Properties**: `market_value`, `cost_basis`, `unrealized_pnl`, `unrealized_pnl_pct`.

#### `Portfolio`
Multi-asset institutional account container.
- `id: str` — Unique portfolio identifier.
- `name: str` — Human-readable name.
- `description: str` — Mandate narrative.
- `cash: float = 0.0` (ge 0) — Free unencumbered cash.
- `positions: List[Position]` — List of open positions.
- `base_currency: str = "USD"`.
- **Computed Properties**: `net_asset_value`, `total_positions_value`, `gross_exposure`, `net_exposure`, `leverage`, `asset_weights`, `sector_weights`.

---

## 2. Value-at-Risk Engine (`engine.var`)

Package: `fintech_algorithmic_trading.engine.var`

### Functions

#### `calculate_historical_var`
```python
def calculate_historical_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult
```
Computes empirical percentile VaR and tail-mean Expected Shortfall (CVaR).
- **Parameters**:
  - `returns`: Array of historical return observations ($\ge 5$).
  - `portfolio_value`: Total portfolio NAV.
  - `confidence_level`: Significance quantile ($\alpha \in (0, 1)$).
  - `horizon_days`: Holding period ($\ge 1$).
- **Returns**: `VaRResult` containing `var_amount`, `var_pct`, `cvar_amount`, `cvar_pct`, `skewness`, and `excess_kurtosis`.

#### `calculate_parametric_var`
```python
def calculate_parametric_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult
```
Analytic Gaussian VaR with closed-form normal Expected Shortfall.

#### `calculate_cornish_fisher_var`
```python
def calculate_cornish_fisher_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult
```
Computes skewness- and kurtosis-adjusted VaR via higher-moment polynomial expansion.

### Class: `VaRCalculator`

- `evaluate_all(returns, portfolio_value, confidence_level=0.95, horizon_days=1, include_evt=True) -> Dict[VaRMethod, VaRResult]`
  - Runs Historical, Parametric, Cornish-Fisher, and EVT calculators concurrently.
- `calculate_portfolio_parametric_var(portfolio, asset_covariance_matrix, symbols_order, confidence_level=0.95, horizon_days=1) -> VaRResult`
  - Matrix-based parametric VaR utilizing multi-asset portfolio weight vector $\mathbf{w}$ and covariance matrix $\mathbf{\Sigma}$.

---

## 3. Extreme Value Theory (`engine.evt`)

Package: `fintech_algorithmic_trading.engine.evt`

### Class: `EVTEngine`

```python
class EVTEngine:
    def __init__(self, threshold_quantile: float = 0.90): ...
```
- **Methods**:
  - `fit_pot(losses: np.ndarray, threshold: Optional[float] = None) -> tuple[float, float, float, np.ndarray]`
    - Fits Generalized Pareto Distribution parameters $(\xi, \beta)$ to exceedances over threshold $u$.
  - `calculate_evt_var(returns, portfolio_value, confidence_level=0.95, horizon_days=1, threshold=None) -> EVTResult`
    - Computes POT VaR and CVaR using estimated GPD shape and scale parameters.

---

## 4. Monte Carlo Simulator (`engine.monte_carlo`)

Package: `fintech_algorithmic_trading.engine.monte_carlo`

### Class: `MonteCarloEngine`

```python
class MonteCarloEngine:
    def __init__(self, config: Optional[MonteCarloConfig] = None): ...
    def simulate_portfolio(
        self,
        portfolio: Portfolio,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> MonteCarloResult: ...
```
Simulates full multi-asset stochastic trajectories with Cholesky-correlated Brownian motions.
- **Supported Drift Models**:
  - `DriftModel.GBM`: Standard Geometric Brownian Motion.
  - `DriftModel.JUMP_DIFFUSION`: Merton Jump-Diffusion with Poisson arrival and compensator.
  - `DriftModel.MEAN_REVERTING`: Ornstein-Uhlenbeck mean-reversion drift.
- **Output (`MonteCarloResult`)**:
  - Terminal distribution: Mean, median, min, max, $P_1, P_5, P_{50}, P_{95}, P_{99}$.
  - Simulated VaR and CVaR at 95% and 99%.
  - Pathwise maximum drawdown distribution: Mean and 95th percentile.
  - Probability of capital loss.

---

## 5. Risk Compliance & Stress Testing (`engine.risk_engine`)

Package: `fintech_algorithmic_trading.engine.risk_engine`

### Class: `PortfolioRiskEngine`

```python
class PortfolioRiskEngine:
    def __init__(self, limits: Optional[RiskLimits] = None): ...
    def check_compliance(self, portfolio: Portfolio, custom_limits: Optional[RiskLimits] = None) -> RiskComplianceReport: ...
    def run_stress_test(self, portfolio: Portfolio, scenario: StressScenario) -> StressTestResult: ...
    def run_stress_suite(self, portfolio: Portfolio, scenarios: List[StressScenario]) -> List[StressTestResult]: ...
```

- **Audited Limits (`RiskLimits`)**:
  - `max_leverage`: Maximum allowed gross leverage ratio.
  - `max_single_position_pct`: Maximum single-name asset concentration.
  - `max_sector_pct`: Maximum industry sector concentration.
  - `max_var_95_pct`: 1-day 95% parametric VaR ceiling.
  - `min_cash_buffer_pct`: Minimum liquid cash buffer ratio.
- **Crisis Stress Testing**: Applies factor shocks across equity (beta-scaled), fixed income (duration-scaled), FX, commodities, and crypto.

---

## 6. Position Sizing & Kelly Criterion (`engine.sizing`)

Package: `fintech_algorithmic_trading.engine.sizing`

### Class: `PositionSizer`

- `calculate_continuous_kelly(expected_return_annual, volatility_annual, risk_free_rate=0.045, portfolio_nav=100000.0, fractional_factor=0.5, max_leverage_cap=2.0) -> KellySizingResult`
  - Merton-continuous Kelly fraction $f^* = \frac{\mu - r}{\sigma^2}$.
- `calculate_discrete_kelly(win_rate, win_loss_ratio, portfolio_nav=100000.0, fractional_factor=0.5, max_fraction_cap=0.30) -> KellySizingResult`
  - Discrete win/loss ratio sizing $f^* = \frac{p \cdot b - q}{b}$.
- `calculate_volatility_target_allocation(target_vol_annual, asset_vols_annual, portfolio_nav, max_weight_cap=0.35) -> dict[str, float]`
  - Risk-budgeted inverse-volatility allocation.
- `calculate_drawdown_scaled_size(base_size, current_drawdown_pct, max_tolerated_drawdown_pct=0.15) -> float`
  - Linear position throttling during account drawdown.

---

## 7. Portfolio Optimization (`engine.optimizer`)

Package: `fintech_algorithmic_trading.engine.optimizer`

### Class: `PortfolioOptimizer`

```python
class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.045): ...
    def optimize(
        self,
        symbols: List[str],
        expected_returns: Sequence[float] | np.ndarray,
        covariance_matrix: np.ndarray,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        max_weight_cap: float = 1.0,
        min_weight_floor: float = 0.0,
    ) -> OptimizationResult: ...
    def optimize_portfolio(
        self,
        portfolio: Portfolio,
        asset_covariance_matrix: np.ndarray,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        max_weight_cap: float = 1.0,
    ) -> OptimizationResult: ...
    def calculate_risk_contributions(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: Sequence[str],
    ) -> Dict[str, float]: ...
```
- **Objectives**:
  - `MAX_SHARPE`: Tangency portfolio maximizing excess return per unit of volatility.
  - `MIN_VARIANCE`: Global minimum variance portfolio.
  - `RISK_PARITY`: Equalizes each asset's percentage contribution to total portfolio risk.
  - `EQUAL_WEIGHT`: $1/N$ naive baseline allocation.

---

## 8. Option Greeks & Sensitivities (`engine.greeks`)

Package: `fintech_algorithmic_trading.engine.greeks`

### Function: `calculate_option_greeks`

```python
def calculate_option_greeks(
    option_type: OptionType,
    spot_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> OptionGreeks
```
Returns Black-Scholes analytical sensitivities:
- `price`: Theoretical contract fair value.
- `delta` ($\Delta$): First derivative with respect to underlying spot price $\frac{\partial V}{\partial S}$.
- `gamma` ($\Gamma$): Second derivative with respect to spot price $\frac{\partial^2 V}{\partial S^2}$.
- `vega` ($\mathcal{V}$): Sensitivity to 1 percentage point change in annualized volatility $\frac{\partial V}{\partial \sigma} \times 0.01$.
- `theta` ($\Theta$): Daily calendar time decay $\frac{\partial V}{\partial t} / 365$.
- `rho` ($\rho$): Sensitivity to 1 percentage point change in risk-free interest rate $\frac{\partial V}{\partial r} \times 0.01$.

---

## 9. Dynamic Hedging Engine (`engine.hedging`)

Package: `fintech_algorithmic_trading.engine.hedging`

### Class: `OptionHedgingEngine`

- `calculate_portfolio_greeks(contracts: List[OptionContract], underlying_shares: float = 0.0, risk_free_rate: float = 0.045) -> PortfolioGreeks`
  - Aggregates Delta, Gamma, Vega, Theta, and Rho across derivative contracts and equity shares.
- `calculate_delta_hedge(current_net_delta: float, target_delta: float = 0.0, underlying_symbol: str = "UNDERLYING") -> DeltaGammaHedge`
  - Rebalances underlying equity shares to target delta posture.
- `calculate_delta_gamma_hedge(current_greeks: PortfolioGreeks, hedge_option_contract: OptionContract, target_delta: float = 0.0, target_gamma: float = 0.0) -> DeltaGammaHedge`
  - Simultaneously solves for option contract count and underlying equity units to achieve delta-gamma neutrality.

---

## 10. Market Data Feed (`data.feed`)

Package: `fintech_algorithmic_trading.data.feed`

### Class: `MarketDataFeed`

- `add_bar(bar: Bar, publish: bool = True) -> None`
  - Appends candlestick and automatically constructs synthetic top-of-book `Quote`.
- `load_bars(symbol: str, bars: List[Bar]) -> None`
  - Bulk imports historical data.
- `get_bars(symbol: str, n_bars: Optional[int] = None) -> List[Bar]`
  - Retrieves stored bar series.
- `get_latest_quote(symbol: str) -> Optional[Quote]`
  - Accesses live bid/ask spread and mid price.
- `get_order_book_depth(symbol: str, levels: int = 5) -> OrderBookDepth`
  - Simulates multi-level L2 order book bids and asks with market depth.
- `subscribe_bars(symbol: str, callback: Callable[[Bar], None]) -> None`
- `subscribe_quotes(symbol: str, callback: Callable[[Quote], None]) -> None`
- `resample_bars(symbol: str, target_timeframe_minutes: int) -> List[Bar]`
  - Aggregates high-frequency bars into higher timeframe intervals.

---

## 11. Order Management System (`execution.oms`)

Package: `fintech_algorithmic_trading.execution.oms`

### Class: `SlippageModel`
- `calculate_fixed_bps_slippage(price: float, side: OrderSide, slippage_bps: float = 2.5) -> float`
- `calculate_market_impact_slippage(price, side, quantity, daily_volume=1_000_000.0, volatility_annual=0.20, impact_factor=0.10) -> Tuple[float, float]`

### Class: `PreTradeRiskGatekeeper`
- `validate_order(order: Order, portfolio: Portfolio, current_quote: Optional[Quote] = None) -> Tuple[bool, Optional[str]]`
  - Validates order notional limits, post-trade leverage, single-position weight caps, and available cash.

### Class: `OrderManagementSystem`
- `submit_order(order: Order, portfolio: Portfolio, current_quote: Optional[Quote] = None, pre_trade_check: bool = True) -> ExecutionReport`
- `simulate_twap_execution(order: Order, portfolio: Portfolio, num_slices: int = 5, current_quote: Optional[Quote] = None) -> ExecutionReport`
- `simulate_vwap_execution(order: Order, portfolio: Portfolio, volume_profile: List[float], current_quote: Optional[Quote] = None) -> ExecutionReport`
- `cancel_order(order_id: str) -> bool`
- `get_order(order_id: str) -> Optional[Order]`

---

## 12. Quantitative Alpha Strategies (`strategy.signals`)

Package: `fintech_algorithmic_trading.strategy.signals`

### Base Class: `AlphaStrategy`
Abstract base class defining `generate_signal(symbol: str, bars: List[Bar]) -> Optional[StrategySignal]`.

### Concrete Strategies

- **`TrendFollowingMACDStrategy(fast_period=12, slow_period=26, signal_period=9)`**
  - Emits `LONG` or `SHORT` signals on MACD histogram zero-line crossovers.
- **`MeanRevertingBollingerStrategy(window=20, num_std=2.0)`**
  - Generates mean-reverting signals when price crosses outer Bollinger bands with z-score normalization.
- **`VolatilityBreakoutStrategy(donchian_period=20, atr_period=14, atr_multiplier=1.5)`**
  - Trades channel breakouts confirmed by Average True Range (ATR) expansion.
- **`PairsStatisticalArbitrageStrategy(pair_symbol_a, pair_symbol_b, lookback_window=60, entry_zscore=2.0, exit_zscore=0.5)`**
  - Cointegrated pairs trading engine calculating rolling spread z-scores.
- **`CompositeAlphaAggregator(strategies: List[AlphaStrategy], weights: Optional[List[float]] = None)`**
  - Ensembles multiple independent signal streams into a single weighted target position allocation.

---

## 13. Backtest Engine (`backtest.engine`)

Package: `fintech_algorithmic_trading.backtest.engine`

### Class: `BacktestEngine`

```python
class BacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None, oms: Optional[OrderManagementSystem] = None): ...
    def run(
        self,
        symbol: str,
        bars: List[Bar],
        strategy: Union[AlphaStrategy, CompositeAlphaAggregator],
        portfolio_id: str = "backtest-account",
        max_position_pct: float = 0.35,
    ) -> BacktestResult: ...
```
Executes realistic sequential backtesting. Tracks:
- `equity_curve: List[float]`
- `daily_returns: List[float]`
- `trades: List[BacktestTrade]`
- `rolling_var_95: List[float]`
- `metrics: BacktestMetrics` (Sharpe, Sortino, Calmar, Max Drawdown duration, Win Rate, Profit Factor, Expectancy).

---

## 14. Risk Surveillance & Circuit Breakers (`surveillance.monitor`)

Package: `fintech_algorithmic_trading.surveillance.monitor`

### Class: `RiskSurveillanceMonitor`

```python
class RiskSurveillanceMonitor:
    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        caution_drawdown_pct: float = 0.08,
        halt_drawdown_pct: float = 0.15,
        margin_warning_pct: float = 0.80,
        margin_critical_pct: float = 0.95,
    ): ...
    def register_alert_listener(self, listener: Callable[[RiskAlert], None]) -> None: ...
    def audit_portfolio(self, portfolio: Portfolio) -> SurveillanceReport: ...
    def execute_emergency_kill_switch(self, portfolio: Portfolio, oms: OrderManagementSystem) -> List[ExecutionReport]: ...
    def reset_circuit_breaker(self) -> None: ...
```

---

## 15. Reporting & Export (`reporting.tearsheet`)

Package: `fintech_algorithmic_trading.reporting.tearsheet`

### Class: `RiskReportGenerator`
- `generate_markdown_tearsheet(portfolio, var_results=None, monte_carlo_res=None, stress_results=None, compliance_report=None) -> str`
- `generate_backtest_tearsheet(backtest_result: BacktestResult) -> str`

### Class: `DataExporter`
- `export_portfolio_positions_json(portfolio: Portfolio) -> str`
- `export_portfolio_positions_csv(portfolio: Portfolio) -> str`
- `export_backtest_trades_csv(trades: List[BacktestTrade]) -> str`
- `export_equity_curve_csv(equity_curve: List[float]) -> str`
- `export_stress_test_csv(results: List[StressTestResult]) -> str`

---

## 16. Repository & Presets (`storage.repository`)

Package: `fintech_algorithmic_trading.storage.repository`

### Class: `PortfolioRepository`
- `get_benchmark_portfolios() -> Dict[str, Portfolio]`
  - Presets: `"global-macro"`, `"tech-momentum"`, `"multi-asset-balanced"`, `"crypto-fx-alpha"`.
- `get_portfolio(portfolio_id: str) -> Optional[Portfolio]`
- `get_stress_scenarios() -> Dict[str, StressScenario]`
  - Scenarios: `"lehman-2008"`, `"covid-2020"`, `"rates-shock-2022"`, `"tech-rout"`, `"crypto-winter"`.
- `get_stress_scenario(scenario_id: str) -> Optional[StressScenario]`
- `generate_synthetic_bars(symbol: str, n_bars: int = 252, start_price: float = 100.0, annual_drift: float = 0.08, annual_vol: float = 0.20, seed: Optional[int] = 42) -> List[Bar]`
- `generate_synthetic_returns(n_days: int = 750, mean_daily: float = 0.0003, vol_daily: float = 0.012, fat_tailed: bool = True, seed: Optional[int] = 42) -> np.ndarray`
