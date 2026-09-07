"""Domain models and type definitions for algorithmic trading risk management."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class AssetClass(str, Enum):
    """Supported financial asset classes."""

    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    COMMODITY = "COMMODITY"
    FX = "FX"
    CRYPTO = "CRYPTO"
    DERIVATIVE = "DERIVATIVE"


class OrderSide(str, Enum):
    """Trading order sides."""

    BUY = "BUY"
    SELL = "SELL"


class PositionType(str, Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class VaRMethod(str, Enum):
    """Value-at-Risk computation methodology."""

    HISTORICAL = "HISTORICAL"
    PARAMETRIC = "PARAMETRIC"
    CORNISH_FISHER = "CORNISH_FISHER"
    MONTE_CARLO = "MONTE_CARLO"
    EXTREME_VALUE_THEORY = "EXTREME_VALUE_THEORY"


class DriftModel(str, Enum):
    """Stochastic process drift and jump dynamics for Monte Carlo."""

    GBM = "GBM"  # Geometric Brownian Motion
    JUMP_DIFFUSION = "JUMP_DIFFUSION"  # Merton Jump Diffusion
    MEAN_REVERTING = "MEAN_REVERTING"  # Ornstein-Uhlenbeck style


class OptionType(str, Enum):
    """Option contract type."""

    CALL = "CALL"
    PUT = "PUT"


class Asset(BaseModel):
    """Tradable financial asset definition."""

    symbol: str
    name: str
    asset_class: AssetClass
    sector: Optional[str] = "General"
    currency: str = "USD"
    current_price: float = Field(gt=0, description="Current spot price")
    volatility_annual: float = Field(
        ge=0, description="Annualized historical volatility (e.g. 0.20 = 20%)"
    )
    beta_to_market: float = Field(
        default=1.0, description="Asset beta relative to market benchmark"
    )
    dividend_yield: float = Field(default=0.0, ge=0, description="Annual dividend or yield rate")


class Position(BaseModel):
    """A position held in a trading portfolio."""

    asset: Asset
    quantity: float = Field(gt=0, description="Units held")
    entry_price: float = Field(gt=0, description="Average entry cost")
    current_price: float = Field(gt=0, description="Current market price")
    side: PositionType = PositionType.LONG

    @property
    def market_value(self) -> float:
        """Total current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total original cost basis."""
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit or loss in base currency."""
        if self.side == PositionType.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized percentage return."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


class Portfolio(BaseModel):
    """Institutional multi-asset portfolio."""

    id: str
    name: str
    description: str
    cash: float = Field(ge=0, default=0.0, description="Unencumbered cash balance")
    positions: List[Position] = Field(default_factory=list)
    base_currency: str = "USD"

    @property
    def total_positions_value(self) -> float:
        """Sum of market values of all open positions."""
        return sum(pos.market_value for pos in self.positions)

    @property
    def net_asset_value(self) -> float:
        """Total NAV (cash + long positions - short liabilities)."""
        long_val = sum(pos.market_value for pos in self.positions if pos.side == PositionType.LONG)
        short_val = sum(
            pos.market_value for pos in self.positions if pos.side == PositionType.SHORT
        )
        return self.cash + long_val - short_val

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute market value across all positions."""
        return sum(pos.market_value for pos in self.positions)

    @property
    def net_exposure(self) -> float:
        """Net directional market exposure (long - short)."""
        long_val = sum(pos.market_value for pos in self.positions if pos.side == PositionType.LONG)
        short_val = sum(
            pos.market_value for pos in self.positions if pos.side == PositionType.SHORT
        )
        return long_val - short_val

    @property
    def leverage(self) -> float:
        """Gross leverage ratio (gross exposure / NAV)."""
        nav = self.net_asset_value
        if nav <= 0:
            return 0.0
        return self.gross_exposure / nav

    @property
    def asset_weights(self) -> Dict[str, float]:
        """Portfolio weight distribution by asset symbol."""
        total = self.gross_exposure
        if total <= 0:
            return {}
        return {pos.asset.symbol: pos.market_value / total for pos in self.positions}

    @property
    def sector_weights(self) -> Dict[str, float]:
        """Portfolio weight distribution by industry/sector."""
        total = self.gross_exposure
        if total <= 0:
            return {}
        sectors: Dict[str, float] = {}
        for pos in self.positions:
            sector = pos.asset.sector or "General"
            sectors[sector] = sectors.get(sector, 0.0) + pos.market_value
        return {sec: val / total for sec, val in sectors.items()}


class VaRResult(BaseModel):
    """Value-at-Risk and Expected Shortfall calculation output."""

    method: VaRMethod
    confidence_level: float = Field(description="e.g. 0.95 or 0.99")
    horizon_days: int = Field(ge=1, description="Holding period in trading days")
    var_amount: float = Field(ge=0, description="Estimated maximum loss in currency units")
    var_pct: float = Field(ge=0, description="VaR as fraction of portfolio value")
    cvar_amount: float = Field(ge=0, description="Conditional VaR / Expected Shortfall in currency")
    cvar_pct: float = Field(ge=0, description="CVaR as fraction of portfolio value")
    portfolio_value: float = Field(gt=0)
    skewness: Optional[float] = None
    excess_kurtosis: Optional[float] = None
    computation_time_ms: float = 0.0


class MonteCarloConfig(BaseModel):
    """Configuration parameters for stochastic multi-asset simulation."""

    n_simulations: int = Field(default=5000, ge=100, le=100000)
    horizon_days: int = Field(default=21, ge=1, le=1260)
    time_steps: int = Field(default=21, ge=1, le=1260)
    drift_model: DriftModel = DriftModel.GBM
    random_seed: Optional[int] = 42
    risk_free_rate: float = Field(default=0.045, description="Annual risk-free interest rate")
    jump_intensity: float = Field(default=0.10, ge=0.0, description="Annual Poisson jump frequency")
    jump_mean: float = Field(default=-0.05, description="Mean jump size percentage")
    jump_std: float = Field(default=0.08, gt=0, description="Standard deviation of jump magnitude")
    mean_reversion_speed: float = Field(
        default=0.5, ge=0, description="Ornstein-Uhlenbeck mean reversion rate"
    )


class MonteCarloResult(BaseModel):
    """Vectorized Monte Carlo path simulation results."""

    config: MonteCarloConfig
    initial_portfolio_value: float
    mean_terminal_value: float
    median_terminal_value: float
    min_terminal_value: float
    max_terminal_value: float
    p1_terminal_value: float
    p5_terminal_value: float
    p50_terminal_value: float
    p95_terminal_value: float
    p99_terminal_value: float
    simulated_var_95_amount: float
    simulated_var_95_pct: float
    simulated_cvar_95_amount: float
    simulated_cvar_95_pct: float
    simulated_var_99_amount: float
    simulated_var_99_pct: float
    simulated_cvar_99_amount: float
    simulated_cvar_99_pct: float
    max_drawdown_mean_pct: float
    max_drawdown_p95_pct: float
    probability_of_loss: float
    computation_time_ms: float = 0.0


class StressScenario(BaseModel):
    """Macro crisis or custom scenario shock profile."""

    id: str
    name: str
    description: str
    equity_shock_pct: float = 0.0
    fixed_income_shock_pct: float = 0.0
    rates_shock_bps: float = 0.0
    fx_shock_pct: float = 0.0
    commodity_shock_pct: float = 0.0
    crypto_shock_pct: float = 0.0
    volatility_multiplier: float = 1.0


class StressTestResult(BaseModel):
    """Stress test simulation impact breakdown."""

    scenario: StressScenario
    portfolio_value_before: float
    portfolio_value_after: float
    absolute_impact: float
    percentage_impact: float
    asset_impacts: Dict[str, float]
    capital_adequacy_passed: bool
    cushion_remaining_pct: float


class RiskLimits(BaseModel):
    """Institutional trading and risk mandate limits."""

    max_leverage: float = Field(default=3.0, gt=0)
    max_single_position_pct: float = Field(default=0.30, gt=0, le=1.0)
    max_sector_pct: float = Field(default=0.45, gt=0, le=1.0)
    max_var_95_pct: float = Field(default=0.06, gt=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1.0)
    min_cash_buffer_pct: float = Field(default=0.05, ge=0.0, le=1.0)


class RiskViolation(BaseModel):
    """Specific policy violation record."""

    rule: str
    limit_value: float
    current_value: float
    severity: str = "CRITICAL"  # WARNING, CRITICAL, BREACH
    message: str


class RiskComplianceReport(BaseModel):
    """Audit report of portfolio against risk limits."""

    portfolio_id: str
    portfolio_name: str
    passed: bool
    violations: List[RiskViolation] = Field(default_factory=list)
    current_leverage: float
    max_single_position_symbol: str
    max_single_position_pct: float
    max_sector_name: str
    max_sector_pct: float
    estimated_var_95_pct: float
    cash_buffer_pct: float


class OptionGreeks(BaseModel):
    """Option sensitivity analysis (Black-Scholes analytical)."""

    option_type: OptionType
    spot_price: float
    strike_price: float
    time_to_expiry_years: float
    volatility: float
    risk_free_rate: float
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class OptimizationObjective(str, Enum):
    """Portfolio optimization target function."""

    MAX_SHARPE = "MAX_SHARPE"
    MIN_VARIANCE = "MIN_VARIANCE"
    RISK_PARITY = "RISK_PARITY"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"


class OptimizationResult(BaseModel):
    """Portfolio optimization output."""

    objective: OptimizationObjective
    weights: Dict[str, float]
    expected_return_annual: float
    volatility_annual: float
    sharpe_ratio: float
    risk_contributions: Dict[str, float] = Field(default_factory=dict)
    computation_time_ms: float = 0.0


class KellyCriterionMode(str, Enum):
    """Kelly position sizing computation mode."""

    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"


class KellySizingResult(BaseModel):
    """Kelly criterion sizing outcome and risk metrics."""

    mode: KellyCriterionMode
    full_kelly_fraction: float
    half_kelly_fraction: float
    quarter_kelly_fraction: float
    recommended_fraction: float
    recommended_position_size: float
    expected_growth_rate: float
    max_drawdown_risk_pct: float


class EVTResult(BaseModel):
    """Peaks-Over-Threshold Extreme Value Theory VaR output."""

    confidence_level: float
    horizon_days: int
    threshold_u: float
    exceedance_count: int
    shape_parameter_xi: float
    scale_parameter_beta: float
    var_amount: float
    var_pct: float
    cvar_amount: float
    cvar_pct: float
    portfolio_value: float
    computation_time_ms: float = 0.0


class OptionContract(BaseModel):
    """Individual option position definition for derivatives portfolio."""

    symbol: str
    underlying_symbol: str
    option_type: OptionType
    spot_price: float = Field(gt=0)
    strike_price: float = Field(gt=0)
    time_to_expiry_years: float = Field(gt=0)
    volatility: float = Field(gt=0)
    quantity: float = Field(description="Contracts held; positive for long, negative for short")
    multiplier: float = Field(default=100.0, gt=0, description="Underlying shares per contract")


class PortfolioGreeks(BaseModel):
    """Aggregate multi-leg derivatives portfolio Greeks."""

    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    net_rho: float
    total_market_value: float


class DeltaGammaHedge(BaseModel):
    """Calculated hedge allocation to neutralize Delta and Gamma."""

    underlying_symbol: str
    target_net_delta: float = 0.0
    target_net_gamma: float = 0.0
    underlying_shares_needed: float
    hedge_option_symbol: Optional[str] = None
    hedge_option_contracts_needed: float = 0.0
    post_hedge_delta: float = 0.0
    post_hedge_gamma: float = 0.0


# ============================================================================
# Subsystem Models: Market Data, Execution, Strategy, Backtesting, Surveillance
# ============================================================================


class OrderType(str, Enum):
    """Execution order classification."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TWAP = "TWAP"
    VWAP = "VWAP"


class OrderStatus(str, Enum):
    """Order lifecycle execution status."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class TimeInForce(str, Enum):
    """Order validity and expiration policy."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Good for Day


class Bar(BaseModel):
    """Standard OHLCV market candlestick bar."""

    symbol: str
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def high_ge_low(self) -> Bar:
        if self.high < self.low:
            raise ValueError(f"High {self.high} cannot be less than low {self.low}")
        return self


class Quote(BaseModel):
    """Top-of-book market quote."""

    symbol: str
    timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    bid_size: float = Field(default=100.0, ge=0)
    ask_size: float = Field(default=100.0, ge=0)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10000.0 if self.mid_price > 0 else 0.0


class OrderBookLevel(BaseModel):
    """Individual price level depth on order book."""

    price: float = Field(gt=0)
    size: float = Field(gt=0)
    order_count: int = Field(default=1, ge=1)


class OrderBookDepth(BaseModel):
    """Aggregated Level 2 depth snapshot."""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def total_bid_depth(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(level.size for level in self.asks)


class Order(BaseModel):
    """Institutional trade order definition."""

    id: str = Field(default_factory=lambda: f"ord-{uuid.uuid4().hex[:10]}")
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(default=None, gt=0)
    stop_price: Optional[float] = Field(default=None, gt=0)
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    rejection_reason: Optional[str] = None


class Fill(BaseModel):
    """Individual trade execution fill report."""

    fill_id: str = Field(default_factory=lambda: f"fill-{uuid.uuid4().hex[:10]}")
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExecutionReport(BaseModel):
    """Consolidated execution response for order routing."""

    order_id: str
    symbol: str
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float
    total_commission: float
    total_slippage: float
    fills: List[Fill] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rejection_reason: Optional[str] = None


class SignalDirection(str, Enum):
    """Alpha trading signal stance."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class StrategySignal(BaseModel):
    """Quantitative alpha generation signal."""

    strategy_id: str
    symbol: str
    direction: SignalDirection
    strength: float = Field(default=1.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    target_weight: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rationale: str = ""
    metadata: Dict[str, float] = Field(default_factory=dict)


class TargetAllocation(BaseModel):
    """Target portfolio position allocation from strategy composite."""

    symbol: str
    target_weight: float
    target_shares: float
    current_shares: float
    shares_delta: float
    reason: str = ""


class BacktestConfig(BaseModel):
    """Configuration parameter set for historical backtesting."""

    initial_capital: float = Field(default=1_000_000.0, gt=0)
    commission_bps: float = Field(default=5.0, ge=0)  # 5 bps = 0.05%
    slippage_bps: float = Field(default=2.5, ge=0)  # 2.5 bps
    risk_free_rate: float = Field(default=0.045, ge=0)
    annualization_factor: int = Field(default=252, gt=0)


class BacktestTrade(BaseModel):
    """Completed round-trip or historical position trade in backtest."""

    trade_id: str = Field(default_factory=lambda: f"trade-{uuid.uuid4().hex[:8]}")
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    commission: float
    holding_period_bars: int


class BacktestMetrics(BaseModel):
    """Institutional performance and risk analytics from backtest."""

    total_return_pct: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    profitable_trades: int
    loss_making_trades: int


class BacktestResult(BaseModel):
    """Full backtest execution record and time series."""

    config: BacktestConfig
    initial_capital: float
    final_capital: float
    equity_curve: List[float]
    daily_returns: List[float]
    trades: List[BacktestTrade]
    metrics: BacktestMetrics
    rolling_var_95: List[float] = Field(default_factory=list)
    computation_time_ms: float = 0.0


class AlertSeverity(str, Enum):
    """Risk surveillance alert classification."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY_KILL_SWITCH = "EMERGENCY_KILL_SWITCH"


class RiskAlert(BaseModel):
    """Real-time risk surveillance notification event."""

    alert_id: str = Field(default_factory=lambda: f"alert-{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: AlertSeverity
    component: str
    rule: str
    message: str
    current_value: float
    threshold_value: float
    action_taken: Optional[str] = None


class CircuitBreakerState(str, Enum):
    """Trading system risk gating state."""

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    HALTED = "HALTED"


class SurveillanceReport(BaseModel):
    """Consolidated real-time risk surveillance audit snapshot."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio_id: str
    state: CircuitBreakerState
    active_alerts: List[RiskAlert] = Field(default_factory=list)
    current_drawdown_pct: float
    margin_utilization_pct: float
    leverage: float
    kill_switch_active: bool
    remedial_instructions: List[str] = Field(default_factory=list)
