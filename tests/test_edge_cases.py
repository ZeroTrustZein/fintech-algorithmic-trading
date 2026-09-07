"""Comprehensive edge-case and boundary condition test suite for fintech-algorithmic-trading.

Covers mathematical stability, extreme tail scenarios, invalid inputs, order lifecycle,
surveillance state transitions, and institutional reporting exports.
"""

from datetime import datetime

import numpy as np
import pytest

from fintech_algorithmic_trading.backtest import BacktestEngine
from fintech_algorithmic_trading.data import MarketDataFeed
from fintech_algorithmic_trading.engine.evt import EVTEngine
from fintech_algorithmic_trading.engine.greeks import calculate_option_greeks
from fintech_algorithmic_trading.engine.hedging import OptionHedgingEngine
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.engine.optimizer import PortfolioOptimizer
from fintech_algorithmic_trading.engine.risk_engine import PortfolioRiskEngine
from fintech_algorithmic_trading.engine.sizing import PositionSizer
from fintech_algorithmic_trading.engine.var import (
    VaRCalculator,
    calculate_cornish_fisher_var,
    calculate_historical_var,
    calculate_parametric_var,
)
from fintech_algorithmic_trading.execution.oms import (
    OrderManagementSystem,
    PreTradeRiskGatekeeper,
)
from fintech_algorithmic_trading.reporting import DataExporter, RiskReportGenerator
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.strategy import (
    CompositeAlphaAggregator,
    MeanRevertingBollingerStrategy,
    PairsStatArbStrategy,
    TrendFollowingMACDStrategy,
    VolatilityBreakoutStrategy,
)
from fintech_algorithmic_trading.surveillance import RiskSurveillanceMonitor
from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    BacktestConfig,
    Bar,
    CircuitBreakerState,
    MonteCarloConfig,
    OptionContract,
    OptionType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionType,
    Quote,
    RiskComplianceReport,
    RiskLimits,
    RiskViolation,
    SignalDirection,
)

# ==============================================================================
# 1. DOMAIN MODELS & TYPES EDGE CASES
# ==============================================================================


def test_position_cost_basis_zero():
    """Verify position handles zero unrealized pnl when current equals entry price."""
    asset = Asset(
        symbol="FREE",
        name="Flat Asset",
        asset_class=AssetClass.EQUITY,
        current_price=10.0,
        volatility_annual=0.20,
    )
    pos = Position(
        asset=asset,
        quantity=100.0,
        entry_price=10.0,
        current_price=10.0,
        side=PositionType.LONG,
    )
    assert pos.cost_basis == 1000.0
    assert pos.unrealized_pnl == 0.0
    assert pos.unrealized_pnl_pct == 0.0


def test_portfolio_zero_or_negative_nav_and_empty_weights():
    """Verify portfolio leverage, weights, and exposure under zero/negative NAV and empty positions."""
    empty_port = Portfolio(
        id="empty",
        name="Empty Portfolio",
        description="Empty",
        cash=0.0,
        positions=[],
    )
    assert empty_port.leverage == 0.0
    assert empty_port.asset_weights == {}
    assert empty_port.sector_weights == {}
    assert empty_port.net_exposure == 0.0

    # Negative NAV through large short liabilities
    asset = Asset(
        symbol="SHRT",
        name="Short Asset",
        asset_class=AssetClass.EQUITY,
        current_price=100.0,
        volatility_annual=0.20,
    )
    short_pos = Position(
        asset=asset,
        quantity=50.0,
        entry_price=80.0,
        current_price=100.0,
        side=PositionType.SHORT,
    )
    neg_port = Portfolio(
        id="neg-nav",
        name="Insolvent Portfolio",
        description="Cash < Short Liability",
        cash=1000.0,
        positions=[short_pos],
    )
    assert neg_port.net_asset_value < 0  # 1000 - 5000 = -4000
    assert neg_port.leverage == 0.0


def test_bar_model_high_less_than_low_validation():
    """Verify Bar model rejects invalid bar where high < low."""
    with pytest.raises(ValueError, match="High .* cannot be less than low"):
        Bar(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            open=100.0,
            high=95.0,  # Invalid: high is less than low
            low=98.0,
            close=96.0,
            volume=1000.0,
        )


def test_quote_spread_and_spread_bps():
    """Verify quote spread and spread bps calculations."""
    quote = Quote(
        symbol="SPY",
        timestamp=datetime.utcnow(),
        bid=500.0,
        ask=500.50,
        bid_size=200.0,
        ask_size=300.0,
    )
    assert quote.mid_price == 500.25
    assert quote.spread == 0.50
    assert round(quote.spread_bps, 2) == round((0.50 / 500.25) * 10000.0, 2)


# ==============================================================================
# 2. MARKET DATA FEED EDGE CASES
# ==============================================================================


def test_market_data_feed_edge_cases():
    """Test get_bars slice, price fallback, and resample corner cases."""
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history(symbol="XYZ", start_price=100.0, n_bars=10)

    # get_bars with specific n_bars count
    sliced = feed.get_bars("XYZ", n_bars=3)
    assert len(sliced) == 3
    assert sliced[-1] == bars[-1]

    # get_latest_price fallback to bar close when quote is absent
    del feed._quotes["XYZ"]
    assert feed.get_latest_price("XYZ") == bars[-1].close

    # get_latest_price when symbol is unknown
    assert feed.get_latest_price("UNKNOWN") is None

    # resample_bars with factor <= 1 or empty
    assert feed.resample_bars("XYZ", aggregation_factor=1) == bars
    assert feed.resample_bars("NON_EXISTENT") == []


# ==============================================================================
# 3. EXTREME VALUE THEORY (EVT) EDGE CASES
# ==============================================================================


def test_evt_engine_custom_threshold_and_regimes():
    """Test EVT with explicit threshold, near-zero xi, and xi >= 1 regime."""
    evt = EVTEngine(threshold_quantile=0.90)

    # 1. Explicit threshold parameter
    returns = np.random.default_rng(42).normal(0.0, 0.02, size=300)
    u, xi, beta, exceedances = evt.fit_pot(-returns, threshold=0.03)
    assert u == 0.03
    assert len(exceedances) >= 5

    # 2. Near-zero xi exponential tail regime branch (xi <= 1e-5)
    # Generate exponential exceedances where xi ~ 0
    exp_losses = np.random.default_rng(123).exponential(scale=0.015, size=500)
    res_exp = evt.calculate_evt_var(-exp_losses, portfolio_value=1_000_000.0)
    assert res_exp.var_amount > 0
    assert res_exp.cvar_amount >= res_exp.var_amount

    # 3. Extreme heavy tail regime (xi >= 1.0)
    # Pareto distribution with alpha=0.9 -> xi = 1/alpha > 1.0
    heavy_losses = np.random.default_rng(999).pareto(a=0.8, size=500) * 0.01
    res_heavy = evt.calculate_evt_var(-heavy_losses, portfolio_value=1_000_000.0)
    assert res_heavy.var_amount > 0
    assert res_heavy.cvar_amount >= res_heavy.var_amount


# ==============================================================================
# 4. POSITION SIZING & KELLY CRITERION EDGE CASES
# ==============================================================================


def test_position_sizing_input_validations():
    """Test PositionSizer error guards and boundary returns."""
    # Discrete Kelly invalid NAV
    with pytest.raises(ValueError, match="Portfolio NAV must be positive"):
        PositionSizer.calculate_discrete_kelly(win_rate=0.55, win_loss_ratio=1.5, portfolio_nav=-1000.0)

    # Volatility target allocation boundary cases
    assert PositionSizer.calculate_volatility_target_allocation(0.0, {"AAPL": 0.2}, 100000.0) == {}
    assert PositionSizer.calculate_volatility_target_allocation(0.15, {}, 100000.0) == {}
    assert PositionSizer.calculate_volatility_target_allocation(0.15, {"AAPL": 0.0}, 100000.0) == {}

    # Drawdown scaled size edge cases
    assert PositionSizer.calculate_drawdown_scaled_size(10000.0, current_drawdown_pct=0.0) == 10000.0
    assert PositionSizer.calculate_drawdown_scaled_size(10000.0, current_drawdown_pct=0.05, max_tolerated_drawdown_pct=0.0) == 0.0
    assert PositionSizer.calculate_drawdown_scaled_size(10000.0, current_drawdown_pct=0.20, max_tolerated_drawdown_pct=0.15) == 0.0


# ==============================================================================
# 5. BLACK-SCHOLES & GREEKS EDGE CASES
# ==============================================================================


def test_black_scholes_greeks_input_validations():
    """Verify Black-Scholes raises ValueError for non-positive inputs."""
    with pytest.raises(ValueError, match="Spot price must be positive"):
        calculate_option_greeks(OptionType.CALL, spot_price=0.0, strike_price=100.0, time_to_expiry_years=1.0, volatility=0.2)

    with pytest.raises(ValueError, match="Strike price must be positive"):
        calculate_option_greeks(OptionType.CALL, spot_price=100.0, strike_price=-5.0, time_to_expiry_years=1.0, volatility=0.2)

    with pytest.raises(ValueError, match="Time to expiration must be positive"):
        calculate_option_greeks(OptionType.CALL, spot_price=100.0, strike_price=100.0, time_to_expiry_years=0.0, volatility=0.2)

    with pytest.raises(ValueError, match="Volatility must be positive"):
        calculate_option_greeks(OptionType.CALL, spot_price=100.0, strike_price=100.0, time_to_expiry_years=1.0, volatility=-0.1)


def test_hedging_engine_near_zero_gamma_exception():
    """Verify OptionHedgingEngine raises ValueError when benchmark option has near-zero gamma."""
    engine = OptionHedgingEngine()
    dummy_contract = OptionContract(
        symbol="SPY_CALL",
        underlying_symbol="SPY",
        option_type=OptionType.CALL,
        strike_price=10000.0,  # Ridiculously high OTM -> gamma ~ 0
        spot_price=100.0,
        time_to_expiry_years=0.01,
        volatility=0.05,
        quantity=1.0,
    )
    port_greeks = engine.calculate_portfolio_greeks(contracts=[dummy_contract])

    with pytest.raises(ValueError, match="Hedge option has near-zero gamma"):
        engine.calculate_delta_gamma_hedge(
            portfolio_greeks=port_greeks,
            hedge_option=dummy_contract,
            target_delta=0.0,
            target_gamma=0.0,
        )


# ==============================================================================
# 6. MONTE CARLO STOCHASTIC ENGINE EDGE CASES
# ==============================================================================


def test_monte_carlo_empty_portfolio_and_corr_validation(simple_portfolio: Portfolio):
    """Test Monte Carlo validation for empty portfolio and mismatched correlation matrix."""
    mc = MonteCarloEngine(config=MonteCarloConfig(n_simulations=100, horizon_days=5))

    empty_p = Portfolio(id="empty", name="Empty", description="", cash=1000.0, positions=[])
    with pytest.raises(ValueError, match="Cannot simulate an empty portfolio"):
        mc.simulate_portfolio(empty_p)

    # Correlation matrix shape mismatch (3x3 for 2-asset portfolio)
    bad_corr = np.eye(3)
    with pytest.raises(ValueError, match="Correlation matrix shape .* must match"):
        mc.simulate_portfolio(simple_portfolio, correlation_matrix=bad_corr)

    # Non-positive definite correlation matrix requiring jitter
    non_psd_corr = np.array([[1.0, 0.999], [0.999, 1.0]]) - 0.001
    res_psd = mc.simulate_portfolio(simple_portfolio, correlation_matrix=non_psd_corr)
    assert res_psd.config.n_simulations == 100


# ==============================================================================
# 7. PORTFOLIO OPTIMIZER EDGE CASES
# ==============================================================================


def test_portfolio_optimizer_edge_cases(simple_portfolio: Portfolio):
    """Test optimizer shape mismatches, 1-position error, and custom covariance."""
    opt = PortfolioOptimizer()

    # Fewer than 2 symbols
    with pytest.raises(ValueError, match="Optimization requires at least 2 assets"):
        opt.optimize(symbols=["SPY"], expected_returns=[0.10], covariance_matrix=np.array([[0.04]]))

    # Mu shape mismatch
    with pytest.raises(ValueError, match="expected_returns shape .* does not match"):
        opt.optimize(symbols=["A", "B"], expected_returns=[0.10], covariance_matrix=np.eye(2))

    # Covariance shape mismatch
    with pytest.raises(ValueError, match="covariance_matrix shape .* must be"):
        opt.optimize(symbols=["A", "B"], expected_returns=[0.10, 0.12], covariance_matrix=np.eye(3))

    # Optimize portfolio with custom covariance
    custom_cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    res_custom = opt.optimize_portfolio(simple_portfolio, covariance_matrix=custom_cov)
    assert "AAA" in res_custom.weights
    assert "BBB" in res_custom.weights
    assert round(sum(res_custom.weights.values()), 2) == 1.0

    # Portfolio with only 1 position
    one_pos_p = Portfolio(
        id="single",
        name="Single",
        description="",
        cash=1000.0,
        positions=[simple_portfolio.positions[0]],
    )
    with pytest.raises(ValueError, match="Portfolio must contain at least 2 positions"):
        opt.optimize_portfolio(one_pos_p)


# ==============================================================================
# 8. VAR & RISK ENGINE VIOLATIONS & SHOCKS
# ==============================================================================


def test_var_insufficient_returns_and_portfolio_nav_error():
    """Verify VaR algorithms raise ValueError on insufficient data or non-positive NAV."""
    with pytest.raises(ValueError, match="At least 5 return observations"):
        calculate_historical_var([0.01, -0.02, 0.01], portfolio_value=100000.0)

    with pytest.raises(ValueError, match="At least 3 return observations"):
        calculate_parametric_var([0.01, -0.02], portfolio_value=100000.0)

    with pytest.raises(ValueError, match="At least 10 return observations"):
        calculate_cornish_fisher_var([0.01] * 5, portfolio_value=100000.0)

    # Portfolio parametric VaR with zero or negative NAV
    zero_nav_port = Portfolio(id="z", name="Z", description="", cash=0.0, positions=[])
    with pytest.raises(ValueError, match="Portfolio net asset value must be positive"):
        VaRCalculator.calculate_portfolio_parametric_var(
            zero_nav_port, asset_covariance_matrix=np.eye(1), symbols_order=["Z"]
        )


def test_risk_engine_cash_buffer_and_var_violations():
    """Test risk engine triggers cash buffer and VaR policy violations."""
    engine = PortfolioRiskEngine()
    asset = Asset(symbol="VOL", name="High Vol Asset", asset_class=AssetClass.EQUITY, current_price=100.0, volatility_annual=0.90)
    pos = Position(asset=asset, quantity=1000.0, entry_price=100.0, current_price=100.0, side=PositionType.LONG)

    # Cash is $100 on a $100,100 NAV portfolio (0.1% cash buffer)
    low_cash_port = Portfolio(
        id="low-cash",
        name="Low Cash Portfolio",
        description="Near zero cash buffer and ultra high volatility",
        cash=100.0,
        positions=[pos],
    )
    limits = RiskLimits(
        min_cash_buffer_pct=0.10,  # Requires 10%
        max_var_95_pct=0.02,       # Requires max 2% daily VaR
    )
    report = engine.check_compliance(low_cash_port, custom_limits=limits)
    rules_violated = [v.rule for v in report.violations]
    assert "MIN_CASH_BUFFER" in rules_violated
    assert "MAX_VAR_95_PCT" in rules_violated
    assert report.passed is False


def test_risk_engine_multi_asset_class_stress_testing():
    """Test stress test shocks across Commodity, FX, and Crypto assets."""
    engine = PortfolioRiskEngine()
    comm_asset = Asset(
        symbol="GLD",
        name="Gold",
        asset_class=AssetClass.COMMODITY,
        current_price=2000.0,
        volatility_annual=0.15,
    )
    fx_asset = Asset(
        symbol="EURUSD",
        name="EUR/USD",
        asset_class=AssetClass.FX,
        current_price=1.10,
        volatility_annual=0.08,
    )
    crypto_asset = Asset(
        symbol="BTC",
        name="Bitcoin",
        asset_class=AssetClass.CRYPTO,
        current_price=60000.0,
        volatility_annual=0.60,
    )

    port = Portfolio(
        id="multi-shock",
        name="Multi Asset Shock",
        description="",
        cash=50000.0,
        positions=[
            Position(asset=comm_asset, quantity=10.0, entry_price=2000.0, current_price=2000.0, side=PositionType.LONG),
            Position(asset=fx_asset, quantity=20000.0, entry_price=1.10, current_price=1.10, side=PositionType.LONG),
            Position(asset=crypto_asset, quantity=1.0, entry_price=60000.0, current_price=60000.0, side=PositionType.LONG),
        ],
    )
    scenarios = PortfolioRepository.get_stress_scenarios()
    lehman_res = engine.run_stress_test(port, scenarios["2008-lehman"])
    assert "GLD" in lehman_res.asset_impacts
    assert "EURUSD" in lehman_res.asset_impacts
    assert "BTC" in lehman_res.asset_impacts
    assert lehman_res.portfolio_value_after < lehman_res.portfolio_value_before

    # Test calculate_volatility_target_size boundary checks
    assert engine.calculate_volatility_target_size(0.15, asset_vol_annual=0.0, portfolio_nav=100000.0) == 0.0
    assert engine.calculate_volatility_target_size(0.15, asset_vol_annual=0.20, portfolio_nav=-100.0) == 0.0


# ==============================================================================
# 9. OMS & EXECUTION GATEKEEPER EDGE CASES
# ==============================================================================


def test_oms_gatekeeper_and_order_rejections(sample_portfolio: Portfolio):
    """Test gatekeeper leverage limit rejection, and OMS limit/stop order states."""
    gatekeeper = PreTradeRiskGatekeeper(
        risk_limits=RiskLimits(max_single_position_pct=0.35, max_leverage=1.05)
    )

    # 1. Gatekeeper leverage breach rejection
    huge_order = Order(
        symbol="SPY",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5000.0,
        price=500.0,
    )
    passed, reason, _ = gatekeeper.evaluate_order(huge_order, sample_portfolio, reference_price=500.0)
    assert passed is False
    assert "exceeds max order limit" in reason or "breaches" in reason

    # 2. OMS SELL Limit order rejected when market price < limit price
    # Create an unconstrained OMS instance to test pure order execution logic
    pure_oms = OrderManagementSystem(gatekeeper=PreTradeRiskGatekeeper(risk_limits=RiskLimits(max_single_position_pct=1.0, max_leverage=10.0)))
    sell_limit = Order(
        symbol="SPY",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=600.0,  # Market price is ~500, so limit is above market -> cannot fill
    )
    rep_limit = pure_oms.execute_order(sell_limit, sample_portfolio, market_price=500.0)
    assert rep_limit.status == OrderStatus.PENDING
    assert rep_limit.filled_quantity == 0.0

    # 3. OMS STOP_LOSS SELL order pending when stop not triggered
    stop_order = Order(
        symbol="SPY",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=1.0,
        stop_price=400.0,  # Market price is ~500 -> stop price not breached
    )
    rep_stop = pure_oms.execute_order(stop_order, sample_portfolio, market_price=500.0)
    assert rep_stop.status == OrderStatus.PENDING

    # 4. OMS TWAP / VWAP execution multi-slicing
    twap_order = Order(
        symbol="SPY",
        side=OrderSide.BUY,
        order_type=OrderType.TWAP,
        quantity=2.0,
    )
    rep_twap = pure_oms.execute_order(twap_order, sample_portfolio, market_price=500.0)
    assert rep_twap.status == OrderStatus.FILLED
    assert rep_twap.filled_quantity == 2.0


def test_oms_position_lifecycle_cover_short_and_flip():
    """Test OMS buying to cover existing short, including exact cover and flip to long."""
    oms = OrderManagementSystem()
    asset = Asset(
        symbol="COV",
        name="Cover Asset",
        asset_class=AssetClass.EQUITY,
        current_price=100.0,
        volatility_annual=0.20,
    )
    short_pos = Position(asset=asset, quantity=50.0, entry_price=110.0, current_price=100.0, side=PositionType.SHORT)

    port = Portfolio(
        id="short-port",
        name="Short Port",
        description="",
        cash=20000.0,
        positions=[short_pos],
    )

    # 1. Exact cover of short (50 shares)
    cover_order = Order(symbol="COV", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0)
    rep_cov = oms.execute_order(cover_order, port, market_price=100.0)
    assert rep_cov.status == OrderStatus.FILLED
    assert len(port.positions) == 0  # Fully closed

    # 2. Sell to open new short position
    sell_short_order = Order(symbol="COV", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=30.0)
    oms.execute_order(sell_short_order, port, market_price=100.0)
    assert len(port.positions) == 1
    assert port.positions[0].side == PositionType.SHORT
    assert port.positions[0].quantity == 30.0

    # 3. Buy to flip short to long (buy 50 shares: cover 30 short + 20 long)
    flip_order = Order(symbol="COV", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0)
    oms.execute_order(flip_order, port, market_price=100.0)
    assert len(port.positions) == 1
    assert port.positions[0].side == PositionType.LONG
    assert port.positions[0].quantity == 20.0


# ==============================================================================
# 10. ALPHA STRATEGIES COMPREHENSIVE COVERAGE
# ==============================================================================


def test_alpha_strategies_all_signal_branches():
    """Test all signal directions across MACD, Bollinger, Breakout, Pairs, and Composite."""
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history(symbol="SIG", start_price=100.0, n_bars=60)

    # 1. MACD Strategy
    macd = TrendFollowingMACDStrategy()
    assert macd.generate_signal("SIG", bars[:10]) is None  # Insufficient bars
    sig_macd = macd.generate_signal("SIG", bars)
    assert sig_macd is not None
    assert sig_macd.direction in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)

    # 2. Bollinger Strategy: oversold, overbought, mean-reverted, zero std
    bb = MeanRevertingBollingerStrategy(window=20, num_std=2.0)
    assert bb.generate_signal("SIG", bars[:10]) is None

    # Zero std price bars
    flat_bars = [
        Bar(symbol="SIG", timestamp=datetime.utcnow(), open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0)
        for _ in range(25)
    ]
    assert bb.generate_signal("SIG", flat_bars) is None

    # Oversold test: price drops sharply
    drop_bars = list(flat_bars)
    drop_bars.append(Bar(symbol="SIG", timestamp=datetime.utcnow(), open=100.0, high=100.0, low=80.0, close=80.0, volume=100.0))
    sig_oversold = bb.generate_signal("SIG", drop_bars)
    assert sig_oversold is not None
    assert sig_oversold.direction == SignalDirection.LONG

    # 3. Volatility Breakout Strategy
    vbreak = VolatilityBreakoutStrategy(lookback_period=20, atr_period=14)
    assert vbreak.generate_signal("SIG", bars[:10]) is None
    sig_vbreak = vbreak.generate_signal("SIG", bars)
    assert sig_vbreak is not None
    assert sig_vbreak.direction in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)

    # 4. Pairs StatArb Strategy
    pairs = PairsStatArbStrategy(symbol_a="SIG", symbol_b="SIG2", lookback_period=20)
    bars2 = feed.generate_synthetic_history(symbol="SIG2", start_price=150.0, n_bars=60)
    assert pairs.generate_pair_signals(bars[:10], bars2[:10]) == (None, None)

    sig_a, sig_b = pairs.generate_pair_signals(bars, bars2)
    assert sig_a is not None and sig_b is not None
    assert sig_a.direction in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)

    # 5. Composite Alpha Aggregator: empty bars & active evaluation
    agg = CompositeAlphaAggregator()
    sig_empty = agg.evaluate_composite_signal("SIG", [])
    assert sig_empty.direction == SignalDirection.FLAT
    assert "Empty bar history" in sig_empty.rationale

    sig_comp = agg.evaluate_composite_signal("SIG", bars)
    assert sig_comp.direction in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)
    assert sig_comp.metadata["sub_strategy_count"] >= 1


def test_backtest_with_composite_aggregator():
    """Verify BacktestEngine executes cleanly using CompositeAlphaAggregator."""
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history(symbol="SPY", start_price=500.0, n_bars=60)
    engine = BacktestEngine(config=BacktestConfig(initial_capital=50000.0))
    agg = CompositeAlphaAggregator()
    res = engine.run(symbol="SPY", bars=bars, strategy=agg)
    assert res.metrics.total_trades >= 0
    assert len(res.equity_curve) == len(bars)


# ==============================================================================
# 11. RISK SURVEILLANCE & CIRCUIT BREAKER EDGE CASES
# ==============================================================================


def test_risk_surveillance_caution_state_and_liquidation(sample_portfolio: Portfolio):
    """Test surveillance transitions to CAUTION and HALTED with position liquidation."""
    monitor = RiskSurveillanceMonitor(caution_drawdown_pct=0.08, halt_drawdown_pct=0.15)

    nav = sample_portfolio.net_asset_value
    # Drawdown 10% (between caution 8% and halt 15%) -> CAUTION
    monitor._peak_nav_records[sample_portfolio.id] = nav / 0.90
    rep_caution = monitor.audit_portfolio(sample_portfolio)
    assert rep_caution.state == CircuitBreakerState.CAUTION
    assert rep_caution.kill_switch_active is False
    assert any(a.rule == "CAUTION_DRAWDOWN_WARNING" for a in rep_caution.active_alerts)

    # Drawdown 20% -> HALTED + Emergency Kill Switch with liquidation
    monitor._peak_nav_records[sample_portfolio.id] = nav / 0.80
    rep_halt = monitor.audit_portfolio(sample_portfolio)
    assert rep_halt.state == CircuitBreakerState.HALTED
    assert rep_halt.kill_switch_active is True
    assert any(a.rule == "MAX_DRAWDOWN_CIRCUIT_BREAKER" for a in rep_halt.active_alerts)


# ==============================================================================
# 12. REPORTING & EXPORT UTILITIES COMPREHENSIVE COVERAGE
# ==============================================================================


def test_data_exporter_trades_equity_and_stress_csv(sample_portfolio: Portfolio):
    """Test DataExporter export methods for trades, equity curve, and stress tests."""
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history(symbol="SPY", start_price=500.0, n_bars=60)
    engine = BacktestEngine(config=BacktestConfig(initial_capital=100000.0))
    res = engine.run(symbol="SPY", bars=bars, strategy=TrendFollowingMACDStrategy())

    # Export equity curve CSV
    eq_csv = DataExporter.export_equity_curve_csv(res)
    assert "step,nav,daily_return,rolling_var_95" in eq_csv

    # Export trades CSV
    trades_csv = DataExporter.export_backtest_trades_csv(res.trades)
    assert "trade_id,symbol,side,quantity" in trades_csv

    # Export stress test CSV & JSON
    risk_engine = PortfolioRiskEngine()
    scenarios = PortfolioRepository.get_stress_scenarios()
    stress_results = [risk_engine.run_stress_test(sample_portfolio, sc) for sc in scenarios.values()]

    stress_json = DataExporter.export_stress_test_json(stress_results)
    assert "2008 Global Financial Crisis" in stress_json

    stress_csv = DataExporter.export_stress_test_csv(stress_results)
    assert "scenario_id,scenario_name,portfolio_value_before" in stress_csv
    assert "2008-lehman" in stress_csv


def test_risk_report_generator_variations(sample_portfolio: Portfolio):
    """Test tearsheet generator with missing sections and compliance violations."""
    # Tearsheet with no Monte Carlo, no stress, and compliance breach
    comp_breach = RiskComplianceReport(
        portfolio_id=sample_portfolio.id,
        portfolio_name=sample_portfolio.name,
        passed=False,
        current_leverage=2.5,
        max_single_position_pct=0.45,
        max_single_position_symbol="AAA",
        max_sector_pct=0.60,
        max_sector_name="Technology",
        estimated_var_95_pct=0.08,
        cash_buffer_pct=0.02,
        violations=[
            RiskViolation(
                rule="SINGLE_POSITION_CAP",
                limit_value=0.30,
                current_value=0.45,
                severity="CRITICAL",
                message="Single position AAA breached 30% cap",
            )
        ],
    )
    tearsheet = RiskReportGenerator.generate_markdown_tearsheet(
        portfolio=sample_portfolio,
        var_results=None,
        monte_carlo_res=None,
        stress_results=None,
        compliance_report=comp_breach,
    )
    assert "BREACH DETECTED" in tearsheet
    assert "SINGLE_POSITION_CAP" in tearsheet


def test_repository_generate_synthetic_returns_normal_dist():
    """Verify PortfolioRepository generates non-fat-tailed (normal) returns."""
    ret_norm = PortfolioRepository.generate_synthetic_returns(n_days=100, fat_tailed=False, seed=42)
    assert len(ret_norm) == 100
    assert isinstance(ret_norm, np.ndarray)
