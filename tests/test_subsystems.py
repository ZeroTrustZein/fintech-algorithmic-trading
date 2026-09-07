"""Comprehensive unit and integration test suite for Stage 4 Subsystems."""

from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from fintech_algorithmic_trading import (
    BacktestConfig,
    BacktestEngine,
    Bar,
    CircuitBreakerState,
    CompositeAlphaAggregator,
    DataExporter,
    MarketDataFeed,
    MeanRevertingBollingerStrategy,
    OrderManagementSystem,
    OrderSide,
    OrderStatus,
    OrderType,
    PairsStatArbStrategy,
    Portfolio,
    PositionType,
    PreTradeRiskGatekeeper,
    RiskReportGenerator,
    RiskSurveillanceMonitor,
    SignalDirection,
    SlippageModel,
    TrendFollowingMACDStrategy,
    VolatilityBreakoutStrategy,
)
from fintech_algorithmic_trading.cli.main import cli

# ============================================================================
# 1. Market Data Feed & Order Book Tests
# ============================================================================


def test_market_data_feed_add_and_get_bars():
    feed = MarketDataFeed()
    t0 = datetime(2026, 1, 1, 9, 30)
    bar1 = Bar(
        symbol="SPY", timestamp=t0, open=500.0, high=505.0, low=498.0, close=502.0, volume=10000.0
    )
    bar2 = Bar(
        symbol="SPY",
        timestamp=t0 + timedelta(minutes=5),
        open=502.0,
        high=506.0,
        low=501.0,
        close=504.0,
        volume=12000.0,
    )

    feed.add_bar(bar1)
    feed.add_bar(bar2)

    bars = feed.get_bars("SPY")
    assert len(bars) == 2
    assert feed.get_latest_bar("SPY") == bar2
    px = feed.get_latest_price("SPY")
    assert px is not None
    assert abs(px - 504.0) < 0.2


def test_market_data_feed_subscribers():
    feed = MarketDataFeed()
    received_bars = []
    received_quotes = []

    feed.subscribe_bar("AAPL", lambda b: received_bars.append(b))
    feed.subscribe_quote("AAPL", lambda q: received_quotes.append(q))

    bar = Bar(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        open=200.0,
        high=202.0,
        low=199.0,
        close=201.0,
        volume=5000.0,
    )
    feed.add_bar(bar)

    assert len(received_bars) == 1
    assert received_bars[0].symbol == "AAPL"
    assert len(received_quotes) == 1
    assert received_quotes[0].mid_price > 0.0


def test_market_data_order_book_depth():
    feed = MarketDataFeed()
    feed.generate_synthetic_history("MSFT", start_price=400.0, n_bars=10)
    ob = feed.generate_order_book("MSFT", levels=5)

    assert ob.symbol == "MSFT"
    assert len(ob.bids) == 5
    assert len(ob.asks) == 5
    assert ob.best_bid is not None
    assert ob.best_ask is not None
    assert ob.best_bid <= ob.best_ask
    assert ob.total_bid_depth > 0.0
    assert ob.total_ask_depth > 0.0


def test_market_data_resample_bars():
    feed = MarketDataFeed()
    feed.generate_synthetic_history("NVDA", start_price=120.0, n_bars=20, freq_minutes=5)
    resampled = feed.resample_bars("NVDA", aggregation_factor=4)

    assert len(resampled) == 5
    assert resampled[0].volume > 0
    assert resampled[0].high >= resampled[0].low


# ============================================================================
# 2. Slippage & Order Management System Tests
# ============================================================================


def test_slippage_models():
    # Fixed bps
    buy_px = SlippageModel.calculate_fixed_bps_slippage(
        price=100.0, side=OrderSide.BUY, slippage_bps=10.0
    )
    assert buy_px == 100.10

    sell_px = SlippageModel.calculate_fixed_bps_slippage(
        price=100.0, side=OrderSide.SELL, slippage_bps=10.0
    )
    assert sell_px == 99.90

    # Market impact square-root
    exec_px, slip = SlippageModel.calculate_market_impact_slippage(
        price=100.0, side=OrderSide.BUY, quantity=10_000.0, daily_volume=1_000_000.0
    )
    assert exec_px > 100.0
    assert slip > 0.0


def test_pre_trade_risk_gatekeeper_notional_rejection(simple_portfolio: Portfolio):
    gatekeeper = PreTradeRiskGatekeeper(max_order_notional=50_000.0)
    oms = OrderManagementSystem(gatekeeper=gatekeeper)

    # Order of $100k > $50k
    order = oms.create_order(symbol="SPY", side=OrderSide.BUY, quantity=1000.0)
    rep = oms.execute_order(order=order, portfolio=simple_portfolio, market_price=100.0)

    assert rep.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)
    assert rep.filled_quantity <= 500.0  # Scaled down to within $50k


def test_pre_trade_risk_gatekeeper_cash_shortage(simple_portfolio: Portfolio):
    simple_portfolio.cash = 100.0
    gatekeeper = PreTradeRiskGatekeeper()
    oms = OrderManagementSystem(gatekeeper=gatekeeper)

    order = oms.create_order(symbol="SPY", side=OrderSide.BUY, quantity=1000.0)
    rep = oms.execute_order(order=order, portfolio=simple_portfolio, market_price=500.0)

    assert rep.status == OrderStatus.REJECTED
    assert "Insufficient unencumbered cash" in (rep.rejection_reason or "")


def test_oms_limit_order_execution():
    oms = OrderManagementSystem()
    port = Portfolio(id="test-p", name="Test", description="", cash=100_000.0, positions=[])

    # Buy Limit at $95, market is at $100 -> Pending (no fill)
    limit_buy = oms.create_order(
        symbol="XYZ", side=OrderSide.BUY, quantity=100.0, order_type=OrderType.LIMIT, price=95.0
    )
    rep1 = oms.execute_order(limit_buy, port, market_price=100.0)
    assert rep1.status == OrderStatus.PENDING

    # Market drops to $94 -> Fills
    rep2 = oms.execute_order(limit_buy, port, market_price=94.0)
    assert rep2.status == OrderStatus.FILLED
    assert len(port.positions) == 1
    assert port.positions[0].asset.symbol == "XYZ"


def test_oms_position_lifecycle_long_to_short():
    oms = OrderManagementSystem()
    port = Portfolio(id="p-life", name="Life", description="", cash=50_000.0, positions=[])

    # 1. Buy 100 shares @ $100
    buy_ord = oms.create_order(symbol="ABC", side=OrderSide.BUY, quantity=100.0)
    oms.execute_order(buy_ord, port, market_price=100.0)
    pos = port.positions[0]
    assert pos.quantity == 100.0
    assert pos.side == PositionType.LONG

    # 2. Sell 150 shares @ $110 -> closes 100 long, flips to 50 short
    sell_ord = oms.create_order(symbol="ABC", side=OrderSide.SELL, quantity=150.0)
    oms.execute_order(sell_ord, port, market_price=110.0)
    assert len(port.positions) == 1
    new_pos = port.positions[0]
    assert new_pos.quantity == 50.0
    assert new_pos.side == PositionType.SHORT


# ============================================================================
# 3. Strategy Signals & Composite Aggregator Tests
# ============================================================================


def test_macd_strategy():
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history("SPY", start_price=500.0, n_bars=50, drift_annual=0.25)
    strat = TrendFollowingMACDStrategy()

    sig = strat.generate_signal("SPY", bars)
    assert sig is not None
    assert sig.symbol == "SPY"
    assert sig.direction in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)
    assert -1.0 <= sig.strength <= 1.0


def test_bollinger_strategy():
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history(
        "TLT", start_price=90.0, n_bars=35, volatility_annual=0.15
    )
    strat = MeanRevertingBollingerStrategy(window=20, num_std=2.0)

    sig = strat.generate_signal("TLT", bars)
    assert sig is not None
    assert "z_score" in sig.metadata


def test_volatility_breakout_strategy():
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history("GLD", start_price=200.0, n_bars=30)
    strat = VolatilityBreakoutStrategy(lookback_period=15, atr_period=10)

    sig = strat.generate_signal("GLD", bars)
    assert sig is not None
    assert "atr" in sig.metadata


def test_pairs_statarb_strategy():
    feed = MarketDataFeed()
    bars_a = feed.generate_synthetic_history("SPY", start_price=500.0, n_bars=40, seed=1)
    bars_b = feed.generate_synthetic_history("QQQ", start_price=450.0, n_bars=40, seed=2)
    strat = PairsStatArbStrategy(symbol_a="SPY", symbol_b="QQQ", lookback_period=25)

    sig_a, sig_b = strat.generate_pair_signals(bars_a, bars_b)
    assert sig_a is not None
    assert sig_b is not None
    assert sig_a.symbol == "SPY"
    assert sig_b.symbol == "QQQ"


def test_composite_alpha_aggregator():
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history("NVDA", start_price=120.0, n_bars=45)
    agg = CompositeAlphaAggregator()

    sig = agg.evaluate_composite_signal("NVDA", bars)
    assert sig.strategy_id == "composite-ensemble"
    assert -1.0 <= sig.strength <= 1.0
    assert 0.0 <= sig.confidence <= 1.0


# ============================================================================
# 4. Backtest Engine Tests
# ============================================================================


def test_backtest_engine_run():
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history("SPY", start_price=500.0, n_bars=60, drift_annual=0.15)
    cfg = BacktestConfig(initial_capital=100_000.0, commission_bps=5.0, slippage_bps=2.5)
    engine = BacktestEngine(config=cfg)

    res = engine.run(symbol="SPY", bars=bars, strategy=TrendFollowingMACDStrategy())

    assert res.initial_capital == 100_000.0
    assert len(res.equity_curve) == 60
    assert len(res.daily_returns) == 59
    assert res.metrics.total_return_pct is not None
    assert res.metrics.max_drawdown_pct >= 0.0
    assert res.metrics.annualized_volatility >= 0.0
    assert res.computation_time_ms > 0.0


def test_backtest_engine_insufficient_bars():
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history("SPY", start_price=500.0, n_bars=5)
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="requires at least 10 bars"):
        engine.run(symbol="SPY", bars=bars, strategy=TrendFollowingMACDStrategy())


# ============================================================================
# 5. Risk Surveillance & Circuit Breaker Tests
# ============================================================================


def test_risk_surveillance_normal_audit(simple_portfolio: Portfolio):
    monitor = RiskSurveillanceMonitor()
    report = monitor.audit_portfolio(simple_portfolio)

    assert report.state in (CircuitBreakerState.NORMAL, CircuitBreakerState.CAUTION)
    assert report.portfolio_id == simple_portfolio.id
    assert not report.kill_switch_active


def test_risk_surveillance_kill_switch_trigger(simple_portfolio: Portfolio):
    monitor = RiskSurveillanceMonitor(halt_drawdown_pct=0.10)
    oms = OrderManagementSystem()

    # Simulate portfolio having peaked at 2x current NAV -> 50% drawdown
    monitor._peak_nav_records[simple_portfolio.id] = simple_portfolio.net_asset_value * 2.0
    report = monitor.audit_portfolio(simple_portfolio)

    assert report.state == CircuitBreakerState.HALTED
    assert report.kill_switch_active

    # Execute emergency liquidation
    liquidations = monitor.trigger_emergency_kill_switch(simple_portfolio, oms)
    assert len(liquidations) > 0
    assert len(simple_portfolio.positions) == 0  # Fully flattened to cash


def test_risk_surveillance_alert_listeners(simple_portfolio: Portfolio):
    monitor = RiskSurveillanceMonitor()
    alerts = []
    monitor.register_alert_listener(lambda a: alerts.append(a))

    # Trigger alert via excessive leverage limit
    monitor.limits.max_leverage = 0.5  # Forces leverage violation
    monitor.audit_portfolio(simple_portfolio)

    assert len(alerts) > 0
    assert any(a.rule == "LEVERAGE_CEILING_BREACH" for a in alerts)


# ============================================================================
# 6. Reporting & Exporter Tests
# ============================================================================


def test_risk_report_generator(simple_portfolio: Portfolio):
    tearsheet = RiskReportGenerator.generate_markdown_tearsheet(portfolio=simple_portfolio)
    assert simple_portfolio.name in tearsheet
    assert "Executive Capital & Leverage Summary" in tearsheet
    assert "Asset Allocation & Position Ledger" in tearsheet


def test_data_exporter_json_and_csv(simple_portfolio: Portfolio):
    # JSON
    json_str = DataExporter.export_portfolio_json(simple_portfolio)
    assert simple_portfolio.id in json_str

    # CSV Positions
    csv_str = DataExporter.export_portfolio_positions_csv(simple_portfolio)
    assert "symbol,name,asset_class" in csv_str
    assert "AAA" in csv_str

    # CSV Trades & Equity Curve
    feed = MarketDataFeed()
    bars = feed.generate_synthetic_history("SPY", start_price=500.0, n_bars=30)
    engine = BacktestEngine()
    b_res = engine.run(symbol="SPY", bars=bars, strategy=TrendFollowingMACDStrategy())

    trades_csv = DataExporter.export_backtest_trades_csv(b_res.trades)
    assert "trade_id,symbol,side" in trades_csv

    eq_csv = DataExporter.export_equity_curve_csv(b_res)
    assert "step,nav,daily_return,rolling_var_95" in eq_csv


# ============================================================================
# 7. CLI Subsystem Integration Tests
# ============================================================================


def test_cli_backtest_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["backtest", "--symbol", "SPY", "--bars", "50", "--strategy", "macd"])
    assert res.exit_code == 0
    assert "Backtest Performance Attribution" in res.output


def test_cli_surveillance_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["surveillance", "--portfolio", "tech-momentum"])
    assert res.exit_code == 0
    assert "Real-Time Risk Surveillance Monitor" in res.output


def test_cli_orderbook_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["orderbook", "--symbol", "NVDA", "--levels", "3"])
    assert res.exit_code == 0
    assert "Level 2 Order Book Depth" in res.output


def test_cli_report_command(tmp_path):
    runner = CliRunner()
    report_file = tmp_path / "test_report.md"
    res = runner.invoke(
        cli, ["report", "--portfolio", "multi-asset-balanced", "--output", str(report_file)]
    )
    assert res.exit_code == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Institutional Portfolio Risk Tearsheet" in content
