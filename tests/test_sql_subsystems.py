"""Comprehensive unit and integration test suite for Stage 4 Subsystems: PostgreSQL SQL export pipeline integration."""

from datetime import datetime, timedelta

import pytest

from fintech_algorithmic_trading import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    Bar,
    CleanEquityRecord,
    CleanPositionRecord,
    CleanTradeRecord,
    DataExporter,
    ExportType,
    OrderManagementSystem,
    OrderSide,
    Portfolio,
    PortfolioRepository,
    PositionType,
    PostgreSQLExportPipeline,
    TrendFollowingMACDStrategy,
)


def _generate_synthetic_bars(symbol: str = "AAPL", count: int = 60) -> list[Bar]:
    bars: list[Bar] = []
    base_time = datetime(2026, 1, 1, 9, 30)
    price = 150.0
    for i in range(count):
        price += 0.5 * (1 if i % 2 == 0 else -0.3)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=5 * i),
                open=round(price - 0.2, 2),
                high=round(price + 0.8, 2),
                low=round(price - 0.5, 2),
                close=round(price, 2),
                volume=10000.0 + i * 100,
            )
        )
    return bars


def test_data_exporter_export_and_clean_positions_benchmark():
    """Verify DataExporter exports portfolio positions into PostgreSQLExportPipeline."""
    portfolio = PortfolioRepository.get_benchmark_portfolios()["global-macro"]
    clean_records, summary = DataExporter.export_and_clean_positions(portfolio)

    assert len(clean_records) == len(portfolio.positions)
    assert all(isinstance(p, CleanPositionRecord) for p in clean_records)
    assert summary.is_clean is True
    assert summary.total_raw_records == len(portfolio.positions)
    assert summary.quarantined_records == 0


def test_data_exporter_export_and_clean_trades_from_backtest():
    """Verify DataExporter exports backtest trades into PostgreSQLExportPipeline."""
    bars = _generate_synthetic_bars("AAPL", count=60)
    config = BacktestConfig(initial_capital=100000.0, commission_bps=5.0)
    engine = BacktestEngine(config=config, strategy=TrendFollowingMACDStrategy(fast_period=5, slow_period=15, signal_period=5))
    result = engine.run(bars)

    clean_records, summary = DataExporter.export_and_clean_trades(result.trades)
    assert len(clean_records) == len(result.trades)
    assert all(isinstance(t, CleanTradeRecord) for t in clean_records)
    assert summary.quarantined_records == 0
    assert summary.is_clean is True


def test_data_exporter_export_and_clean_equity_curve():
    """Verify DataExporter exports backtest equity curve and verifies CleanEquityRecords."""
    bars = _generate_synthetic_bars("AAPL", count=40)
    config = BacktestConfig(initial_capital=50000.0)
    engine = BacktestEngine(config=config, strategy=TrendFollowingMACDStrategy(fast_period=4, slow_period=10, signal_period=4))
    result = engine.run(bars)

    clean_records, summary = DataExporter.export_and_clean_equity_curve(
        result, portfolio_id="MACD-ALPHA-1"
    )
    assert len(clean_records) == len(result.equity_curve)
    assert all(isinstance(eq, CleanEquityRecord) for eq in clean_records)
    assert all(eq.portfolio_id == "MACD-ALPHA-1" for eq in clean_records)
    assert summary.is_clean is True
    assert clean_records[0].portfolio_value == 50000.0


def test_data_exporter_export_to_sql_staging_domain_objects():
    """Verify DataExporter generates valid bulk SQL statements from rich domain models."""
    portfolio = PortfolioRepository.get_benchmark_portfolios()["global-macro"]
    sql_positions = DataExporter.export_to_sql_staging(ExportType.POSITIONS, portfolio)
    assert "INSERT INTO raw_positions_export" in sql_positions
    assert "SPY" in sql_positions
    assert "global-macro" in sql_positions

    trades = [
        BacktestTrade(
            trade_id="TR-STAGE-1",
            symbol="NVDA",
            side=OrderSide.BUY,
            quantity=50.0,
            entry_price=120.0,
            exit_price=130.0,
            entry_time=datetime(2026, 3, 1, 10, 0),
            exit_time=datetime(2026, 3, 1, 15, 0),
            pnl=500.0,
            return_pct=0.0833,
            commission=2.0,
            holding_period_bars=10,
        )
    ]
    sql_trades = DataExporter.export_to_sql_staging(ExportType.TRADES, trades)
    assert "INSERT INTO raw_trades_export" in sql_trades
    assert "TR-STAGE-1" in sql_trades
    assert "NVDA" in sql_trades


def test_data_exporter_export_pipeline_staging_script():
    """Verify DataExporter constructs comprehensive SQL staging bundle."""
    portfolio = PortfolioRepository.get_benchmark_portfolios()["global-macro"]
    bars = _generate_synthetic_bars("AAPL", count=30)
    config = BacktestConfig(initial_capital=50000.0)
    engine = BacktestEngine(config=config, strategy=TrendFollowingMACDStrategy())
    result = engine.run(bars)

    script = DataExporter.export_pipeline_staging_script(
        portfolio=portfolio,
        backtest_result=result,
        portfolio_id="SCRIPT-TEST-PORT",
    )
    assert "CREATE TABLE IF NOT EXISTS raw_positions_export" in script
    assert "CREATE TABLE IF NOT EXISTS raw_trades_export" in script
    assert "INSERT INTO raw_positions_export" in script
    assert "INSERT INTO raw_trades_export" in script
    assert "INSERT INTO raw_equity_curve_export" in script
    assert "clean_and_validate_exports()" in script


def test_pipeline_reconstruct_portfolio_from_clean_records():
    """Verify PostgreSQLExportPipeline reconstructs live Portfolio from clean records."""
    portfolio_orig = PortfolioRepository.get_benchmark_portfolios()["global-macro"]
    clean_records, _ = DataExporter.export_and_clean_positions(portfolio_orig)

    reconstructed = PostgreSQLExportPipeline.reconstruct_portfolio(
        portfolio_id="RECON-GM",
        clean_positions=clean_records,
        cash=10000.0,
    )
    assert isinstance(reconstructed, Portfolio)
    assert reconstructed.id == "RECON-GM"
    assert reconstructed.cash == 10000.0
    assert len(reconstructed.positions) == len(portfolio_orig.positions)

    spy_pos = next(p for p in reconstructed.positions if p.asset.symbol == "SPY")
    assert spy_pos.quantity == 1000
    assert spy_pos.current_price == 540.0
    assert spy_pos.side == PositionType.LONG


def test_pipeline_reconstruct_trades_from_clean_records():
    """Verify PostgreSQLExportPipeline reconstructs BacktestTrade models from clean records."""
    trades_orig = [
        BacktestTrade(
            trade_id="TR-REC-1",
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=200.0,
            entry_price=400.0,
            exit_price=420.0,
            entry_time=datetime(2026, 3, 1, 9, 30),
            exit_time=datetime(2026, 3, 1, 16, 0),
            pnl=4000.0,
            return_pct=0.05,
            commission=5.0,
            holding_period_bars=24,
        )
    ]
    clean_records, _ = DataExporter.export_and_clean_trades(trades_orig)
    reconstructed_trades = PostgreSQLExportPipeline.reconstruct_trades(clean_records)

    assert len(reconstructed_trades) == 1
    t = reconstructed_trades[0]
    assert isinstance(t, BacktestTrade)
    assert t.trade_id == "TR-REC-1"
    assert t.symbol == "MSFT"
    assert t.side == OrderSide.BUY
    assert t.quantity == 200.0
    assert t.pnl == 4000.0


def test_subsystem_backtest_to_sql_pipeline_end_to_end():
    """End-to-end integration: Strategy -> BacktestEngine -> DataExporter -> SQL Pipeline."""
    bars = _generate_synthetic_bars("AAPL", count=70)
    config = BacktestConfig(initial_capital=100000.0, commission_bps=5.0)
    engine = BacktestEngine(config=config, strategy=TrendFollowingMACDStrategy())
    result = engine.run(bars)

    # 1. Clean trades
    clean_trades, trade_sum = DataExporter.export_and_clean_trades(result.trades)
    assert trade_sum.is_clean is True

    # 2. Clean equity curve
    clean_equity, eq_sum = DataExporter.export_and_clean_equity_curve(
        result, portfolio_id="MACD-AUTO"
    )
    assert eq_sum.is_clean is True
    assert len(clean_equity) == len(result.equity_curve)

    # 3. Generate Staging SQL
    sql_staging = DataExporter.export_to_sql_staging(ExportType.TRADES, result)
    assert "INSERT INTO raw_trades_export" in sql_staging


def test_subsystem_oms_to_sql_export_integration():
    """Verify OMS order lifecycle updates portfolio, which successfully exports and validates."""
    oms = OrderManagementSystem()
    portfolio = Portfolio(
        id="OMS-PORT-1",
        name="OMS Dynamic Test Portfolio",
        description="Testing OMS export integration",
        cash=100000.0,
        positions=[],
    )

    # Place and execute market buy
    order = oms.create_order(symbol="NVDA", side=OrderSide.BUY, quantity=100.0)
    report = oms.execute_order(order=order, portfolio=portfolio, market_price=120.0)
    assert report.status.value == "FILLED"
    assert len(portfolio.positions) == 1

    clean_records, summary = DataExporter.export_and_clean_positions(portfolio)
    assert summary.is_clean is True
    assert len(clean_records) == 1
    assert clean_records[0].symbol == "NVDA"
    assert clean_records[0].quantity == 100.0


def test_subsystem_quarantine_dirty_domain_exports():
    """Verify corrupted records within subsystem exports are quarantined by the pipeline."""
    dirty_trades = [
        BacktestTrade(
            trade_id="TR-CORRUPT",
            symbol="",  # Empty symbol should violate SYMBOL_EMPTY
            side=OrderSide.BUY,
            quantity=-50.0,  # Negative quantity violates QUANTITY_NON_POSITIVE
            entry_price=100.0,
            exit_price=105.0,
            entry_time=datetime(2026, 3, 2, 10, 0),
            exit_time=datetime(2026, 3, 1, 10, 0),  # Exit before entry
            pnl=250.0,
            return_pct=0.05,
            commission=1.0,
            holding_period_bars=5,
        )
    ]
    clean_records, summary = DataExporter.export_and_clean_trades(dirty_trades)
    assert len(clean_records) == 0
    assert summary.quarantined_records == 1
    assert summary.is_clean is False
    assert len(summary.errors) >= 3
    rule_names = [e.rule_name for e in summary.errors]
    assert any("SYMBOL" in r for r in rule_names)
    assert any("QUANTITY" in r for r in rule_names)
    assert any("TIMESTAMP" in r or "TIME" in r for r in rule_names)
