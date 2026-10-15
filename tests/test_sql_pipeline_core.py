"""Comprehensive unit and integration test suite for Stage 3 Core Logic: PostgreSQL SQL export pipeline."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from fintech_algorithmic_trading import (
    BacktestTrade,
    CleanPositionRecord,
    CleanTradeRecord,
    DataExporter,
    ExportType,
    ExportValidationRule,
    OrderSide,
    PortfolioRepository,
    PostgreSQLExportPipeline,
    RawPositionExport,
    RawTradeExport,
)

# ============================================================================
# 1. Trade Batch Processing & Core Validation Logic
# ============================================================================


def test_process_trades_clean_batch():
    """Verify clean batch of raw trades transforms completely into CleanTradeRecord objects."""
    raw_trades: list[dict[str, Any] | RawTradeExport] = [
        {
            "trade_id": "T-100",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 150.0,
            "exit_price": 155.0,
            "entry_time": "2026-03-01T09:30:00",
            "exit_time": "2026-03-01T15:00:00",
            "pnl": 500.0,
            "return_pct": 0.0333,
            "commission": 2.0,
            "holding_period_bars": 12,
        },
        RawTradeExport(
            trade_id="T-101",
            symbol="MSFT",
            side="LONG",
            quantity=50.0,
            entry_price=400.0,
            exit_price=410.0,
            entry_time=datetime(2026, 3, 2, 9, 30),
            exit_time=datetime(2026, 3, 2, 16, 0),
            pnl=500.0,
            return_pct=0.025,
            commission=1.5,
            holding_period_bars=15,
        ),
    ]

    clean_trades, summary = PostgreSQLExportPipeline.process_trades(raw_trades)

    assert len(clean_trades) == 2
    assert all(isinstance(t, CleanTradeRecord) for t in clean_trades)
    assert clean_trades[0].trade_id == "T-100"
    assert clean_trades[1].trade_id == "T-101"

    assert summary.export_type == ExportType.TRADES
    assert summary.total_raw_records == 2
    assert summary.valid_clean_records == 2
    assert summary.quarantined_records == 0
    assert summary.duplicate_records == 0
    assert summary.is_clean is True
    assert len(summary.errors) == 0


def test_process_trades_dirty_quarantine():
    """Verify invalid trade records are quarantined and logged in summary errors."""
    raw_trades = [
        # Valid trade
        {
            "trade_id": "T-GOOD",
            "symbol": "NVDA",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 120.0,
            "exit_price": 125.0,
            "entry_time": "2026-03-01T10:00:00",
            "exit_time": "2026-03-01T11:00:00",
        },
        # Missing trade_id and empty symbol
        {
            "trade_id": "",
            "symbol": "",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 100.0,
            "exit_price": 110.0,
            "entry_time": "2026-03-01T10:00:00",
            "exit_time": "2026-03-01T11:00:00",
        },
        # Temporal violation (exit < entry) and negative quantity
        {
            "trade_id": "T-BAD-TIME",
            "symbol": "TSLA",
            "side": "SELL",
            "quantity": -5,
            "entry_price": 200.0,
            "exit_price": 190.0,
            "entry_time": "2026-03-02T15:00:00",
            "exit_time": "2026-03-02T09:30:00",
            "commission": -5.0,
        },
    ]

    clean_trades, summary = PostgreSQLExportPipeline.process_trades(raw_trades)

    assert len(clean_trades) == 1
    assert clean_trades[0].trade_id == "T-GOOD"

    assert summary.total_raw_records == 3
    assert summary.valid_clean_records == 1
    assert summary.quarantined_records == 2
    assert summary.is_clean is False

    rule_names = [e.rule_name for e in summary.errors]
    assert ExportValidationRule.NOT_NULL_TRADE_ID.value in rule_names
    assert ExportValidationRule.NOT_NULL_SYMBOL.value in rule_names
    assert ExportValidationRule.POSITIVE_QUANTITY.value in rule_names
    assert ExportValidationRule.TEMPORAL_ORDER_VIOLATION.value in rule_names
    assert ExportValidationRule.NON_NEGATIVE_COMMISSION.value in rule_names


def test_process_trades_deduplication_upsert():
    """Verify duplicate trade_ids trigger UPSERT update and duplicate tracking."""
    raw_trades = [
        {
            "trade_id": "T-DUP",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 500.0,
            "exit_price": 505.0,
            "entry_time": "2026-03-01T09:30:00",
            "exit_time": "2026-03-01T10:00:00",
            "pnl": 500.0,
        },
        # Duplicate with updated exit price and PnL
        {
            "trade_id": "T-DUP",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 500.0,
            "exit_price": 510.0,
            "entry_time": "2026-03-01T09:30:00",
            "exit_time": "2026-03-01T10:00:00",
            "pnl": 1000.0,
        },
    ]

    clean_trades, summary = PostgreSQLExportPipeline.process_trades(raw_trades, deduplicate=True)

    assert len(clean_trades) == 1
    assert clean_trades[0].trade_id == "T-DUP"
    assert clean_trades[0].exit_price == 510.0  # UPSERT updated
    assert clean_trades[0].pnl == 1000.0

    assert summary.total_raw_records == 2
    assert summary.valid_clean_records == 1
    assert summary.duplicate_records == 1
    assert any(e.rule_name == ExportValidationRule.DUPLICATE_TRADE_ID.value for e in summary.errors)


# ============================================================================
# 2. Position Batch Processing & Core Validation Logic
# ============================================================================


def test_process_positions_clean_and_dirty_batch():
    """Verify position batch processing with validation, quarantine, and composite key UPSERT."""
    raw_positions = [
        # Valid position
        {
            "portfolio_id": "PORT-ALPHA",
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "EQUITY",
            "side": "LONG",
            "quantity": 100,
            "entry_price": 180.0,
            "current_price": 190.0,
            "market_value": 19000.0,
            "unrealized_pnl": 1000.0,
            "unrealized_pnl_pct": 0.0556,
        },
        # Duplicate position (same portfolio, symbol, side) with updated current_price
        {
            "portfolio_id": "PORT-ALPHA",
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "EQUITY",
            "side": "LONG",
            "quantity": 100,
            "entry_price": 180.0,
            "current_price": 195.0,
            "market_value": 19500.0,
            "unrealized_pnl": 1500.0,
            "unrealized_pnl_pct": 0.0833,
        },
        # Dirty position (missing portfolio_id and negative current_price)
        RawPositionExport(
            portfolio_id=None,
            symbol="BAD",
            side="FLAT",
            quantity=0.0,
            current_price=-10.0,
            market_value=None,
        ),
    ]

    clean_pos, summary = PostgreSQLExportPipeline.process_positions(raw_positions)

    assert len(clean_pos) == 1
    assert clean_pos[0].portfolio_id == "PORT-ALPHA"
    assert clean_pos[0].symbol == "AAPL"
    assert clean_pos[0].current_price == 195.0

    assert summary.export_type == ExportType.POSITIONS
    assert summary.total_raw_records == 3
    assert summary.valid_clean_records == 1
    assert summary.quarantined_records == 1
    assert summary.duplicate_records == 1

    rule_names = [e.rule_name for e in summary.errors]
    assert ExportValidationRule.NOT_NULL_PORTFOLIO_ID.value in rule_names
    assert ExportValidationRule.INVALID_POSITION_SIDE.value in rule_names


# ============================================================================
# 3. Equity Curve Batch Processing & Core Validation Logic
# ============================================================================


def test_process_equity_curve_batch():
    """Verify equity curve batch validation, deduplication on (portfolio_id, timestamp), and bounds checking."""
    raw_equity = [
        {
            "portfolio_id": "PORT-BETA",
            "timestamp": "2026-03-01T00:00:00",
            "portfolio_value": 100000.0,
            "cash": 20000.0,
            "drawdown": 0.0,
            "drawdown_pct": 0.0,
        },
        # Duplicate timestamp for same portfolio
        {
            "portfolio_id": "PORT-BETA",
            "timestamp": "2026-03-01T00:00:00",
            "portfolio_value": 102000.0,
            "cash": 22000.0,
            "drawdown": 500.0,
            "drawdown_pct": 0.005,
        },
        # Dirty: negative portfolio_value, missing cash, and invalid drawdown_pct > 1.0
        {
            "portfolio_id": "PORT-BETA",
            "timestamp": "2026-03-02T00:00:00",
            "portfolio_value": -500.0,
            "cash": None,
            "drawdown": -10.0,
            "drawdown_pct": 1.5,
        },
    ]

    clean_eq, summary = PostgreSQLExportPipeline.process_equity_curve(raw_equity)

    assert len(clean_eq) == 1
    assert clean_eq[0].portfolio_id == "PORT-BETA"
    assert clean_eq[0].portfolio_value == 102000.0

    assert summary.export_type == ExportType.EQUITY_CURVE
    assert summary.total_raw_records == 3
    assert summary.valid_clean_records == 1
    assert summary.quarantined_records == 1
    assert summary.duplicate_records == 1

    rule_names = [e.rule_name for e in summary.errors]
    assert ExportValidationRule.NON_NEGATIVE_PORTFOLIO_VALUE.value in rule_names
    assert ExportValidationRule.NOT_NULL_CASH.value in rule_names
    assert ExportValidationRule.INVALID_DRAWDOWN_PCT.value in rule_names


# ============================================================================
# 4. CSV Ingestion and Unified Pipeline Dispatcher
# ============================================================================


def test_process_csv_trades_integration():
    """Verify DataExporter backtest trades CSV ingestion into PostgreSQLExportPipeline."""
    trades = [
        BacktestTrade(
            trade_id="TR-CSV-1",
            symbol="AMD",
            side=OrderSide.BUY,
            quantity=150.0,
            entry_price=160.0,
            exit_price=170.0,
            entry_time=datetime(2026, 3, 10, 9, 30),
            exit_time=datetime(2026, 3, 10, 15, 30),
            pnl=1500.0,
            return_pct=0.0625,
            commission=3.0,
            holding_period_bars=10,
        )
    ]
    csv_text = DataExporter.export_backtest_trades_csv(trades)

    clean_records, summary = PostgreSQLExportPipeline.process_csv(csv_text, ExportType.TRADES)

    assert len(clean_records) == 1
    assert isinstance(clean_records[0], CleanTradeRecord)
    assert clean_records[0].trade_id == "TR-CSV-1"
    assert clean_records[0].symbol == "AMD"
    assert summary.is_clean is True


def test_process_csv_positions_with_default_portfolio():
    """Verify position CSV parsing with default portfolio injection."""
    portfolio = PortfolioRepository.get_benchmark_portfolios()["global-macro"]
    csv_text = DataExporter.export_portfolio_positions_csv(portfolio)

    clean_records, summary = PostgreSQLExportPipeline.process_csv(
        csv_text,
        export_type=ExportType.POSITIONS,
        default_portfolio_id="BENCHMARK-GM",
    )

    assert len(clean_records) == len(portfolio.positions)
    assert all(isinstance(p, CleanPositionRecord) for p in clean_records)
    assert all(p.portfolio_id == "BENCHMARK-GM" for p in clean_records)
    assert summary.is_clean is True


def test_process_export_unified_dispatcher():
    """Verify unified process_export dispatcher handles enum and string types correctly."""
    trade_batch = [
        {
            "trade_id": "T-DISPATCH",
            "symbol": "GOOGL",
            "side": "BUY",
            "quantity": 25,
            "entry_price": 175.0,
            "exit_price": 180.0,
            "entry_time": "2026-03-01T10:00:00",
            "exit_time": "2026-03-01T11:00:00",
        }
    ]

    # Via Enum
    records1, summary1 = PostgreSQLExportPipeline.process_export(ExportType.TRADES, trade_batch)
    assert len(records1) == 1

    # Via string
    records2, summary2 = PostgreSQLExportPipeline.process_export("trades", trade_batch)
    assert len(records2) == 1
    assert summary2.export_type == ExportType.TRADES

    # Invalid export type
    with pytest.raises(ValueError):
        PostgreSQLExportPipeline.process_export("invalid_type", trade_batch)


# ============================================================================
# 5. PostgreSQL Bulk Staging SQL Generation & Database Adapter
# ============================================================================


def test_generate_staging_insert_sql():
    """Verify SQL statement generation for bulk staging ingestion with proper escaping."""
    trade_records = [
        {
            "trade_id": "TRD-O'CONNOR",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 150.5,
            "exit_price": 155.0,
            "entry_time": datetime(2026, 3, 1, 9, 30),
            "exit_time": datetime(2026, 3, 1, 16, 0),
            "pnl": 450.0,
            "return_pct": 0.03,
            "commission": 1.5,
            "holding_period_bars": 8,
        }
    ]

    sql = PostgreSQLExportPipeline.generate_staging_insert_sql(ExportType.TRADES, trade_records)

    assert "INSERT INTO raw_trades_export" in sql
    assert "'TRD-O''CONNOR'" in sql  # Single quote escaped
    assert "'AAPL'" in sql
    assert "150.5" in sql
    assert "VALUES" in sql

    # Empty records
    assert PostgreSQLExportPipeline.generate_staging_insert_sql(ExportType.TRADES, []) == ""


def test_execute_db_pipeline_adapter_mock():
    """Verify execute_db_pipeline orchestrates DDL, insert, and transformation via DB cursor."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = [
        ("trades", "BAD-TRD", "POSITIVE_QUANTITY", "quantity must be strictly positive")
    ]

    records = [
        {"trade_id": "BAD-TRD", "symbol": "AAPL", "quantity": -10}
    ]

    summary = PostgreSQLExportPipeline.execute_db_pipeline(
        mock_conn, ExportType.TRADES, records, init_schema=True
    )

    assert summary.export_type == ExportType.TRADES
    assert summary.total_raw_records == 1
    assert summary.quarantined_records == 1
    assert len(summary.errors) == 1
    assert summary.errors[0].rule_name == "POSITIVE_QUANTITY"

    # Verify SQL calls made on cursor
    mock_cursor.execute.assert_any_call(PostgreSQLExportPipeline.get_schema_ddl())
    mock_cursor.execute.assert_any_call(PostgreSQLExportPipeline.get_clean_validate_sql())
    mock_cursor.close.assert_called_once()
