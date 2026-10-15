"""Test suite for PostgreSQL raw export cleaning and validation scaffold (Stage 1)."""

from fintech_algorithmic_trading import PostgreSQLExportPipeline


def test_package_exports_pipeline():
    """Verify PostgreSQLExportPipeline is exported from top-level package."""
    assert PostgreSQLExportPipeline is not None


def test_schema_ddl_structure():
    """Verify PostgreSQL schema DDL contains required staging, clean, and error tables."""
    ddl = PostgreSQLExportPipeline.get_schema_ddl()
    assert isinstance(ddl, str)

    # Staging tables
    assert "raw_trades_export" in ddl
    assert "raw_positions_export" in ddl
    assert "raw_equity_curve_export" in ddl

    # Clean production tables
    assert "clean_trades" in ddl
    assert "clean_positions" in ddl
    assert "clean_equity_curve" in ddl

    # Error logging audit table
    assert "export_validation_errors" in ddl

    # Indices and constraints
    assert "idx_clean_trades_symbol" in ddl
    assert "idx_clean_trades_times" in ddl
    assert "uq_clean_pos" in ddl
    assert "uq_clean_equity" in ddl


def test_clean_validate_sql_rules():
    """Verify clean and validate SQL transformation rules."""
    sql = PostgreSQLExportPipeline.get_clean_validate_sql()
    assert isinstance(sql, str)

    # Core validation rule identifiers
    assert "NOT_NULL_TRADE_ID" in sql
    assert "POSITIVE_QUANTITY" in sql
    assert "POSITIVE_ENTRY_PRICE" in sql
    assert "POSITIVE_EXIT_PRICE" in sql
    assert "TEMPORAL_ORDER_VIOLATION" in sql
    assert "DUPLICATE_TRADE_ID" in sql
    assert "INVALID_SIDE" in sql

    # Position validation rules
    assert "NOT_NULL_PORTFOLIO_ID" in sql
    assert "INVALID_POSITION_SIDE" in sql
    assert "POSITIVE_CURRENT_PRICE" in sql

    # Equity curve validation rules
    assert "NON_NEGATIVE_PORTFOLIO_VALUE" in sql
    assert "INVALID_DRAWDOWN_PCT" in sql

    # UPSERT semantics
    assert "ON CONFLICT (trade_id) DO UPDATE" in sql
    assert "ON CONFLICT (portfolio_id, symbol, side) DO UPDATE" in sql
    assert "ON CONFLICT (portfolio_id, timestamp) DO UPDATE" in sql


def test_generate_copy_command():
    """Verify PostgreSQL COPY command generation for bulk CSV ingestion."""
    cmd = PostgreSQLExportPipeline.generate_copy_command(
        "raw_trades_export", "/var/data/trades_2026.csv"
    )
    assert (
        cmd
        == "COPY raw_trades_export FROM '/var/data/trades_2026.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');"
    )

    # Escaping single quotes in file paths
    cmd_escaped = PostgreSQLExportPipeline.generate_copy_command(
        "raw_trades_export", "C:\\data\\'tricky'\\trades.csv"
    )
    assert "''tricky''" in cmd_escaped


def test_validate_trade_record():
    """Verify trade validation rules for clean vs dirty export data."""
    valid_trade = {
        "trade_id": "TRD-001",
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "entry_price": 150.0,
        "exit_price": 155.0,
        "entry_time": "2026-01-01T09:30:00",
        "exit_time": "2026-01-01T16:00:00",
        "pnl": 500.0,
        "commission": 1.5,
    }
    assert PostgreSQLExportPipeline.validate_trade_record(valid_trade) == []

    dirty_trade = {
        "trade_id": "",
        "symbol": "",
        "side": "HOLD",
        "quantity": -10,
        "entry_price": 0,
        "exit_price": -5,
        "entry_time": "2026-01-02T10:00:00",
        "exit_time": "2026-01-01T10:00:00",
        "commission": -2.0,
    }
    errors = PostgreSQLExportPipeline.validate_trade_record(dirty_trade)
    assert any("NOT_NULL_TRADE_ID" in e for e in errors)
    assert any("NOT_NULL_SYMBOL" in e for e in errors)
    assert any("INVALID_SIDE" in e for e in errors)
    assert any("POSITIVE_QUANTITY" in e for e in errors)
    assert any("POSITIVE_ENTRY_PRICE" in e for e in errors)
    assert any("POSITIVE_EXIT_PRICE" in e for e in errors)
    assert any("TEMPORAL_ORDER_VIOLATION" in e for e in errors)
    assert any("NON_NEGATIVE_COMMISSION" in e for e in errors)


def test_validate_position_record():
    """Verify position validation rules for clean vs dirty export data."""
    valid_pos = {
        "portfolio_id": "PORT-1",
        "symbol": "MSFT",
        "side": "LONG",
        "quantity": 50,
        "entry_price": 400.0,
        "current_price": 420.0,
        "market_value": 21000.0,
    }
    assert PostgreSQLExportPipeline.validate_position_record(valid_pos) == []

    dirty_pos = {
        "portfolio_id": "",
        "symbol": "",
        "side": "FLAT",
        "quantity": 0,
        "current_price": -10.0,
        "market_value": None,
    }
    errors = PostgreSQLExportPipeline.validate_position_record(dirty_pos)
    assert any("NOT_NULL_PORTFOLIO_ID" in e for e in errors)
    assert any("NOT_NULL_SYMBOL" in e for e in errors)
    assert any("INVALID_POSITION_SIDE" in e for e in errors)
    assert any("POSITIVE_QUANTITY" in e for e in errors)
    assert any("POSITIVE_CURRENT_PRICE" in e for e in errors)
    assert any("NOT_NULL_MARKET_VALUE" in e for e in errors)


def test_validate_equity_record():
    """Verify equity curve validation rules for clean vs dirty export data."""
    valid_equity = {
        "portfolio_id": "PORT-1",
        "timestamp": "2026-01-01T16:00:00",
        "portfolio_value": 100000.0,
        "cash": 25000.0,
        "drawdown": 500.0,
        "drawdown_pct": 0.005,
    }
    assert PostgreSQLExportPipeline.validate_equity_record(valid_equity) == []

    dirty_equity = {
        "portfolio_id": "",
        "timestamp": None,
        "portfolio_value": -500.0,
        "cash": None,
        "drawdown": -100.0,
        "drawdown_pct": 1.5,
    }
    errors = PostgreSQLExportPipeline.validate_equity_record(dirty_equity)
    assert any("NOT_NULL_PORTFOLIO_ID" in e for e in errors)
    assert any("NOT_NULL_TIMESTAMP" in e for e in errors)
    assert any("NON_NEGATIVE_PORTFOLIO_VALUE" in e for e in errors)
    assert any("NOT_NULL_CASH" in e for e in errors)
    assert any("NON_NEGATIVE_DRAWDOWN" in e for e in errors)
    assert any("INVALID_DRAWDOWN_PCT" in e for e in errors)
