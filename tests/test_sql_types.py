"""Test suite for PostgreSQL raw export cleaning and validation domain types (Stage 2: Types)."""

from datetime import datetime, timedelta
import pytest
from pydantic import ValidationError

from fintech_algorithmic_trading import (
    CleanEquityRecord,
    CleanPositionRecord,
    CleanTradeRecord,
    ExportType,
    ExportValidationError,
    ExportValidationRule,
    ExportValidationSummary,
    PostgreSQLExportPipeline,
    RawEquityCurveExport,
    RawPositionExport,
    RawTradeExport,
)
from fintech_algorithmic_trading.storage import (
    CleanEquityRecord as StorageCleanEquity,
    CleanPositionRecord as StorageCleanPosition,
    CleanTradeRecord as StorageCleanTrade,
    ExportType as StorageExportType,
    ExportValidationError as StorageValidationError,
    ExportValidationRule as StorageValidationRule,
    ExportValidationSummary as StorageValidationSummary,
    RawEquityCurveExport as StorageRawEquity,
    RawPositionExport as StorageRawPosition,
    RawTradeExport as StorageRawTrade,
)


def test_top_level_and_storage_type_exports():
    """Verify all Stage 2 types are properly exported at package and storage levels."""
    assert CleanEquityRecord is StorageCleanEquity
    assert CleanPositionRecord is StorageCleanPosition
    assert CleanTradeRecord is StorageCleanTrade
    assert ExportType is StorageExportType
    assert ExportValidationError is StorageValidationError
    assert ExportValidationRule is StorageValidationRule
    assert ExportValidationSummary is StorageValidationSummary
    assert RawEquityCurveExport is StorageRawEquity
    assert RawPositionExport is StorageRawPosition
    assert RawTradeExport is StorageRawTrade


def test_export_type_enum():
    """Verify ExportType members and string representations."""
    assert ExportType.TRADES == "trades"
    assert ExportType.POSITIONS == "positions"
    assert ExportType.EQUITY_CURVE == "equity_curve"
    assert set(ExportType) == {
        ExportType.TRADES,
        ExportType.POSITIONS,
        ExportType.EQUITY_CURVE,
    }


def test_export_validation_rule_enum():
    """Verify ExportValidationRule enum members."""
    assert ExportValidationRule.NOT_NULL_TRADE_ID == "NOT_NULL_TRADE_ID"
    assert ExportValidationRule.NOT_NULL_SYMBOL == "NOT_NULL_SYMBOL"
    assert ExportValidationRule.INVALID_SIDE == "INVALID_SIDE"
    assert ExportValidationRule.POSITIVE_QUANTITY == "POSITIVE_QUANTITY"
    assert ExportValidationRule.POSITIVE_ENTRY_PRICE == "POSITIVE_ENTRY_PRICE"
    assert ExportValidationRule.POSITIVE_EXIT_PRICE == "POSITIVE_EXIT_PRICE"
    assert ExportValidationRule.NOT_NULL_TIMESTAMPS == "NOT_NULL_TIMESTAMPS"
    assert ExportValidationRule.TEMPORAL_ORDER_VIOLATION == "TEMPORAL_ORDER_VIOLATION"
    assert ExportValidationRule.NON_NEGATIVE_COMMISSION == "NON_NEGATIVE_COMMISSION"
    assert ExportValidationRule.DUPLICATE_TRADE_ID == "DUPLICATE_TRADE_ID"
    assert ExportValidationRule.NOT_NULL_PORTFOLIO_ID == "NOT_NULL_PORTFOLIO_ID"
    assert ExportValidationRule.INVALID_POSITION_SIDE == "INVALID_POSITION_SIDE"
    assert ExportValidationRule.POSITIVE_CURRENT_PRICE == "POSITIVE_CURRENT_PRICE"
    assert ExportValidationRule.NOT_NULL_MARKET_VALUE == "NOT_NULL_MARKET_VALUE"
    assert ExportValidationRule.NOT_NULL_TIMESTAMP == "NOT_NULL_TIMESTAMP"
    assert ExportValidationRule.NON_NEGATIVE_PORTFOLIO_VALUE == "NON_NEGATIVE_PORTFOLIO_VALUE"
    assert ExportValidationRule.NOT_NULL_CASH == "NOT_NULL_CASH"
    assert ExportValidationRule.NON_NEGATIVE_DRAWDOWN == "NON_NEGATIVE_DRAWDOWN"
    assert ExportValidationRule.INVALID_DRAWDOWN_PCT == "INVALID_DRAWDOWN_PCT"


def test_export_validation_error_model():
    """Verify ExportValidationError model instantiation and payload serialization."""
    err = ExportValidationError(
        export_type=ExportType.TRADES,
        record_identifier="TRD-999",
        rule_name=ExportValidationRule.POSITIVE_QUANTITY,
        error_message="Quantity must be strictly positive",
        raw_payload={"trade_id": "TRD-999", "quantity": -5.0},
    )
    assert err.error_id is None
    assert err.export_type == ExportType.TRADES
    assert err.record_identifier == "TRD-999"
    assert err.rule_name == "POSITIVE_QUANTITY"
    assert isinstance(err.detected_at, datetime)
    assert err.raw_payload == {"trade_id": "TRD-999", "quantity": -5.0}


def test_raw_export_staging_models():
    """Verify raw models allow permissive ingestion of unvalidated staging rows."""
    raw_trade = RawTradeExport(
        trade_id="RAW-1",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        entry_price=150.0,
    )
    assert raw_trade.trade_id == "RAW-1"
    assert raw_trade.exit_price is None
    assert isinstance(raw_trade.ingested_at, datetime)

    raw_pos = RawPositionExport(
        portfolio_id="PORT-A",
        symbol="MSFT",
        side="LONG",
        quantity=50.0,
    )
    assert raw_pos.portfolio_id == "PORT-A"
    assert raw_pos.current_price is None

    raw_eq = RawEquityCurveExport(
        portfolio_id="PORT-A",
        portfolio_value=1_000_000.0,
    )
    assert raw_eq.portfolio_value == 1_000_000.0
    assert raw_eq.cash is None


def test_clean_trade_record_valid():
    """Verify CleanTradeRecord instantiates and normalizes valid trade data."""
    now = datetime.utcnow()
    later = now + timedelta(hours=2)
    trade = CleanTradeRecord(
        trade_id="TRD-100",
        symbol="nvda",
        side="buy",
        quantity=50.0,
        entry_price=120.0,
        exit_price=130.0,
        entry_time=now,
        exit_time=later,
        pnl=500.0,
        commission=2.5,
    )
    assert trade.trade_id == "TRD-100"
    assert trade.side == "BUY"
    assert trade.quantity == 50.0
    assert trade.pnl == 500.0
    assert trade.commission == 2.5
    assert trade.validated_at is not None


def test_clean_trade_record_validation_failures():
    """Verify CleanTradeRecord rejects invalid trade fields."""
    now = datetime.utcnow()
    earlier = now - timedelta(hours=1)

    # 1. Invalid side
    with pytest.raises(ValidationError):
        CleanTradeRecord(
            trade_id="T1",
            symbol="AAPL",
            side="HOLD",
            quantity=10.0,
            entry_price=100.0,
            exit_price=105.0,
            entry_time=now,
            exit_time=now,
        )

    # 2. Non-positive quantity
    with pytest.raises(ValidationError):
        CleanTradeRecord(
            trade_id="T2",
            symbol="AAPL",
            side="BUY",
            quantity=0.0,
            entry_price=100.0,
            exit_price=105.0,
            entry_time=now,
            exit_time=now,
        )

    # 3. Temporal order violation
    with pytest.raises(ValidationError):
        CleanTradeRecord(
            trade_id="T3",
            symbol="AAPL",
            side="BUY",
            quantity=10.0,
            entry_price=100.0,
            exit_price=105.0,
            entry_time=now,
            exit_time=earlier,
        )

    # 4. Negative commission
    with pytest.raises(ValidationError):
        CleanTradeRecord(
            trade_id="T4",
            symbol="AAPL",
            side="BUY",
            quantity=10.0,
            entry_price=100.0,
            exit_price=105.0,
            entry_time=now,
            exit_time=now,
            commission=-1.0,
        )


def test_clean_position_record_valid_and_invalid():
    """Verify CleanPositionRecord validation rules."""
    pos = CleanPositionRecord(
        portfolio_id="PORT-GLOBAL",
        symbol="SPY",
        name="SPDR S&P 500",
        side="long",
        quantity=100.0,
        entry_price=500.0,
        current_price=520.0,
        market_value=52000.0,
    )
    assert pos.side == "LONG"
    assert pos.quantity == 100.0
    assert pos.market_value == 52000.0

    # Invalid side for position (must be LONG or SHORT)
    with pytest.raises(ValidationError):
        CleanPositionRecord(
            portfolio_id="PORT-GLOBAL",
            symbol="SPY",
            name="SPDR S&P 500",
            side="BUY",
            quantity=100.0,
            entry_price=500.0,
            current_price=520.0,
            market_value=52000.0,
        )

    # Non-positive current price
    with pytest.raises(ValidationError):
        CleanPositionRecord(
            portfolio_id="PORT-GLOBAL",
            symbol="SPY",
            name="SPDR S&P 500",
            side="LONG",
            quantity=100.0,
            entry_price=500.0,
            current_price=-10.0,
            market_value=52000.0,
        )


def test_clean_equity_record_valid_and_invalid():
    """Verify CleanEquityRecord constraints."""
    now = datetime.utcnow()
    eq = CleanEquityRecord(
        portfolio_id="PORT-1",
        timestamp=now,
        portfolio_value=1_000_000.0,
        cash=250_000.0,
        drawdown=10_000.0,
        drawdown_pct=0.01,
    )
    assert eq.portfolio_value == 1_000_000.0
    assert eq.drawdown_pct == 0.01

    # Negative portfolio value
    with pytest.raises(ValidationError):
        CleanEquityRecord(
            portfolio_id="PORT-1",
            timestamp=now,
            portfolio_value=-500.0,
            cash=100.0,
        )

    # Invalid drawdown percentage (> 1.0)
    with pytest.raises(ValidationError):
        CleanEquityRecord(
            portfolio_id="PORT-1",
            timestamp=now,
            portfolio_value=100_000.0,
            cash=10_000.0,
            drawdown_pct=1.5,
        )


def test_export_validation_summary():
    """Verify ExportValidationSummary aggregation and is_clean property."""
    clean_summary = ExportValidationSummary(
        export_type=ExportType.TRADES,
        total_raw_records=100,
        valid_clean_records=100,
        quarantined_records=0,
        duplicate_records=0,
    )
    assert clean_summary.is_clean is True

    dirty_summary = ExportValidationSummary(
        export_type=ExportType.POSITIONS,
        total_raw_records=100,
        valid_clean_records=95,
        quarantined_records=5,
        duplicate_records=0,
        errors=[
            ExportValidationError(
                export_type=ExportType.POSITIONS,
                record_identifier="PORT-POS-1",
                rule_name=ExportValidationRule.POSITIVE_QUANTITY,
                error_message="Invalid quantity",
            )
        ],
    )
    assert dirty_summary.is_clean is False
    assert len(dirty_summary.errors) == 1


def test_pipeline_model_helpers():
    """Verify PostgreSQLExportPipeline clean helpers work with both models and dicts."""
    raw_trade = RawTradeExport(
        trade_id="TRD-P-01",
        symbol="AAPL",
        side="BUY",
        quantity=150.0,
        entry_price=140.0,
        exit_price=150.0,
        entry_time=datetime(2026, 1, 1, 9, 30),
        exit_time=datetime(2026, 1, 1, 16, 0),
        pnl=1500.0,
        commission=1.5,
    )
    # Validate method accepts model directly
    errors = PostgreSQLExportPipeline.validate_trade_record(raw_trade)
    assert errors == []

    # Clean helper produces CleanTradeRecord from raw model
    clean_trade = PostgreSQLExportPipeline.clean_trade(raw_trade)
    assert isinstance(clean_trade, CleanTradeRecord)
    assert clean_trade.trade_id == "TRD-P-01"
    assert clean_trade.side == "BUY"

    # Clean helper produces CleanPositionRecord from dict
    pos_dict = {
        "portfolio_id": "PORT-99",
        "symbol": "QQQ",
        "name": "Invesco QQQ",
        "side": "LONG",
        "quantity": 200,
        "entry_price": 400.0,
        "current_price": 420.0,
        "market_value": 84000.0,
    }
    clean_pos = PostgreSQLExportPipeline.clean_position(pos_dict)
    assert isinstance(clean_pos, CleanPositionRecord)
    assert clean_pos.symbol == "QQQ"

    # Clean helper produces CleanEquityRecord from ISO string dict
    equity_dict = {
        "portfolio_id": "PORT-99",
        "timestamp": "2026-01-02T16:00:00Z",
        "portfolio_value": 500000.0,
        "cash": 120000.0,
        "drawdown": 5000.0,
        "drawdown_pct": 0.01,
    }
    clean_eq = PostgreSQLExportPipeline.clean_equity(equity_dict)
    assert isinstance(clean_eq, CleanEquityRecord)
    assert clean_eq.portfolio_value == 500000.0

    # Dirty trade raises ValueError
    dirty_trade = RawTradeExport(
        trade_id="",
        symbol="AAPL",
        side="INVALID",
        quantity=-10.0,
        entry_price=0.0,
        exit_price=0.0,
    )
    with pytest.raises(ValueError) as exc_info:
        PostgreSQLExportPipeline.clean_trade(dirty_trade)
    assert "Trade validation failed" in str(exc_info.value)
