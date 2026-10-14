"""PostgreSQL SQL schema definitions, raw export staging, and validation pipeline."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from fintech_algorithmic_trading.types import (
    CleanEquityRecord,
    CleanPositionRecord,
    CleanTradeRecord,
    RawEquityCurveExport,
    RawPositionExport,
    RawTradeExport,
)


class PostgreSQLExportPipeline:
    """Manages PostgreSQL schema DDL, SQL transformations, and validation for raw exports."""

    SQL_DIR = Path(__file__).parent / "sql"

    @classmethod
    def get_schema_ddl(cls) -> str:
        """Return the PostgreSQL schema DDL for staging, cleaning, and error tables."""
        schema_path = cls.SQL_DIR / "schema.sql"
        if schema_path.is_file():
            return schema_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    @classmethod
    def get_clean_validate_sql(cls) -> str:
        """Return the PostgreSQL SQL script for data cleaning, deduplication, and validation."""
        script_path = cls.SQL_DIR / "clean_validate.sql"
        if script_path.is_file():
            return script_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Clean & validate SQL file not found at {script_path}")

    @staticmethod
    def _to_dict(record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Convert pydantic model or dict-like object to a plain dictionary."""
        if hasattr(record, "model_dump"):
            return record.model_dump()
        return dict(record)

    @staticmethod
    def generate_copy_command(table_name: str, csv_file_path: str) -> str:
        """Generate PostgreSQL COPY command to ingest raw CSV export."""
        sanitized_path = csv_file_path.replace("'", "''")
        return (
            f"COPY {table_name} FROM '{sanitized_path}' "
            f"WITH (FORMAT csv, HEADER true, DELIMITER ',');"
        )

    @classmethod
    def validate_trade_record(cls, record: Union[Dict[str, Any], RawTradeExport]) -> List[str]:
        """In-memory validation mirror of the SQL validation rules for trade records."""
        rec = cls._to_dict(record)
        errors: List[str] = []
        trade_id = rec.get("trade_id")
        if not trade_id:
            errors.append("NOT_NULL_TRADE_ID: trade_id cannot be null or empty")

        symbol = rec.get("symbol")
        if not symbol or not str(symbol).strip():
            errors.append("NOT_NULL_SYMBOL: symbol cannot be empty")

        side = str(rec.get("side", "")).upper().strip()
        if side not in ("LONG", "SHORT", "BUY", "SELL"):
            errors.append("INVALID_SIDE: side must be LONG, SHORT, BUY, or SELL")

        try:
            qty = float(rec.get("quantity", 0))
            if qty <= 0:
                errors.append("POSITIVE_QUANTITY: quantity must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_QUANTITY: invalid numeric quantity")

        try:
            entry_p = float(rec.get("entry_price", 0))
            if entry_p <= 0:
                errors.append("POSITIVE_ENTRY_PRICE: entry_price must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_ENTRY_PRICE: invalid numeric entry_price")

        try:
            exit_p = float(rec.get("exit_price", 0))
            if exit_p <= 0:
                errors.append("POSITIVE_EXIT_PRICE: exit_price must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_EXIT_PRICE: invalid numeric exit_price")

        entry_t = rec.get("entry_time")
        exit_t = rec.get("exit_time")
        if not entry_t or not exit_t:
            errors.append("NOT_NULL_TIMESTAMPS: timestamps must not be null")
        elif str(exit_t) < str(entry_t):
            errors.append("TEMPORAL_ORDER_VIOLATION: exit_time must be >= entry_time")

        comm = rec.get("commission")
        if comm is not None:
            try:
                if float(comm) < 0:
                    errors.append("NON_NEGATIVE_COMMISSION: commission cannot be negative")
            except (ValueError, TypeError):
                errors.append("NON_NEGATIVE_COMMISSION: invalid numeric commission")

        return errors

    @classmethod
    def validate_position_record(cls, record: Union[Dict[str, Any], RawPositionExport]) -> List[str]:
        """In-memory validation mirror of the SQL validation rules for position records."""
        rec = cls._to_dict(record)
        errors: List[str] = []
        if not rec.get("portfolio_id"):
            errors.append("NOT_NULL_PORTFOLIO_ID: portfolio_id cannot be empty")

        symbol = rec.get("symbol")
        if not symbol or not str(symbol).strip():
            errors.append("NOT_NULL_SYMBOL: symbol cannot be empty")

        side = str(rec.get("side", "")).upper().strip()
        if side not in ("LONG", "SHORT"):
            errors.append("INVALID_POSITION_SIDE: side must be LONG or SHORT")

        try:
            qty = float(rec.get("quantity", 0))
            if qty <= 0:
                errors.append("POSITIVE_QUANTITY: quantity must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_QUANTITY: invalid numeric quantity")

        try:
            cur_p = float(rec.get("current_price", 0))
            if cur_p <= 0:
                errors.append("POSITIVE_CURRENT_PRICE: current_price must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_CURRENT_PRICE: invalid numeric current_price")

        if rec.get("market_value") is None:
            errors.append("NOT_NULL_MARKET_VALUE: market_value cannot be null")

        return errors

    @classmethod
    def validate_equity_record(cls, record: Union[Dict[str, Any], RawEquityCurveExport]) -> List[str]:
        """In-memory validation mirror of the SQL validation rules for equity curve records."""
        rec = cls._to_dict(record)
        errors: List[str] = []
        if not rec.get("portfolio_id"):
            errors.append("NOT_NULL_PORTFOLIO_ID: portfolio_id cannot be empty")

        if not rec.get("timestamp"):
            errors.append("NOT_NULL_TIMESTAMP: timestamp cannot be null")

        try:
            val = float(rec.get("portfolio_value", 0))
            if val < 0:
                errors.append("NON_NEGATIVE_PORTFOLIO_VALUE: portfolio_value cannot be negative")
        except (ValueError, TypeError):
            errors.append("NON_NEGATIVE_PORTFOLIO_VALUE: invalid numeric portfolio_value")

        if rec.get("cash") is None:
            errors.append("NOT_NULL_CASH: cash cannot be null")

        dd = rec.get("drawdown")
        if dd is not None:
            try:
                if float(dd) < 0:
                    errors.append("NON_NEGATIVE_DRAWDOWN: drawdown cannot be negative")
            except (ValueError, TypeError):
                errors.append("NON_NEGATIVE_DRAWDOWN: invalid numeric drawdown")

        dd_pct = rec.get("drawdown_pct")
        if dd_pct is not None:
            try:
                val_pct = float(dd_pct)
                if val_pct < 0 or val_pct > 1.0:
                    errors.append("INVALID_DRAWDOWN_PCT: drawdown_pct must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("INVALID_DRAWDOWN_PCT: invalid numeric drawdown_pct")

        return errors

    @classmethod
    def clean_trade(cls, record: Union[Dict[str, Any], RawTradeExport]) -> CleanTradeRecord:
        """Validate and construct a CleanTradeRecord or raise ValueError."""
        rec = cls._to_dict(record)
        errors = cls.validate_trade_record(rec)
        if errors:
            raise ValueError(f"Trade validation failed: {'; '.join(errors)}")

        entry_t = rec["entry_time"]
        if isinstance(entry_t, str):
            entry_t = datetime.fromisoformat(entry_t.replace("Z", "+00:00"))
        exit_t = rec["exit_time"]
        if isinstance(exit_t, str):
            exit_t = datetime.fromisoformat(exit_t.replace("Z", "+00:00"))

        return CleanTradeRecord(
            trade_id=str(rec["trade_id"]).strip(),
            symbol=str(rec["symbol"]).upper().strip(),
            side=str(rec["side"]).upper().strip(),
            quantity=float(rec["quantity"]),
            entry_price=float(rec["entry_price"]),
            exit_price=float(rec["exit_price"]),
            entry_time=entry_t,
            exit_time=exit_t,
            pnl=float(rec.get("pnl") or 0.0),
            return_pct=float(rec.get("return_pct") or 0.0),
            commission=float(rec.get("commission") or 0.0),
            holding_period_bars=int(rec.get("holding_period_bars") or 0),
        )

    @classmethod
    def clean_position(cls, record: Union[Dict[str, Any], RawPositionExport]) -> CleanPositionRecord:
        """Validate and construct a CleanPositionRecord or raise ValueError."""
        rec = cls._to_dict(record)
        errors = cls.validate_position_record(rec)
        if errors:
            raise ValueError(f"Position validation failed: {'; '.join(errors)}")

        entry_price = float(rec.get("entry_price") or rec.get("current_price") or 0.0)

        return CleanPositionRecord(
            position_id=rec.get("position_id"),
            portfolio_id=str(rec["portfolio_id"]).strip(),
            symbol=str(rec["symbol"]).upper().strip(),
            name=str(rec.get("name") or rec["symbol"]).strip(),
            asset_class=str(rec.get("asset_class") or "EQUITY").upper().strip(),
            sector=rec.get("sector"),
            side=str(rec["side"]).upper().strip(),
            quantity=float(rec["quantity"]),
            entry_price=entry_price,
            current_price=float(rec["current_price"]),
            market_value=float(rec["market_value"]),
            unrealized_pnl=float(rec.get("unrealized_pnl") or 0.0),
            unrealized_pnl_pct=float(rec.get("unrealized_pnl_pct") or 0.0),
        )

    @classmethod
    def clean_equity(cls, record: Union[Dict[str, Any], RawEquityCurveExport]) -> CleanEquityRecord:
        """Validate and construct a CleanEquityRecord or raise ValueError."""
        rec = cls._to_dict(record)
        errors = cls.validate_equity_record(rec)
        if errors:
            raise ValueError(f"Equity curve validation failed: {'; '.join(errors)}")

        ts = rec["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        return CleanEquityRecord(
            curve_id=rec.get("curve_id"),
            portfolio_id=str(rec["portfolio_id"]).strip(),
            timestamp=ts,
            portfolio_value=float(rec["portfolio_value"]),
            cash=float(rec["cash"]),
            drawdown=float(rec.get("drawdown") or 0.0),
            drawdown_pct=float(rec.get("drawdown_pct") or 0.0),
        )
