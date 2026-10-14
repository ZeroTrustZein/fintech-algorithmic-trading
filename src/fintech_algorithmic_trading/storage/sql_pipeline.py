"""PostgreSQL SQL schema definitions, raw export staging, and validation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


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
    def generate_copy_command(table_name: str, csv_file_path: str) -> str:
        """Generate PostgreSQL COPY command to ingest raw CSV export."""
        sanitized_path = csv_file_path.replace("'", "''")
        return (
            f"COPY {table_name} FROM '{sanitized_path}' "
            f"WITH (FORMAT csv, HEADER true, DELIMITER ',');"
        )

    @staticmethod
    def validate_trade_record(record: Dict[str, Any]) -> List[str]:
        """In-memory validation mirror of the SQL validation rules for trade records."""
        errors: List[str] = []
        trade_id = record.get("trade_id")
        if not trade_id:
            errors.append("NOT_NULL_TRADE_ID: trade_id cannot be null or empty")

        symbol = record.get("symbol")
        if not symbol or not str(symbol).strip():
            errors.append("NOT_NULL_SYMBOL: symbol cannot be empty")

        side = str(record.get("side", "")).upper().strip()
        if side not in ("LONG", "SHORT", "BUY", "SELL"):
            errors.append("INVALID_SIDE: side must be LONG, SHORT, BUY, or SELL")

        try:
            qty = float(record.get("quantity", 0))
            if qty <= 0:
                errors.append("POSITIVE_QUANTITY: quantity must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_QUANTITY: invalid numeric quantity")

        try:
            entry_p = float(record.get("entry_price", 0))
            if entry_p <= 0:
                errors.append("POSITIVE_ENTRY_PRICE: entry_price must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_ENTRY_PRICE: invalid numeric entry_price")

        try:
            exit_p = float(record.get("exit_price", 0))
            if exit_p <= 0:
                errors.append("POSITIVE_EXIT_PRICE: exit_price must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_EXIT_PRICE: invalid numeric exit_price")

        entry_t = record.get("entry_time")
        exit_t = record.get("exit_time")
        if not entry_t or not exit_t:
            errors.append("NOT_NULL_TIMESTAMPS: timestamps must not be null")
        elif str(exit_t) < str(entry_t):
            errors.append("TEMPORAL_ORDER_VIOLATION: exit_time must be >= entry_time")

        comm = record.get("commission")
        if comm is not None:
            try:
                if float(comm) < 0:
                    errors.append("NON_NEGATIVE_COMMISSION: commission cannot be negative")
            except (ValueError, TypeError):
                errors.append("NON_NEGATIVE_COMMISSION: invalid numeric commission")

        return errors

    @staticmethod
    def validate_position_record(record: Dict[str, Any]) -> List[str]:
        """In-memory validation mirror of the SQL validation rules for position records."""
        errors: List[str] = []
        if not record.get("portfolio_id"):
            errors.append("NOT_NULL_PORTFOLIO_ID: portfolio_id cannot be empty")

        symbol = record.get("symbol")
        if not symbol or not str(symbol).strip():
            errors.append("NOT_NULL_SYMBOL: symbol cannot be empty")

        side = str(record.get("side", "")).upper().strip()
        if side not in ("LONG", "SHORT"):
            errors.append("INVALID_POSITION_SIDE: side must be LONG or SHORT")

        try:
            qty = float(record.get("quantity", 0))
            if qty <= 0:
                errors.append("POSITIVE_QUANTITY: quantity must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_QUANTITY: invalid numeric quantity")

        try:
            cur_p = float(record.get("current_price", 0))
            if cur_p <= 0:
                errors.append("POSITIVE_CURRENT_PRICE: current_price must be strictly positive")
        except (ValueError, TypeError):
            errors.append("POSITIVE_CURRENT_PRICE: invalid numeric current_price")

        if record.get("market_value") is None:
            errors.append("NOT_NULL_MARKET_VALUE: market_value cannot be null")

        return errors

    @staticmethod
    def validate_equity_record(record: Dict[str, Any]) -> List[str]:
        """In-memory validation mirror of the SQL validation rules for equity curve records."""
        errors: List[str] = []
        if not record.get("portfolio_id"):
            errors.append("NOT_NULL_PORTFOLIO_ID: portfolio_id cannot be empty")

        if not record.get("timestamp"):
            errors.append("NOT_NULL_TIMESTAMP: timestamp cannot be null")

        try:
            val = float(record.get("portfolio_value", 0))
            if val < 0:
                errors.append("NON_NEGATIVE_PORTFOLIO_VALUE: portfolio_value cannot be negative")
        except (ValueError, TypeError):
            errors.append("NON_NEGATIVE_PORTFOLIO_VALUE: invalid numeric portfolio_value")

        if record.get("cash") is None:
            errors.append("NOT_NULL_CASH: cash cannot be null")

        dd = record.get("drawdown")
        if dd is not None:
            try:
                if float(dd) < 0:
                    errors.append("NON_NEGATIVE_DRAWDOWN: drawdown cannot be negative")
            except (ValueError, TypeError):
                errors.append("NON_NEGATIVE_DRAWDOWN: invalid numeric drawdown")

        dd_pct = record.get("drawdown_pct")
        if dd_pct is not None:
            try:
                val_pct = float(dd_pct)
                if val_pct < 0 or val_pct > 1.0:
                    errors.append("INVALID_DRAWDOWN_PCT: drawdown_pct must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("INVALID_DRAWDOWN_PCT: invalid numeric drawdown_pct")

        return errors
