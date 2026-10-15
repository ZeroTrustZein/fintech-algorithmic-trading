"""PostgreSQL SQL schema definitions, raw export staging, and validation pipeline."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    BacktestTrade,
    CleanEquityRecord,
    CleanPositionRecord,
    CleanTradeRecord,
    ExportType,
    ExportValidationError,
    ExportValidationRule,
    ExportValidationSummary,
    OrderSide,
    Portfolio,
    Position,
    PositionType,
    RawEquityCurveExport,
    RawPositionExport,
    RawTradeExport,
)


class PostgreSQLExportPipeline:
    """Manages PostgreSQL schema DDL, SQL transformations, and validation for raw exports."""

    SQL_DIR = Path(__file__).parent / "sql"

    STAGING_COLUMNS: Dict[ExportType, Tuple[str, List[str]]] = {
        ExportType.TRADES: (
            "raw_trades_export",
            [
                "trade_id",
                "symbol",
                "side",
                "quantity",
                "entry_price",
                "exit_price",
                "entry_time",
                "exit_time",
                "pnl",
                "return_pct",
                "commission",
                "holding_period_bars",
            ],
        ),
        ExportType.POSITIONS: (
            "raw_positions_export",
            [
                "portfolio_id",
                "symbol",
                "name",
                "asset_class",
                "sector",
                "side",
                "quantity",
                "entry_price",
                "current_price",
                "market_value",
                "unrealized_pnl",
                "unrealized_pnl_pct",
            ],
        ),
        ExportType.EQUITY_CURVE: (
            "raw_equity_curve_export",
            ["portfolio_id", "timestamp", "portfolio_value", "cash", "drawdown", "drawdown_pct"],
        ),
    }

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
    def _parse_iso_datetime(val: Union[str, datetime]) -> datetime:
        """Parse ISO timestamp string or return datetime object."""
        if isinstance(val, str):
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        return val

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
    def validate_position_record(
        cls, record: Union[Dict[str, Any], RawPositionExport]
    ) -> List[str]:
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
    def validate_equity_record(
        cls, record: Union[Dict[str, Any], RawEquityCurveExport]
    ) -> List[str]:
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

        entry_t = cls._parse_iso_datetime(rec["entry_time"])
        exit_t = cls._parse_iso_datetime(rec["exit_time"])

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
    def clean_position(
        cls, record: Union[Dict[str, Any], RawPositionExport]
    ) -> CleanPositionRecord:
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

        ts = cls._parse_iso_datetime(rec["timestamp"])

        return CleanEquityRecord(
            curve_id=rec.get("curve_id"),
            portfolio_id=str(rec["portfolio_id"]).strip(),
            timestamp=ts,
            portfolio_value=float(rec["portfolio_value"]),
            cash=float(rec["cash"]),
            drawdown=float(rec.get("drawdown") or 0.0),
            drawdown_pct=float(rec.get("drawdown_pct") or 0.0),
        )

    @classmethod
    def _process_batch(
        cls,
        export_type: ExportType,
        records: Sequence[Union[Dict[str, Any], Any]],
        validate_fn: Any,
        clean_fn: Any,
        key_fn: Any,
        rec_id_fn: Any,
        dup_rule: str,
        dup_msg_fn: Any,
        deduplicate: bool = True,
    ) -> Tuple[List[Any], ExportValidationSummary]:
        """Generic processor for validating, deduplicating, and summarizing record batches."""
        clean_map: Dict[Any, Any] = {}
        errors: List[ExportValidationError] = []
        quarantined = 0
        duplicates = 0

        for r in records:
            rec = cls._to_dict(r)
            err_msgs = validate_fn(rec)
            rec_id = rec_id_fn(rec)

            if err_msgs:
                quarantined += 1
                for msg in err_msgs:
                    rule = (
                        msg.split(":")[0].strip()
                        if ":" in msg
                        else ExportValidationRule.GENERIC_INVALID.value
                    )
                    errors.append(
                        ExportValidationError(
                            export_type=export_type,
                            record_identifier=rec_id,
                            rule_name=rule,
                            error_message=msg,
                            raw_payload=rec,
                        )
                    )
            else:
                clean_rec = clean_fn(rec)
                key = key_fn(clean_rec, rec)
                if key in clean_map:
                    duplicates += 1
                    errors.append(
                        ExportValidationError(
                            export_type=export_type,
                            record_identifier=rec_id,
                            rule_name=dup_rule,
                            error_message=dup_msg_fn(key),
                            raw_payload=rec,
                        )
                    )
                    if deduplicate:
                        clean_map[key] = clean_rec
                else:
                    clean_map[key] = clean_rec

        summary = ExportValidationSummary(
            export_type=export_type,
            total_raw_records=len(records),
            valid_clean_records=len(clean_map),
            quarantined_records=quarantined,
            duplicate_records=duplicates,
            errors=errors,
        )
        return list(clean_map.values()), summary

    @classmethod
    def process_trades(
        cls,
        records: Sequence[Union[Dict[str, Any], RawTradeExport]],
        deduplicate: bool = True,
    ) -> Tuple[List[CleanTradeRecord], ExportValidationSummary]:
        """Batch clean, validate, and deduplicate raw trade export records."""
        return cls._process_batch(
            export_type=ExportType.TRADES,
            records=records,
            validate_fn=cls.validate_trade_record,
            clean_fn=cls.clean_trade,
            key_fn=lambda c, r: c.trade_id,
            rec_id_fn=lambda r: str(r["trade_id"]).strip() if r.get("trade_id") else "UNKNOWN_ID",
            dup_rule=ExportValidationRule.DUPLICATE_TRADE_ID.value,
            dup_msg_fn=lambda k: (
                f"DUPLICATE_TRADE_ID: trade_id '{k}' already exists in batch; UPSERT applied"
            ),
            deduplicate=deduplicate,
        )

    @classmethod
    def process_positions(
        cls,
        records: Sequence[Union[Dict[str, Any], RawPositionExport]],
        deduplicate: bool = True,
    ) -> Tuple[List[CleanPositionRecord], ExportValidationSummary]:
        """Batch clean, validate, and deduplicate raw position export records."""
        return cls._process_batch(
            export_type=ExportType.POSITIONS,
            records=records,
            validate_fn=cls.validate_position_record,
            clean_fn=cls.clean_position,
            key_fn=lambda c, r: (c.portfolio_id, c.symbol, c.side),
            rec_id_fn=lambda r: (
                f"{r.get('portfolio_id') or 'UNKNOWN_PORTFOLIO'}:{r.get('symbol') or 'UNKNOWN_SYM'}:{r.get('side') or 'UNKNOWN_SIDE'}"
            ),
            dup_rule="DUPLICATE_POSITION_KEY",
            dup_msg_fn=lambda k: (
                f"DUPLICATE_POSITION: Key {k} already exists in batch; UPSERT applied"
            ),
            deduplicate=deduplicate,
        )

    @classmethod
    def process_equity_curve(
        cls,
        records: Sequence[Union[Dict[str, Any], RawEquityCurveExport]],
        deduplicate: bool = True,
    ) -> Tuple[List[CleanEquityRecord], ExportValidationSummary]:
        """Batch clean, validate, and deduplicate raw equity curve export records."""
        return cls._process_batch(
            export_type=ExportType.EQUITY_CURVE,
            records=records,
            validate_fn=cls.validate_equity_record,
            clean_fn=cls.clean_equity,
            key_fn=lambda c, r: (c.portfolio_id, c.timestamp.isoformat()),
            rec_id_fn=lambda r: (
                f"{r.get('portfolio_id') or 'UNKNOWN_PORTFOLIO'}:{r.get('timestamp') or 'UNKNOWN_TS'}"
            ),
            dup_rule="DUPLICATE_EQUITY_SNAPSHOT",
            dup_msg_fn=lambda k: f"DUPLICATE_EQUITY: Snapshot {k} already exists; UPSERT applied",
            deduplicate=deduplicate,
        )

    @classmethod
    def process_export(
        cls,
        export_type: Union[ExportType, str],
        records: List[Union[Dict[str, Any], Any]],
        deduplicate: bool = True,
    ) -> Tuple[List[Any], ExportValidationSummary]:
        """Unified dispatch for batch cleaning and validation by export type."""
        exp_type = ExportType(export_type)
        if exp_type == ExportType.TRADES:
            return cls.process_trades(records, deduplicate=deduplicate)
        elif exp_type == ExportType.POSITIONS:
            return cls.process_positions(records, deduplicate=deduplicate)
        elif exp_type == ExportType.EQUITY_CURVE:
            return cls.process_equity_curve(records, deduplicate=deduplicate)
        raise ValueError(f"Unsupported export type: {export_type}")

    @classmethod
    def process_csv(
        cls,
        csv_content: str,
        export_type: Union[ExportType, str],
        default_portfolio_id: Optional[str] = None,
        deduplicate: bool = True,
    ) -> Tuple[List[Any], ExportValidationSummary]:
        """Ingest raw CSV string export, parse rows, and run through the cleaning pipeline."""
        reader = csv.DictReader(io.StringIO(csv_content.strip()))
        rows: List[Dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            if default_portfolio_id and "portfolio_id" not in parsed:
                parsed["portfolio_id"] = default_portfolio_id
            rows.append(parsed)
        return cls.process_export(export_type, rows, deduplicate=deduplicate)

    @staticmethod
    def _sql_escape(val: Any) -> str:
        """Safely format values for PostgreSQL SQL literal insertion."""
        if val is None:
            return "NULL"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, datetime):
            return f"'{val.isoformat()}'"
        clean_str = str(val).replace("'", "''")
        return f"'{clean_str}'"

    @classmethod
    def generate_staging_insert_sql(
        cls, export_type: Union[ExportType, str], records: List[Dict[str, Any]]
    ) -> str:
        """Generate PostgreSQL bulk INSERT statement into staging table."""
        if not records:
            return ""
        exp_type = ExportType(export_type)
        table_name, cols = cls.STAGING_COLUMNS[exp_type]
        value_rows = [
            f"    ({', '.join(cls._sql_escape(r.get(c)) for c in cols)})" for r in records
        ]
        return (
            f"INSERT INTO {table_name} ({', '.join(cols)})\nVALUES\n" + ",\n".join(value_rows) + ";"
        )

    @classmethod
    def execute_db_pipeline(
        cls,
        connection: Any,
        export_type: Union[ExportType, str],
        records: List[Dict[str, Any]],
        init_schema: bool = True,
    ) -> ExportValidationSummary:
        """Execute PostgreSQL staging, cleaning, and error audit via database connection."""
        cursor = connection.cursor()
        try:
            if init_schema:
                cursor.execute(cls.get_schema_ddl())
            insert_sql = cls.generate_staging_insert_sql(export_type, records)
            if insert_sql:
                cursor.execute(insert_sql)
            cursor.execute(cls.get_clean_validate_sql())

            cursor.execute(
                "SELECT export_type, record_identifier, rule_name, error_message "
                "FROM export_validation_errors WHERE export_type = %s",
                (ExportType(export_type).value,),
            )
            rows = cursor.fetchall()
            db_errors = [
                ExportValidationError(
                    export_type=ExportType(row[0]),
                    record_identifier=row[1] or "",
                    rule_name=row[2],
                    error_message=row[3],
                )
                for row in rows
            ]
            return ExportValidationSummary(
                export_type=ExportType(export_type),
                total_raw_records=len(records),
                valid_clean_records=len(records) - len(db_errors),
                quarantined_records=len(db_errors),
                duplicate_records=sum(1 for e in db_errors if "DUPLICATE" in e.rule_name),
                errors=db_errors,
            )
        finally:
            cursor.close()

    @classmethod
    def reconstruct_portfolio(
        cls,
        portfolio_id: str,
        clean_positions: List[CleanPositionRecord],
        name: Optional[str] = None,
        cash: float = 0.0,
    ) -> Portfolio:
        """Reconstruct a domain Portfolio model from cleaned position export records."""
        positions: List[Position] = []
        for p in clean_positions:
            ac_val = p.asset_class.upper() if isinstance(p.asset_class, str) else str(p.asset_class)
            try:
                ac = AssetClass(ac_val)
            except ValueError:
                ac = AssetClass.EQUITY

            side_val = p.side.upper() if isinstance(p.side, str) else str(p.side)
            side = PositionType.LONG if side_val in ("LONG", "BUY") else PositionType.SHORT

            asset = Asset(
                symbol=p.symbol,
                name=p.name,
                asset_class=ac,
                sector=p.sector,
                current_price=p.current_price,
                volatility_annual=0.20,
            )
            positions.append(
                Position(
                    asset=asset,
                    quantity=p.quantity,
                    entry_price=p.entry_price,
                    current_price=p.current_price,
                    side=side,
                )
            )
        return Portfolio(
            id=portfolio_id,
            name=name or f"Portfolio {portfolio_id}",
            description=f"Reconstructed from PostgreSQL clean exports for {portfolio_id}",
            cash=cash,
            positions=positions,
        )

    @classmethod
    def reconstruct_trades(
        cls,
        clean_trades: List[CleanTradeRecord],
    ) -> List[BacktestTrade]:
        """Reconstruct domain BacktestTrade models from cleaned trade export records."""
        trades: List[BacktestTrade] = []
        for t in clean_trades:
            side_val = t.side.upper() if isinstance(t.side, str) else str(t.side)
            side = OrderSide.BUY if side_val in ("BUY", "LONG") else OrderSide.SELL
            trades.append(
                BacktestTrade(
                    trade_id=t.trade_id,
                    symbol=t.symbol,
                    side=side,
                    quantity=t.quantity,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    entry_time=t.entry_time,
                    exit_time=t.exit_time,
                    pnl=t.pnl,
                    return_pct=t.return_pct,
                    commission=t.commission,
                    holding_period_bars=t.holding_period_bars,
                )
            )
        return trades
