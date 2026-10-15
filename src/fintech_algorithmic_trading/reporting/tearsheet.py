"""Institutional risk tearsheet generation, performance attribution, and CSV/JSON export."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from fintech_algorithmic_trading.types import (
    BacktestResult,
    BacktestTrade,
    CleanEquityRecord,
    CleanPositionRecord,
    CleanTradeRecord,
    ExportType,
    ExportValidationSummary,
    MonteCarloResult,
    Portfolio,
    RiskComplianceReport,
    StressTestResult,
    VaRMethod,
    VaRResult,
)


class RiskReportGenerator:
    """Generates institutional Markdown tearsheets and quantitative risk audit memos."""

    @staticmethod
    def generate_markdown_tearsheet(
        portfolio: Portfolio,
        var_results: Optional[Dict[VaRMethod, VaRResult]] = None,
        monte_carlo_res: Optional[MonteCarloResult] = None,
        stress_results: Optional[List[StressTestResult]] = None,
        compliance_report: Optional[RiskComplianceReport] = None,
    ) -> str:
        """Construct comprehensive institutional risk tearsheet."""
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"# Institutional Portfolio Risk Tearsheet: {portfolio.name}",
            f"**Report Generated:** {now_str}  ",
            f"**Portfolio Identifier:** `{portfolio.id}`  ",
            f"**Mandate Description:** {portfolio.description}",
            "",
            "## 1. Executive Capital & Leverage Summary",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Net Asset Value (NAV)** | **${portfolio.net_asset_value:,.2f}** |",
            f"| Unencumbered Cash | ${portfolio.cash:,.2f} |",
            f"| Gross Market Exposure | ${portfolio.gross_exposure:,.2f} |",
            f"| Net Directional Exposure | ${portfolio.net_exposure:,.2f} |",
            f"| Current Gross Leverage | **{portfolio.leverage:.2f}x** |",
            f"| Active Position Count | {len(portfolio.positions)} |",
            "",
            "## 2. Asset Allocation & Position Ledger",
            "",
            "| Symbol | Asset Class | Sector | Side | Quantity | Entry ($) | Mark ($) | Market Value ($) | Unrealized P&L ($) | P&L (%) |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for p in portfolio.positions:
            unreal_pnl = p.unrealized_pnl
            unreal_pct = p.unrealized_pnl_pct * 100.0
            pnl_sign = "+" if unreal_pnl >= 0 else ""
            lines.append(
                f"| **{p.asset.symbol}** | {p.asset.asset_class.value} | {p.asset.sector or 'N/A'} | "
                f"`{p.side.value}` | {p.quantity:,.1f} | ${p.entry_price:,.2f} | ${p.current_price:,.2f} | "
                f"${p.market_value:,.2f} | {pnl_sign}${unreal_pnl:,.2f} | {pnl_sign}{unreal_pct:.2f}% |"
            )

        # 3. Value-at-Risk Section
        if var_results:
            lines.extend(
                [
                    "",
                    "## 3. Multi-Methodology Value-at-Risk & Expected Shortfall (CVaR)",
                    "",
                    "| Methodology | Confidence | Horizon | VaR ($) | VaR (%) | CVaR ($) | CVaR (%) | Skew / Kurt |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for m, res in var_results.items():
                sk_kurt = (
                    f"{res.skewness:.2f} / {res.excess_kurtosis:.2f}"
                    if res.skewness is not None
                    else "N/A"
                )
                lines.append(
                    f"| **{m.value}** | {res.confidence_level * 100:.1f}% | {res.horizon_days}d | "
                    f"${res.var_amount:,.2f} | {res.var_pct * 100:.2f}% | "
                    f"${res.cvar_amount:,.2f} | {res.cvar_pct * 100:.2f}% | {sk_kurt} |"
                )

        # 4. Monte Carlo Distribution Section
        if monte_carlo_res:
            mc = monte_carlo_res
            lines.extend(
                [
                    "",
                    f"## 4. Correlated Monte Carlo Simulation ({mc.config.n_simulations:,} paths, {mc.config.horizon_days}d Horizon)",
                    "",
                    "| Percentile / Quantile Metric | Terminal Value ($) | Cumulative Return (%) |",
                    "| :--- | :---: | :---: |",
                    f"| **Mean Expected Terminal NAV** | ${mc.mean_terminal_value:,.2f} | {((mc.mean_terminal_value - mc.initial_portfolio_value) / mc.initial_portfolio_value) * 100:.2f}% |",
                    f"| Median (P50) Scenario | ${mc.median_terminal_value:,.2f} | {((mc.median_terminal_value - mc.initial_portfolio_value) / mc.initial_portfolio_value) * 100:.2f}% |",
                    f"| 95th Percentile Upside (P95) | ${mc.p95_terminal_value:,.2f} | {((mc.p95_terminal_value - mc.initial_portfolio_value) / mc.initial_portfolio_value) * 100:.2f}% |",
                    f"| 5th Percentile Downside (P5) | ${mc.p5_terminal_value:,.2f} | {((mc.p5_terminal_value - mc.initial_portfolio_value) / mc.initial_portfolio_value) * 100:.2f}% |",
                    f"| 1st Percentile Tail Risk (P1) | ${mc.p1_terminal_value:,.2f} | {((mc.p1_terminal_value - mc.initial_portfolio_value) / mc.initial_portfolio_value) * 100:.2f}% |",
                    f"| Mean Maximum Drawdown | - | {mc.max_drawdown_mean_pct * 100:.2f}% |",
                    f"| Probability of Capital Loss | - | **{mc.probability_of_loss * 100:.2f}%** |",
                ]
            )

        # 5. Stress Testing Section
        if stress_results:
            lines.extend(
                [
                    "",
                    "## 5. Crisis Stress Testing & Shock Sensitivity",
                    "",
                    "| Crisis Scenario | Post-Shock NAV ($) | Absolute P&L ($) | Impact (%) | Capital Adequacy |",
                    "| :--- | :---: | :---: | :---: | :---: |",
                ]
            )
            for st in stress_results:
                status_str = "PASS" if st.capital_adequacy_passed else "**FAIL**"
                sign = "+" if st.absolute_impact >= 0 else ""
                lines.append(
                    f"| **{st.scenario.name}** | ${st.portfolio_value_after:,.2f} | "
                    f"{sign}${st.absolute_impact:,.2f} | {st.percentage_impact * 100:.2f}% | {status_str} |"
                )

        # 6. Compliance Section
        if compliance_report:
            lines.extend(
                [
                    "",
                    "## 6. Risk Policy Mandate Compliance Audit",
                    "",
                    f"**Audit Status:** {'PASSED' if compliance_report.passed else '**BREACH DETECTED**'}  ",
                    f"**Peak Position Concentration:** {compliance_report.max_single_position_pct * 100:.1f}% (`{compliance_report.max_single_position_symbol}`)  ",
                    f"**Peak Sector Concentration:** {compliance_report.max_sector_pct * 100:.1f}% (`{compliance_report.max_sector_name}`)  ",
                    "",
                ]
            )
            if compliance_report.violations:
                lines.extend(
                    [
                        "| Violation Rule | Severity | Limit | Observed | Remediation Action |",
                        "| :--- | :--- | :---: | :---: | :--- |",
                    ]
                )
                for v in compliance_report.violations:
                    lines.append(
                        f"| `{v.rule}` | **{v.severity}** | {v.limit_value} | {v.current_value} | {v.message} |"
                    )

        lines.append("")
        lines.append("---")
        lines.append(
            "*FinTech Institutional Algorithmic Trading Risk Engine & Quantitative Analyzer*"
        )
        return "\n".join(lines)


class DataExporter:
    """Institutional CSV and JSON data export utility."""

    @staticmethod
    def export_portfolio_json(portfolio: Portfolio) -> str:
        """Export portfolio data structure as JSON."""
        return portfolio.model_dump_json(indent=2)

    @staticmethod
    def export_portfolio_positions_csv(portfolio: Portfolio) -> str:
        """Export position ledger as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
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
            ]
        )
        for p in portfolio.positions:
            writer.writerow(
                [
                    p.asset.symbol,
                    p.asset.name,
                    p.asset.asset_class.value,
                    p.asset.sector or "",
                    p.side.value,
                    p.quantity,
                    p.entry_price,
                    p.current_price,
                    p.market_value,
                    round(p.unrealized_pnl, 2),
                    round(p.unrealized_pnl_pct, 4),
                ]
            )
        return output.getvalue()

    @staticmethod
    def export_backtest_trades_csv(trades: List[BacktestTrade]) -> str:
        """Export completed backtest trades to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
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
            ]
        )
        for t in trades:
            writer.writerow(
                [
                    t.trade_id,
                    t.symbol,
                    t.side.value,
                    t.quantity,
                    t.entry_price,
                    t.exit_price,
                    t.entry_time.isoformat(),
                    t.exit_time.isoformat(),
                    t.pnl,
                    t.return_pct,
                    t.commission,
                    t.holding_period_bars,
                ]
            )
        return output.getvalue()

    @staticmethod
    def export_equity_curve_csv(result: BacktestResult) -> str:
        """Export sequential equity curve and daily returns to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["step", "nav", "daily_return", "rolling_var_95"])
        for i, nav in enumerate(result.equity_curve):
            ret = (
                result.daily_returns[i - 1]
                if i > 0 and (i - 1) < len(result.daily_returns)
                else 0.0
            )
            var_val = result.rolling_var_95[i] if i < len(result.rolling_var_95) else 0.0
            writer.writerow([i, nav, ret, var_val])
        return output.getvalue()

    @staticmethod
    def export_stress_test_json(results: List[StressTestResult]) -> str:
        """Export macroeconomic stress test outcomes as JSON."""
        return json.dumps([r.model_dump() for r in results], indent=2)

    @staticmethod
    def export_stress_test_csv(results: List[StressTestResult]) -> str:
        """Export macroeconomic stress test outcomes as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "scenario_id",
                "scenario_name",
                "portfolio_value_before",
                "portfolio_value_after",
                "absolute_impact",
                "percentage_impact",
                "capital_adequacy_passed",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.scenario.id,
                    r.scenario.name,
                    r.portfolio_value_before,
                    r.portfolio_value_after,
                    r.absolute_impact,
                    r.percentage_impact,
                    r.capital_adequacy_passed,
                ]
            )
        return output.getvalue()

    @classmethod
    def export_and_clean_trades(
        cls, trades: List[BacktestTrade], deduplicate: bool = True
    ) -> Tuple[List[CleanTradeRecord], ExportValidationSummary]:
        """Export domain backtest trades and run through PostgreSQL cleaning and validation."""
        from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline

        raw_trades: List[Dict[str, Any]] = []
        for t in trades:
            raw_trades.append(
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": (
                        t.entry_time.isoformat()
                        if hasattr(t.entry_time, "isoformat")
                        else str(t.entry_time)
                    ),
                    "exit_time": (
                        t.exit_time.isoformat()
                        if hasattr(t.exit_time, "isoformat")
                        else str(t.exit_time)
                    ),
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "commission": t.commission,
                    "holding_period_bars": t.holding_period_bars,
                }
            )
        return PostgreSQLExportPipeline.process_trades(raw_trades, deduplicate=deduplicate)

    @classmethod
    def export_and_clean_positions(
        cls, portfolio: Portfolio, deduplicate: bool = True
    ) -> Tuple[List[CleanPositionRecord], ExportValidationSummary]:
        """Export domain portfolio positions and run through PostgreSQL cleaning and validation."""
        from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline

        raw_positions: List[Dict[str, Any]] = []
        for p in portfolio.positions:
            raw_positions.append(
                {
                    "portfolio_id": portfolio.id,
                    "symbol": p.asset.symbol,
                    "name": p.asset.name,
                    "asset_class": (
                        p.asset.asset_class.value
                        if hasattr(p.asset.asset_class, "value")
                        else str(p.asset.asset_class)
                    ),
                    "sector": p.asset.sector or "",
                    "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
            )
        return PostgreSQLExportPipeline.process_positions(raw_positions, deduplicate=deduplicate)

    @classmethod
    def export_and_clean_equity_curve(
        cls,
        result: BacktestResult,
        portfolio_id: str = "BACKTEST-PORTFOLIO",
        start_time: Optional[datetime] = None,
    ) -> Tuple[List[CleanEquityRecord], ExportValidationSummary]:
        """Export backtest equity curve and run through PostgreSQL cleaning and validation."""
        from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline

        raw_records: List[Dict[str, Any]] = []
        t = start_time or datetime(2026, 1, 1, 9, 30)
        peak = result.initial_capital
        for i, val in enumerate(result.equity_curve):
            peak = max(peak, val)
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0.0
            raw_records.append(
                {
                    "curve_id": i + 1,
                    "portfolio_id": portfolio_id,
                    "timestamp": t + timedelta(days=i),
                    "portfolio_value": val,
                    "cash": val,
                    "drawdown": dd,
                    "drawdown_pct": dd_pct,
                }
            )
        return PostgreSQLExportPipeline.process_equity_curve(raw_records)

    @classmethod
    def export_to_sql_staging(
        cls,
        export_type: Union[ExportType, str],
        data: Union[Portfolio, List[BacktestTrade], BacktestResult, List[Dict[str, Any]]],
        portfolio_id: Optional[str] = None,
    ) -> str:
        """Convert domain models or raw records into PostgreSQL bulk INSERT staging statement."""
        from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline

        exp_type = ExportType(export_type)
        if isinstance(data, list) and (not data or isinstance(data[0], dict)):
            return PostgreSQLExportPipeline.generate_staging_insert_sql(exp_type, data)  # type: ignore[arg-type]

        if exp_type == ExportType.POSITIONS and isinstance(data, Portfolio):
            pid = portfolio_id or data.id
            raw_pos = [
                {
                    "portfolio_id": pid,
                    "symbol": p.asset.symbol,
                    "name": p.asset.name,
                    "asset_class": (
                        p.asset.asset_class.value
                        if hasattr(p.asset.asset_class, "value")
                        else str(p.asset.asset_class)
                    ),
                    "sector": p.asset.sector or "",
                    "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in data.positions
            ]
            return PostgreSQLExportPipeline.generate_staging_insert_sql(exp_type, raw_pos)

        if exp_type == ExportType.TRADES:
            trade_list: List[BacktestTrade] = (
                data.trades
                if isinstance(data, BacktestResult)
                else (data if isinstance(data, list) else [])  # type: ignore[assignment]
            )
            raw_tr = [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": (
                        t.entry_time.isoformat()
                        if hasattr(t.entry_time, "isoformat")
                        else str(t.entry_time)
                    ),
                    "exit_time": (
                        t.exit_time.isoformat()
                        if hasattr(t.exit_time, "isoformat")
                        else str(t.exit_time)
                    ),
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "commission": t.commission,
                    "holding_period_bars": t.holding_period_bars,
                }
                for t in trade_list
            ]
            return PostgreSQLExportPipeline.generate_staging_insert_sql(exp_type, raw_tr)

        if exp_type == ExportType.EQUITY_CURVE and isinstance(data, BacktestResult):
            pid = portfolio_id or "BACKTEST-PORTFOLIO"
            raw_eq = []
            t = datetime(2026, 1, 1, 9, 30)
            peak = data.initial_capital
            for i, val in enumerate(data.equity_curve):
                peak = max(peak, val)
                dd = peak - val
                dd_pct = dd / peak if peak > 0 else 0.0
                raw_eq.append(
                    {
                        "curve_id": i + 1,
                        "portfolio_id": pid,
                        "timestamp": t + timedelta(days=i),
                        "portfolio_value": val,
                        "cash": val,
                        "drawdown": dd,
                        "drawdown_pct": dd_pct,
                    }
                )
            return PostgreSQLExportPipeline.generate_staging_insert_sql(exp_type, raw_eq)

        raise ValueError(f"Unsupported data type for export {exp_type}: {type(data)}")

    @classmethod
    def export_pipeline_staging_script(
        cls,
        portfolio: Optional[Portfolio] = None,
        backtest_result: Optional[BacktestResult] = None,
        portfolio_id: str = "PORTFOLIO-1",
    ) -> str:
        """Construct full SQL script containing DDL, staging inserts, and cleaning execution."""
        from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline

        sections = [
            "-- ==========================================================================",
            "-- STAGING BUNDLE: Generated by Subsystems DataExporter",
            "-- ==========================================================================",
            PostgreSQLExportPipeline.get_schema_ddl(),
        ]
        if portfolio is not None:
            sql_pos = cls.export_to_sql_staging(ExportType.POSITIONS, portfolio, portfolio_id=portfolio_id)
            if sql_pos:
                sections.append(sql_pos)

        if backtest_result is not None:
            sql_tr = cls.export_to_sql_staging(ExportType.TRADES, backtest_result, portfolio_id=portfolio_id)
            if sql_tr:
                sections.append(sql_tr)
            sql_eq = cls.export_to_sql_staging(ExportType.EQUITY_CURVE, backtest_result, portfolio_id=portfolio_id)
            if sql_eq:
                sections.append(sql_eq)

        sections.append(PostgreSQLExportPipeline.get_clean_validate_sql())
        return "\n\n".join(sections)

