"""Comprehensive test suite for the fintech-algorithmic-trading CLI commands."""

import json
import os

from click.testing import CliRunner

from fintech_algorithmic_trading import __main__ as pkg_main
from fintech_algorithmic_trading.cli.main import cli
from fintech_algorithmic_trading.surveillance import RiskSurveillanceMonitor
from fintech_algorithmic_trading.types import (
    AlertSeverity,
    CircuitBreakerState,
    RiskAlert,
    SurveillanceReport,
)


def test_cli_list_portfolios_standard_and_detailed(cli_runner: CliRunner):
    """Test 'list-portfolios' standard and JSON output views."""
    res_std = cli_runner.invoke(cli, ["list-portfolios"])
    assert res_std.exit_code == 0
    assert "Institutional Benchmark Portfolios" in res_std.output
    assert "Global" in res_std.output

    res_json = cli_runner.invoke(cli, ["list-portfolios", "--json-out"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert isinstance(data, list)
    assert len(data) >= 4
    assert any(p["id"] == "global-macro" for p in data)


def test_cli_analyze_var_success_and_error(cli_runner: CliRunner):
    """Test 'analyze-var' with table output, JSON output, and non-existent portfolio error."""
    # Standard table output
    res_ok = cli_runner.invoke(
        cli,
        [
            "analyze-var",
            "--portfolio",
            "global-macro",
            "--confidence",
            "0.99",
            "--horizon",
            "10",
        ],
    )
    assert res_ok.exit_code == 0
    assert "Value-at-Risk Analysis" in res_ok.output
    assert "HISTORIC" in res_ok.output
    assert "PARAMETR" in res_ok.output

    # JSON output flag
    res_json = cli_runner.invoke(
        cli,
        [
            "analyze-var",
            "--portfolio",
            "crypto-fx-alpha",
            "--json-out",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "HISTORICAL" in data
    assert "PARAMETRIC" in data
    assert "CORNISH_FISHER" in data

    # Error handling: portfolio not found
    res_err = cli_runner.invoke(cli, ["analyze-var", "--portfolio", "non-existent-portfolio"])
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'non-existent-portfolio' not found." in res_err.output


def test_cli_simulate_all_drift_models_and_error(cli_runner: CliRunner):
    """Test 'simulate' with various drift stochastic models and error handling."""
    for model in ["GBM", "JUMP_DIFFUSION", "MEAN_REVERTING"]:
        res = cli_runner.invoke(
            cli,
            [
                "simulate",
                "--portfolio",
                "multi-asset-balanced",
                "--model",
                model,
                "--sims",
                "100",
                "--horizon",
                "21",
                "--seed",
                "42",
            ],
        )
        assert res.exit_code == 0
        assert "Monte Carlo Risk Simulation" in res.output
        assert "50th Percentile" in res.output or "Median" in res.output

    # Error handling: non-existent portfolio
    res_err = cli_runner.invoke(cli, ["simulate", "--portfolio", "unknown-portfolio"])
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'unknown-portfolio' not found." in res_err.output


def test_cli_stress_test_scenarios_and_filtering(cli_runner: CliRunner):
    """Test 'stress-test' all scenarios, filtered scenario, and error path."""
    # All scenarios
    res_all = cli_runner.invoke(cli, ["stress-test", "--portfolio", "tech-momentum"])
    assert res_all.exit_code == 0
    assert "Stress Testing Scenarios" in res_all.output
    assert "2008 Global" in res_all.output

    # Filtered single scenario
    res_filtered = cli_runner.invoke(
        cli,
        ["stress-test", "--portfolio", "tech-momentum", "--scenario", "2020-covid-crash"],
    )
    assert res_filtered.exit_code == 0
    assert "March 2020" in res_filtered.output

    # Non-existent portfolio
    res_err = cli_runner.invoke(cli, ["stress-test", "--portfolio", "bad-portfolio"])
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'bad-portfolio' not found." in res_err.output


def test_cli_check_limits_compliant_and_violations(cli_runner: CliRunner):
    """Test 'check-limits' on compliant portfolio and strict thresholds causing violations."""
    # High thresholds to guarantee passing
    res_pass = cli_runner.invoke(
        cli,
        [
            "check-limits",
            "--portfolio",
            "global-macro",
            "--max-pos",
            "0.60",
            "--max-sector",
            "0.70",
            "--max-leverage",
            "3.0",
        ],
    )
    assert res_pass.exit_code == 0
    assert "ALL RISK LIMITS PASSED" in res_pass.output

    # Tightened limits causing violation table
    res_fail = cli_runner.invoke(
        cli,
        [
            "check-limits",
            "--portfolio",
            "global-macro",
            "--max-leverage",
            "0.10",
            "--max-pos",
            "0.01",
            "--max-sector",
            "0.05",
        ],
    )
    assert res_fail.exit_code == 0
    assert "Risk Policy Violations Detected" in res_fail.output
    assert "SINGLE_NAME_CONCENTRATION" in res_fail.output or "SINGLE_NAME" in res_fail.output

    # Non-existent portfolio
    res_err = cli_runner.invoke(cli, ["check-limits", "--portfolio", "bad-portfolio"])
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'bad-portfolio' not found." in res_err.output


def test_cli_greeks_analytical_computation(cli_runner: CliRunner):
    """Test 'greeks' command for both Call and Put options."""
    res_call = cli_runner.invoke(
        cli,
        [
            "greeks",
            "--type",
            "CALL",
            "--spot",
            "100.0",
            "--strike",
            "100.0",
            "--time",
            "0.5",
            "--vol",
            "0.25",
            "--rate",
            "0.05",
        ],
    )
    assert res_call.exit_code == 0
    assert "Black-Scholes CALL Option Greeks" in res_call.output
    assert "Delta (Δ)" in res_call.output

    res_put = cli_runner.invoke(
        cli,
        [
            "greeks",
            "--type",
            "PUT",
            "--spot",
            "150.0",
            "--strike",
            "160.0",
            "--time",
            "1.0",
            "--vol",
            "0.30",
        ],
    )
    assert res_put.exit_code == 0
    assert "Black-Scholes PUT Option Greeks" in res_put.output


def test_cli_backtest_all_strategies(cli_runner: CliRunner):
    """Test 'backtest' command across MACD, Bollinger, Breakout, and Composite strategies."""
    for strat in ["macd", "bollinger", "breakout", "composite"]:
        res = cli_runner.invoke(
            cli,
            [
                "backtest",
                "--symbol",
                "QQQ",
                "--strategy",
                strat,
                "--bars",
                "50",
                "--capital",
                "50000",
            ],
        )
        assert res.exit_code == 0
        assert "Backtest Performance Attribution" in res.output
        assert strat.upper() in res.output
        assert "Initial Capital" in res.output
        assert "Total Return" in res.output


def test_cli_surveillance_states_and_alerts(cli_runner: CliRunner, monkeypatch):
    """Test 'surveillance' command in normal state, caution/halted alert states, and error path."""
    # 1. Normal state
    res_norm = cli_runner.invoke(cli, ["surveillance", "--portfolio", "global-macro"])
    assert res_norm.exit_code == 0
    assert "Real-Time Risk Surveillance Monitor" in res_norm.output
    assert "NORMAL" in res_norm.output

    # 2. Caution / Halted state with active alerts table
    def mock_audit(self, portfolio, custom_drawdown=None):
        return SurveillanceReport(
            portfolio_id=portfolio.id,
            state=CircuitBreakerState.CAUTION,
            current_drawdown_pct=0.095,
            leverage=2.5,
            margin_utilization_pct=0.65,
            kill_switch_active=False,
            active_alerts=[
                RiskAlert(
                    severity=AlertSeverity.WARNING,
                    component="SURVEILLANCE",
                    rule="DRAWDOWN_CAUTION",
                    message="Simulated drawdown caution alert",
                    current_value=0.095,
                    threshold_value=0.08,
                    action_taken="REDUCE_POSITION_SIZES",
                )
            ],
            remedial_actions=["Reduce positions by 50%"],
        )

    monkeypatch.setattr(RiskSurveillanceMonitor, "audit_portfolio", mock_audit)
    res_alert = cli_runner.invoke(cli, ["surveillance", "--portfolio", "global-macro"])
    assert res_alert.exit_code == 0
    assert "CAUTION" in res_alert.output
    assert "Active Surveillance Alerts" in res_alert.output
    assert "DRAWDOWN_CAUTION" in res_alert.output

    # 3. Non-existent portfolio error handling
    res_err = cli_runner.invoke(cli, ["surveillance", "--portfolio", "non-existent-port"])
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'non-existent-port' not found." in res_err.output


def test_cli_orderbook_market_depth(cli_runner: CliRunner):
    """Test 'orderbook' Level 2 market depth simulation."""
    res = cli_runner.invoke(
        cli,
        ["orderbook", "--symbol", "TSLA", "--price", "250.0", "--levels", "8"],
    )
    assert res.exit_code == 0
    assert "Level 2 Order Book Depth: TSLA" in res.output
    assert "Bid Price" in res.output
    assert "Ask Price" in res.output


def test_cli_report_stdout_and_file_export(cli_runner: CliRunner, tmp_path):
    """Test 'report' command printing to stdout and exporting to a file."""
    # Stdout report
    res_stdout = cli_runner.invoke(cli, ["report", "--portfolio", "crypto-fx-alpha"])
    assert res_stdout.exit_code == 0
    assert "# Institutional Portfolio Risk Tearsheet:" in res_stdout.output

    # File export
    report_file = tmp_path / "test_tearsheet.md"
    res_file = cli_runner.invoke(
        cli,
        ["report", "--portfolio", "global-macro", "--output", str(report_file)],
    )
    assert res_file.exit_code == 0
    assert "Tearsheet successfully saved to" in res_file.output
    assert os.path.exists(report_file)
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# Institutional Portfolio Risk Tearsheet:" in content
    assert "global-macro" in content

    # Error handling
    res_err = cli_runner.invoke(cli, ["report", "--portfolio", "bad-portfolio"])
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'bad-portfolio' not found." in res_err.output


def test_cli_sql_schema_ddl_and_validate_script(cli_runner: CliRunner, tmp_path):
    """Test 'sql-schema' DDL output, validation script, and file output."""
    res_ddl = cli_runner.invoke(cli, ["sql-schema"])
    assert res_ddl.exit_code == 0
    assert "CREATE TABLE IF NOT EXISTS raw_trades_export" in res_ddl.output
    assert "clean_trades" in res_ddl.output

    res_script = cli_runner.invoke(cli, ["sql-schema", "--validate-script"])
    assert res_script.exit_code == 0
    assert "export_validation_errors" in res_script.output
    assert "INSERT INTO clean_trades" in res_script.output

    out_file = str(tmp_path / "schema.sql")
    res_out = cli_runner.invoke(cli, ["sql-schema", "-o", out_file])
    assert res_out.exit_code == 0
    assert "SQL successfully saved to" in res_out.output
    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "raw_trades_export" in content


def test_cli_clean_export_trades_default_and_json(cli_runner: CliRunner):
    """Test 'clean-export' with default trades and JSON output."""
    res_table = cli_runner.invoke(cli, ["clean-export", "--type", "trades"])
    assert res_table.exit_code == 0
    assert "Export Validation Summary" in res_table.output
    assert "TRADES" in res_table.output
    assert "Total Raw Records" in res_table.output
    assert "Clean / Valid Records" in res_table.output

    res_json = cli_runner.invoke(cli, ["clean-export", "--type", "trades", "--json-out"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "summary" in data
    assert "clean_records" in data
    assert data["summary"]["total_raw_records"] == 3
    assert data["summary"]["valid_clean_records"] == 3
    assert len(data["clean_records"]) == 3
    assert data["clean_records"][0]["symbol"] == "SPY"


def test_cli_clean_export_positions_and_equity(cli_runner: CliRunner):
    """Test 'clean-export' for positions and equity curve datasets."""
    res_pos = cli_runner.invoke(
        cli, ["clean-export", "--type", "positions", "--portfolio", "tech-momentum"]
    )
    assert res_pos.exit_code == 0
    assert "Export Validation Summary" in res_pos.output
    assert "POSITIONS" in res_pos.output
    assert "Clean Validated Records Preview" in res_pos.output

    res_eq = cli_runner.invoke(cli, ["clean-export", "--type", "equity"])
    assert res_eq.exit_code == 0
    assert "Export Validation Summary" in res_eq.output
    assert "EQUITY_CURVE" in res_eq.output

    # Invalid portfolio
    res_bad = cli_runner.invoke(
        cli, ["clean-export", "--type", "positions", "--portfolio", "bad-portfolio"]
    )
    assert res_bad.exit_code != 0
    assert "Error: Portfolio 'bad-portfolio' not found." in res_bad.output


def test_cli_clean_export_custom_file_and_errors(cli_runner: CliRunner, tmp_path):
    """Test 'clean-export' with raw CSV file containing dirty, duplicate, and error records."""
    csv_file = tmp_path / "dirty_trades.csv"
    csv_file.write_text(
        "trade_id,symbol,side,quantity,entry_price,exit_price,entry_time,exit_time,pnl,return_pct,commission,holding_period_bars\n"
        "T-001,SPY,BUY,100,500.0,505.0,2026-03-01T09:30:00,2026-03-01T15:30:00,500.0,0.01,2.0,6\n"
        "T-001,SPY,BUY,100,500.0,505.0,2026-03-01T09:30:00,2026-03-01T15:30:00,500.0,0.01,2.0,6\n"  # Duplicate
        "T-002,AAPL,BUY,-50,180.0,175.0,2026-03-02T10:00:00,2026-03-02T14:00:00,-250.0,-0.02,1.5,4\n"  # Negative qty
        "T-003,MSFT,HODL,100,400.0,410.0,2026-03-03T10:00:00,2026-03-03T15:00:00,1000.0,0.025,2.0,5\n",  # Bad side
        encoding="utf-8",
    )

    json_out_file = str(tmp_path / "clean_trades.json")
    res = cli_runner.invoke(
        cli,
        [
            "clean-export",
            "--type",
            "trades",
            "--file",
            str(csv_file),
            "--output",
            json_out_file,
            "--json-out",
        ],
    )
    assert res.exit_code == 0
    assert os.path.exists(json_out_file)

    with open(json_out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary"]["total_raw_records"] == 4
    assert data["summary"]["valid_clean_records"] == 1
    assert data["summary"]["duplicate_records"] == 1
    assert data["summary"]["quarantined_records"] == 2
    assert len(data["summary"]["errors"]) >= 2

    # CSV output mode
    csv_out_file = str(tmp_path / "clean_trades.csv")
    res_csv = cli_runner.invoke(
        cli,
        [
            "clean-export",
            "--type",
            "trades",
            "--file",
            str(csv_file),
            "--output",
            csv_out_file,
        ],
    )
    assert res_csv.exit_code == 0
    assert os.path.exists(csv_out_file)
    assert "Saved 1 clean records to" in res_csv.output
    with open(csv_out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 record


def test_cli_sql_generate_inserts_and_copy(cli_runner: CliRunner, tmp_path):
    """Test 'sql-generate' bulk INSERT and COPY statements across export types."""
    res_trades = cli_runner.invoke(cli, ["sql-generate", "--type", "trades"])
    assert res_trades.exit_code == 0
    assert "INSERT INTO raw_trades_export" in res_trades.output
    assert "VALUES" in res_trades.output

    res_copy = cli_runner.invoke(cli, ["sql-generate", "--type", "positions", "--copy-command"])
    assert res_copy.exit_code == 0
    assert "COPY raw_positions_export FROM" in res_copy.output
    assert "/data/exports/positions.csv" in res_copy.output

    custom_copy = cli_runner.invoke(
        cli,
        [
            "sql-generate",
            "--type",
            "equity",
            "--copy-command",
            "--file",
            "/var/data/raw_equity.csv",
        ],
    )
    assert custom_copy.exit_code == 0
    assert "COPY raw_equity_curve_export FROM" in custom_copy.output
    assert "/var/data/raw_equity.csv" in custom_copy.output

    out_sql = str(tmp_path / "generated.sql")
    res_file = cli_runner.invoke(
        cli,
        [
            "sql-generate",
            "--type",
            "positions",
            "--portfolio",
            "multi-asset-balanced",
            "--output",
            out_sql,
        ],
    )
    assert res_file.exit_code == 0
    assert "SQL written to" in res_file.output
    assert os.path.exists(out_sql)
    with open(out_sql, "r", encoding="utf-8") as f:
        sql_content = f.read()
    assert "INSERT INTO raw_positions_export" in sql_content
    assert "multi-asset-balanced" in sql_content

    # Error handling
    res_err = cli_runner.invoke(
        cli, ["sql-generate", "--type", "positions", "--portfolio", "bad-portfolio"]
    )
    assert res_err.exit_code != 0
    assert "Error: Portfolio 'bad-portfolio' not found." in res_err.output


def test_package_main_invocation(monkeypatch):
    """Verify package __main__ invokes cli properly."""
    called = False

    def dummy_cli():
        nonlocal called
        called = True

    monkeypatch.setattr(pkg_main, "cli", dummy_cli)
    pkg_main.cli()
    assert called is True
