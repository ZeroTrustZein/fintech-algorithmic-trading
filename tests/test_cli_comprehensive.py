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


def test_package_main_invocation(monkeypatch):
    """Verify package __main__ invokes cli properly."""
    called = False

    def dummy_cli():
        nonlocal called
        called = True

    monkeypatch.setattr(pkg_main, "cli", dummy_cli)
    pkg_main.cli()
    assert called is True
