"""Rich-enabled command-line interface for institutional algorithmic trading risk engine."""

from __future__ import annotations

import csv
import io
import json
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fintech_algorithmic_trading.backtest import BacktestEngine
from fintech_algorithmic_trading.data import MarketDataFeed
from fintech_algorithmic_trading.engine.greeks import calculate_option_greeks
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.engine.risk_engine import PortfolioRiskEngine
from fintech_algorithmic_trading.engine.var import VaRCalculator
from fintech_algorithmic_trading.reporting import RiskReportGenerator
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.storage.sql_pipeline import PostgreSQLExportPipeline
from fintech_algorithmic_trading.strategy import (
    AlphaStrategy,
    CompositeAlphaAggregator,
    MeanRevertingBollingerStrategy,
    TrendFollowingMACDStrategy,
    VolatilityBreakoutStrategy,
)
from fintech_algorithmic_trading.surveillance import RiskSurveillanceMonitor
from fintech_algorithmic_trading.types import (
    BacktestConfig,
    CircuitBreakerState,
    DriftModel,
    ExportType,
    MonteCarloConfig,
    OptionType,
    RiskLimits,
)

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="fintech-algorithmic-trading")
def cli():
    """FinTech Algorithmic Trading Risk Engine & Monte Carlo VaR Analyzer."""
    pass


@cli.command("list-portfolios")
@click.option("--json-out", is_flag=True, help="Output raw JSON")
def list_portfolios_cmd(json_out: bool = False):
    """List available institutional benchmark portfolios."""
    portfolios = PortfolioRepository.get_benchmark_portfolios()
    if json_out:
        out = [
            {
                "id": p.id,
                "name": p.name,
                "positions_count": len(p.positions),
                "nav": p.net_asset_value,
                "gross_exposure": p.gross_exposure,
                "leverage": p.leverage,
            }
            for p in portfolios.values()
        ]
        console.print(json.dumps(out, indent=2))
        return

    table = Table(
        title="Institutional Benchmark Portfolios", show_header=True, header_style="bold cyan"
    )
    table.add_column("ID", style="bold")
    table.add_column("Name", style="white")
    table.add_column("Positions", justify="right")
    table.add_column("NAV ($)", justify="right")
    table.add_column("Gross Exposure ($)", justify="right")
    table.add_column("Leverage", justify="right")

    for p in portfolios.values():
        table.add_row(
            p.id,
            p.name,
            str(len(p.positions)),
            f"${p.net_asset_value:,.2f}",
            f"${p.gross_exposure:,.2f}",
            f"{p.leverage:.2f}x",
        )

    console.print(table)


@cli.command("analyze-var")
@click.option("--portfolio", "-p", default="global-macro", help="Portfolio preset ID")
@click.option(
    "--confidence", "-c", default=0.95, type=float, help="Confidence level (e.g. 0.95, 0.99)"
)
@click.option("--horizon", "-h", default=1, type=int, help="Holding horizon in trading days")
@click.option("--json-out", is_flag=True, help="Output raw JSON")
def analyze_var_cmd(portfolio: str, confidence: float, horizon: int, json_out: bool):
    """Compute Historical, Parametric, and Cornish-Fisher VaR & Expected Shortfall."""
    port = PortfolioRepository.get_portfolio(portfolio)
    if not port:
        console.print(f"[bold red]Error: Portfolio '{portfolio}' not found.[/bold red]")
        raise click.Abort()

    # Generate synthetic returns calibrated to portfolio volatility
    synthetic_returns = PortfolioRepository.generate_synthetic_returns(
        n_days=750,
        mean_daily=0.0003,
        vol_daily=0.012,
        fat_tailed=True,
    )

    results = VaRCalculator.evaluate_all(
        returns=synthetic_returns,
        portfolio_value=port.net_asset_value,
        confidence_level=confidence,
        horizon_days=horizon,
    )

    if json_out:
        out = {k.value: v.model_dump() for k, v in results.items()}
        console.print(json.dumps(out, indent=2))
        return

    table = Table(
        title=f"Value-at-Risk Analysis: {port.name} ({confidence * 100:.1f}% Confidence, {horizon}d Horizon)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Methodology", style="bold")
    table.add_column("VaR ($)", justify="right", style="red")
    table.add_column("VaR (%)", justify="right")
    table.add_column("CVaR / Expected Shortfall ($)", justify="right", style="bold red")
    table.add_column("CVaR (%)", justify="right")
    table.add_column("Skew / Kurt", justify="right")
    table.add_column("Latency", justify="right", style="dim")

    for method, res in results.items():
        skew_kurt = (
            f"{res.skewness:.2f} / {res.excess_kurtosis:.2f}" if res.skewness is not None else "N/A"
        )
        table.add_row(
            method.value,
            f"${res.var_amount:,.2f}",
            f"{res.var_pct * 100:.2f}%",
            f"${res.cvar_amount:,.2f}",
            f"{res.cvar_pct * 100:.2f}%",
            skew_kurt,
            f"{res.computation_time_ms:.2f} ms",
        )

    console.print(table)


@cli.command("simulate")
@click.option("--portfolio", "-p", default="tech-momentum", help="Portfolio preset ID")
@click.option("--sims", "-n", default=2000, type=int, help="Number of Monte Carlo paths")
@click.option("--horizon", "-h", default=21, type=int, help="Forecast horizon in days")
@click.option(
    "--model",
    "-m",
    type=click.Choice(["GBM", "JUMP_DIFFUSION", "MEAN_REVERTING"]),
    default="GBM",
    help="Stochastic process model",
)
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
def simulate_cmd(portfolio: str, sims: int, horizon: int, model: str, seed: int):
    """Run multi-asset correlated Monte Carlo stochastic simulation."""
    port = PortfolioRepository.get_portfolio(portfolio)
    if not port:
        console.print(f"[bold red]Error: Portfolio '{portfolio}' not found.[/bold red]")
        raise click.Abort()

    cfg = MonteCarloConfig(
        n_simulations=sims,
        horizon_days=horizon,
        time_steps=horizon,
        drift_model=DriftModel(model),
        random_seed=seed,
    )
    engine = MonteCarloEngine(config=cfg)

    with console.status(f"[bold green]Running {sims:,} paths ({model}) across {horizon} days..."):
        res = engine.simulate_portfolio(port)

    summary = Table(
        title=f"Monte Carlo Risk Simulation ({sims:,} paths | {model} | {horizon} Days)",
        show_header=False,
    )
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")

    summary.add_row("Initial Portfolio NAV", f"${res.initial_portfolio_value:,.2f}")
    summary.add_row("Mean Expected Terminal NAV", f"${res.mean_terminal_value:,.2f}")
    summary.add_row("Median (P50) Terminal NAV", f"${res.median_terminal_value:,.2f}")
    summary.add_row("5th Percentile (P5) NAV", f"${res.p5_terminal_value:,.2f}")
    summary.add_row("1st Percentile (P1) NAV", f"${res.p1_terminal_value:,.2f}")
    summary.add_row(
        "95% Horizon VaR",
        f"${res.simulated_var_95_amount:,.2f} ({res.simulated_var_95_pct * 100:.2f}%)",
    )
    summary.add_row(
        "95% Horizon CVaR",
        f"${res.simulated_cvar_95_amount:,.2f} ({res.simulated_cvar_95_pct * 100:.2f}%)",
    )
    summary.add_row(
        "99% Horizon VaR",
        f"${res.simulated_var_99_amount:,.2f} ({res.simulated_var_99_pct * 100:.2f}%)",
    )
    summary.add_row(
        "99% Horizon CVaR",
        f"${res.simulated_cvar_99_amount:,.2f} ({res.simulated_cvar_99_pct * 100:.2f}%)",
    )
    summary.add_row("Mean Path Max Drawdown", f"{res.max_drawdown_mean_pct * 100:.2f}%")
    summary.add_row("95th Percentile Max Drawdown", f"{res.max_drawdown_p95_pct * 100:.2f}%")
    summary.add_row("Probability of Capital Loss", f"{res.probability_of_loss * 100:.2f}%")
    summary.add_row("Execution Time", f"{res.computation_time_ms:.2f} ms")

    console.print(summary)


@cli.command("stress-test")
@click.option("--portfolio", "-p", default="multi-asset-balanced", help="Portfolio preset ID")
@click.option("--scenario", "-s", default=None, help="Specific scenario ID (default: all)")
def stress_test_cmd(portfolio: str, scenario: Optional[str]):
    """Execute macroeconomic crisis stress test scenarios."""
    port = PortfolioRepository.get_portfolio(portfolio)
    if not port:
        console.print(f"[bold red]Error: Portfolio '{portfolio}' not found.[/bold red]")
        raise click.Abort()

    scenarios = PortfolioRepository.get_stress_scenarios()
    if scenario:
        if scenario not in scenarios:
            console.print(f"[bold red]Error: Scenario '{scenario}' not found.[/bold red]")
            raise click.Abort()
        scenarios = {scenario: scenarios[scenario]}

    engine = PortfolioRiskEngine()
    table = Table(
        title=f"Stress Testing Scenarios: {port.name} (NAV: ${port.net_asset_value:,.2f})",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Scenario", style="bold")
    table.add_column("NAV Post-Shock ($)", justify="right")
    table.add_column("P&L Impact ($)", justify="right")
    table.add_column("Impact (%)", justify="right")
    table.add_column("Capital Adequacy", justify="center")

    for sc in scenarios.values():
        res = engine.run_stress_test(port, sc)
        pass_str = (
            "[bold green]PASS[/bold green]"
            if res.capital_adequacy_passed
            else "[bold red]FAIL[/bold red]"
        )
        table.add_row(
            sc.name,
            f"${res.portfolio_value_after:,.2f}",
            f"${res.absolute_impact:,.2f}",
            f"{res.percentage_impact * 100:.2f}%",
            pass_str,
        )

    console.print(table)


@cli.command("check-limits")
@click.option("--portfolio", "-p", default="global-macro", help="Portfolio preset ID")
@click.option("--max-leverage", default=3.0, type=float, help="Max gross leverage limit")
@click.option("--max-pos", default=0.30, type=float, help="Max single position concentration")
@click.option("--max-sector", default=0.45, type=float, help="Max sector concentration")
def check_limits_cmd(portfolio: str, max_leverage: float, max_pos: float, max_sector: float):
    """Audit portfolio against risk policy limits and mandate restrictions."""
    port = PortfolioRepository.get_portfolio(portfolio)
    if not port:
        console.print(f"[bold red]Error: Portfolio '{portfolio}' not found.[/bold red]")
        raise click.Abort()

    custom_limits = RiskLimits(
        max_leverage=max_leverage,
        max_single_position_pct=max_pos,
        max_sector_pct=max_sector,
    )
    engine = PortfolioRiskEngine(limits=custom_limits)
    report = engine.check_compliance(port)

    if report.passed:
        console.print(
            Panel.fit(
                f"[bold green]✓ ALL RISK LIMITS PASSED[/bold green]\n"
                f"Portfolio: {report.portfolio_name}\n"
                f"Leverage: {report.current_leverage:.2f}x (Limit: {max_leverage:.2f}x)\n"
                f"Top Position ({report.max_single_position_symbol}): {report.max_single_position_pct * 100:.1f}%\n"
                f"Top Sector ({report.max_sector_name}): {report.max_sector_pct * 100:.1f}%\n"
                f"Est. 1-day 95% VaR: {report.estimated_var_95_pct * 100:.2f}%",
                title="Risk Policy Audit",
            )
        )
    else:
        table = Table(
            title=f"Risk Policy Violations Detected: {report.portfolio_name}",
            show_header=True,
            header_style="bold red",
        )
        table.add_column("Rule", style="bold")
        table.add_column("Severity", style="bold yellow")
        table.add_column("Limit", justify="right")
        table.add_column("Current", justify="right", style="bold red")
        table.add_column("Details")

        for v in report.violations:
            table.add_row(
                v.rule,
                v.severity,
                str(v.limit_value),
                str(v.current_value),
                v.message,
            )
        console.print(table)


@cli.command("greeks")
@click.option("--type", "option_type", type=click.Choice(["CALL", "PUT"]), default="CALL")
@click.option("--spot", "-s", default=100.0, type=float, help="Underlying spot price")
@click.option("--strike", "-k", default=100.0, type=float, help="Strike price")
@click.option("--time", "-t", default=0.25, type=float, help="Years to expiration")
@click.option("--vol", "-v", default=0.20, type=float, help="Implied volatility")
@click.option("--rate", "-r", default=0.045, type=float, help="Risk-free rate")
def greeks_cmd(
    option_type: str,
    spot: float,
    strike: float,
    time: float,
    vol: float,
    rate: float,
):
    """Compute Black-Scholes analytical option price and Greeks."""
    greeks = calculate_option_greeks(
        option_type=OptionType(option_type),
        spot_price=spot,
        strike_price=strike,
        time_to_expiry_years=time,
        volatility=vol,
        risk_free_rate=rate,
    )

    table = Table(title=f"Black-Scholes {option_type} Option Greeks", show_header=False)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Theoretical Price", f"${greeks.price:.4f}")
    table.add_row("Delta (Δ)", f"{greeks.delta:.4f}")
    table.add_row("Gamma (Γ)", f"{greeks.gamma:.6f}")
    table.add_row("Vega (V)", f"{greeks.vega:.4f} per 1% vol")
    table.add_row("Theta (Θ)", f"${greeks.theta:.4f} per day")
    table.add_row("Rho (ρ)", f"{greeks.rho:.4f} per 100 bps")

    console.print(table)


@cli.command("backtest")
@click.option("--symbol", "-s", default="SPY", help="Asset ticker symbol")
@click.option(
    "--strategy",
    "-t",
    type=click.Choice(["macd", "bollinger", "breakout", "composite"]),
    default="composite",
    help="Alpha strategy model",
)
@click.option("--bars", "-b", default=252, type=int, help="Historical bars count")
@click.option("--capital", "-c", default=100_000.0, type=float, help="Initial capital")
def backtest_cmd(symbol: str, strategy: str, bars: int, capital: float):
    """Run event-driven backtest with realistic execution slippage and commissions."""
    feed = MarketDataFeed()
    history = feed.generate_synthetic_history(
        symbol=symbol,
        start_price=500.0,
        n_bars=bars,
        volatility_annual=0.18,
        drift_annual=0.10,
    )

    strat: AlphaStrategy | CompositeAlphaAggregator
    if strategy == "macd":
        strat = TrendFollowingMACDStrategy()
    elif strategy == "bollinger":
        strat = MeanRevertingBollingerStrategy()
    elif strategy == "breakout":
        strat = VolatilityBreakoutStrategy()
    else:
        strat = CompositeAlphaAggregator()

    cfg = BacktestConfig(initial_capital=capital)
    engine = BacktestEngine(config=cfg)

    with console.status(
        f"[bold green]Running {strategy.upper()} backtest on {symbol} across {bars} bars..."
    ):
        res = engine.run(symbol=symbol, bars=history, strategy=strat)

    m = res.metrics
    table = Table(
        title=f"Backtest Performance Attribution: {symbol} ({strategy.upper()})",
        show_header=False,
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Initial Capital", f"${res.initial_capital:,.2f}")
    table.add_row("Final Capital", f"${res.final_capital:,.2f}")
    table.add_row(
        "Total Return",
        f"[{'green' if m.total_return_pct >= 0 else 'red'}]{m.total_return_pct * 100:.2f}%[/{'green' if m.total_return_pct >= 0 else 'red'}]",
    )
    table.add_row("Annualized Return", f"{m.annualized_return * 100:.2f}%")
    table.add_row("Annualized Volatility", f"{m.annualized_volatility * 100:.2f}%")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{m.sortino_ratio:.2f}")
    table.add_row("Calmar Ratio", f"{m.calmar_ratio:.2f}")
    table.add_row("Max Drawdown", f"[red]{m.max_drawdown_pct * 100:.2f}%[/red]")
    table.add_row("Max Drawdown Duration", f"{m.max_drawdown_duration_bars} bars")
    table.add_row("Win Rate", f"{m.win_rate * 100:.1f}%")
    table.add_row("Profit Factor", f"{m.profit_factor:.2f}")
    table.add_row("Trade Expectancy", f"${m.expectancy:,.2f}")
    table.add_row(
        "Total Trades",
        f"{m.total_trades} (Win: {m.profitable_trades}, Loss: {m.loss_making_trades})",
    )
    table.add_row("Execution Latency", f"{res.computation_time_ms:.2f} ms")

    console.print(table)


@cli.command("surveillance")
@click.option("--portfolio", "-p", default="global-macro", help="Portfolio preset ID")
def surveillance_cmd(portfolio: str):
    """Run real-time risk surveillance audit and circuit breaker status check."""
    port = PortfolioRepository.get_portfolio(portfolio)
    if not port:
        console.print(f"[bold red]Error: Portfolio '{portfolio}' not found.[/bold red]")
        raise click.Abort()

    monitor = RiskSurveillanceMonitor()
    report = monitor.audit_portfolio(port)

    status_color = (
        "green"
        if report.state == CircuitBreakerState.NORMAL
        else "yellow"
        if report.state == CircuitBreakerState.CAUTION
        else "red"
    )
    console.print(
        Panel.fit(
            f"Portfolio: [bold]{port.name}[/bold] (ID: `{port.id}`)\n"
            f"Circuit Breaker Status: [{status_color}][bold]{report.state.value}[/bold][/{status_color}]\n"
            f"Current Drawdown: {report.current_drawdown_pct * 100:.2f}%\n"
            f"Gross Leverage: {report.leverage:.2f}x\n"
            f"Margin Utilization: {report.margin_utilization_pct * 100:.1f}%\n"
            f"Kill-Switch Active: {'[bold red]YES[/bold red]' if report.kill_switch_active else '[bold green]NO[/bold green]'}\n"
            f"Active Alerts Count: {len(report.active_alerts)}",
            title="Real-Time Risk Surveillance Monitor",
        )
    )

    if report.active_alerts:
        table = Table(title="Active Surveillance Alerts", show_header=True, header_style="bold red")
        table.add_column("Severity", style="bold")
        table.add_column("Rule")
        table.add_column("Current")
        table.add_column("Threshold")
        table.add_column("Details")
        for a in report.active_alerts:
            table.add_row(
                a.severity.value,
                a.rule,
                str(a.current_value),
                str(a.threshold_value),
                a.message,
            )
        console.print(table)


@cli.command("orderbook")
@click.option("--symbol", "-s", default="SPY", help="Asset ticker symbol")
@click.option("--price", "-p", default=540.0, type=float, help="Reference spot price")
@click.option("--levels", "-l", default=5, type=int, help="Depth levels to display")
def orderbook_cmd(symbol: str, price: float, levels: int):
    """Simulate and display Level 2 Order Book market depth."""
    feed = MarketDataFeed()
    feed.generate_synthetic_history(symbol=symbol, start_price=price, n_bars=10)
    ob = feed.generate_order_book(symbol=symbol, levels=levels)

    table = Table(
        title=f"Level 2 Order Book Depth: {symbol} (Best Bid: {ob.best_bid}, Best Ask: {ob.best_ask})",
        show_header=True,
    )
    table.add_column("Bid Orders", justify="right", style="dim")
    table.add_column("Bid Size", justify="right", style="green")
    table.add_column("Bid Price ($)", justify="right", style="bold green")
    table.add_column("Ask Price ($)", justify="right", style="bold red")
    table.add_column("Ask Size", justify="right", style="red")
    table.add_column("Ask Orders", justify="right", style="dim")

    for i in range(min(len(ob.bids), len(ob.asks))):
        bid = ob.bids[i]
        ask = ob.asks[i]
        table.add_row(
            str(bid.order_count),
            f"{bid.size:,.0f}",
            f"${bid.price:.2f}",
            f"${ask.price:.2f}",
            f"{ask.size:,.0f}",
            str(ask.order_count),
        )

    console.print(table)


@cli.command("report")
@click.option("--portfolio", "-p", default="global-macro", help="Portfolio preset ID")
@click.option("--output", "-o", default=None, help="Save markdown report to file path")
def report_cmd(portfolio: str, output: Optional[str]):
    """Generate institutional quantitative risk tearsheet."""
    port = PortfolioRepository.get_portfolio(portfolio)
    if not port:
        console.print(f"[bold red]Error: Portfolio '{portfolio}' not found.[/bold red]")
        raise click.Abort()

    # Generate synthetic returns calibrated to portfolio
    synthetic_returns = PortfolioRepository.generate_synthetic_returns(
        n_days=750,
        mean_daily=0.0003,
        vol_daily=0.012,
        fat_tailed=True,
    )

    var_res = VaRCalculator.evaluate_all(
        returns=synthetic_returns,
        portfolio_value=port.net_asset_value,
    )

    mc_engine = MonteCarloEngine(config=MonteCarloConfig(n_simulations=500, horizon_days=21))
    mc_res = mc_engine.simulate_portfolio(port)

    risk_engine = PortfolioRiskEngine()
    scenarios = PortfolioRepository.get_stress_scenarios()
    stress_res = [risk_engine.run_stress_test(port, sc) for sc in scenarios.values()]
    comp_rep = risk_engine.check_compliance(port)

    tearsheet = RiskReportGenerator.generate_markdown_tearsheet(
        portfolio=port,
        var_results=var_res,
        monte_carlo_res=mc_res,
        stress_results=stress_res,
        compliance_report=comp_rep,
    )

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(tearsheet)
        console.print(f"[bold green]Tearsheet successfully saved to {output}[/bold green]")
    else:
        console.print(tearsheet)


@cli.command("sql-schema")
@click.option(
    "--validate-script",
    is_flag=True,
    help="Display clean and validate SQL script instead of table DDL",
)
@click.option("--output", "-o", default=None, help="Save SQL output to file path")
def sql_schema_cmd(validate_script: bool, output: Optional[str]):
    """Display or export PostgreSQL schema DDL and validation scripts."""
    sql = (
        PostgreSQLExportPipeline.get_clean_validate_sql()
        if validate_script
        else PostgreSQLExportPipeline.get_schema_ddl()
    )
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(sql)
        console.print(f"[bold green]SQL successfully saved to {output}[/bold green]")
    else:
        console.print(sql)


@cli.command("clean-export")
@click.option(
    "--type",
    "-t",
    "export_type",
    type=click.Choice(["trades", "positions", "equity", "equity_curve"], case_sensitive=False),
    default="trades",
    help="Export type to clean and validate",
)
@click.option("--file", "-f", default=None, help="Path to raw CSV export file")
@click.option("--portfolio", "-p", default=None, help="Portfolio preset ID (for positions export)")
@click.option("--output", "-o", default=None, help="Save clean records to CSV or JSON file")
@click.option("--json-out", is_flag=True, help="Output validation summary and records as JSON")
def clean_export_cmd(
    export_type: str,
    file: Optional[str],
    portfolio: Optional[str],
    output: Optional[str],
    json_out: bool,
):
    """Clean, deduplicate, and validate raw exports with PostgreSQL pipeline logic."""
    exp_t = export_type.lower()
    type_map = {
        "trades": ExportType.TRADES,
        "positions": ExportType.POSITIONS,
        "equity": ExportType.EQUITY_CURVE,
        "equity_curve": ExportType.EQUITY_CURVE,
    }
    exp_enum = type_map[exp_t]
    default_port_id: Optional[str] = None

    if file:
        with open(file, "r", encoding="utf-8") as f:
            csv_text = f.read()
    elif exp_enum == ExportType.POSITIONS:
        port_id = portfolio or "global-macro"
        default_port_id = port_id
        port = PortfolioRepository.get_portfolio(port_id)
        if not port:
            console.print(f"[bold red]Error: Portfolio '{port_id}' not found.[/bold red]")
            raise click.Abort()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
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
        ])
        for p in port.positions:
            writer.writerow([
                port.id,
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
            ])
        csv_text = buf.getvalue()
    elif exp_enum == ExportType.EQUITY_CURVE:
        default_port_id = portfolio or "global-macro"
        csv_text = (
            "portfolio_id,timestamp,portfolio_value,cash,drawdown,drawdown_pct\n"
            f"{default_port_id},2026-03-01T00:00:00,100000000.0,15000000.0,0.0,0.0\n"
            f"{default_port_id},2026-03-02T00:00:00,100500000.0,14800000.0,0.0,0.0\n"
            f"{default_port_id},2026-03-03T00:00:00,99800000.0,15200000.0,700000.0,0.007\n"
        )
    else:  # trades
        csv_text = (
            "trade_id,symbol,side,quantity,entry_price,exit_price,entry_time,exit_time,pnl,return_pct,commission,holding_period_bars\n"
            "T-001,SPY,BUY,100,500.0,505.0,2026-03-01T09:30:00,2026-03-01T15:30:00,500.0,0.01,2.0,6\n"
            "T-002,AAPL,BUY,50,180.0,175.0,2026-03-02T10:00:00,2026-03-02T14:00:00,-250.0,-0.0278,1.5,4\n"
            "T-003,MSFT,BUY,200,400.0,408.0,2026-03-03T11:00:00,2026-03-03T16:00:00,1600.0,0.02,3.0,10\n"
        )

    clean_records, summary = PostgreSQLExportPipeline.process_csv(
        csv_content=csv_text,
        export_type=exp_enum,
        default_portfolio_id=default_port_id,
    )

    if json_out:
        out = {
            "summary": summary.model_dump(),
            "clean_records": [r.model_dump() for r in clean_records],
        }
        console.print(json.dumps(out, indent=2, default=str))
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(json.dumps(out, indent=2, default=str))
            console.print(f"[bold green]Output saved to {output}[/bold green]")
        return

    table = Table(
        title=f"PostgreSQL Export Validation Summary: {exp_enum.value.upper()}",
        show_header=False,
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total Raw Records", str(summary.total_raw_records))
    table.add_row("Clean / Valid Records", f"[green]{summary.valid_clean_records}[/green]")
    table.add_row("Duplicate Records Dropped", f"[yellow]{summary.duplicate_records}[/yellow]")
    err_style = "red" if summary.quarantined_records > 0 else "dim"
    table.add_row(
        "Quarantined Error Records",
        f"[{err_style}]{summary.quarantined_records}[/{err_style}]",
    )
    pass_rate = (
        (summary.valid_clean_records / summary.total_raw_records * 100.0)
        if summary.total_raw_records > 0
        else 0.0
    )
    table.add_row("Validation Pass Rate", f"{pass_rate:.1f}%")
    console.print(table)

    if summary.errors:
        err_table = Table(
            title=f"Validation Errors Encountered ({len(summary.errors)} issues)",
            show_header=True,
            header_style="bold red",
        )
        err_table.add_column("Record", style="bold")
        err_table.add_column("Rule")
        err_table.add_column("Message")
        for err in summary.errors[:15]:
            err_table.add_row(
                str(err.record_identifier or "N/A"),
                str(err.rule_name),
                err.error_message,
            )
        console.print(err_table)

    if clean_records:
        rec_table = Table(
            title=f"Clean Validated Records Preview (First {min(len(clean_records), 10)} of {len(clean_records)})",
            show_header=True,
            header_style="bold cyan",
        )
        if exp_t == "trades":
            rec_table.add_column("Trade ID", style="bold")
            rec_table.add_column("Symbol")
            rec_table.add_column("Side")
            rec_table.add_column("Qty", justify="right")
            rec_table.add_column("Entry ($)", justify="right")
            rec_table.add_column("Exit ($)", justify="right")
            rec_table.add_column("PnL ($)", justify="right")
            rec_table.add_column("Return (%)", justify="right")
            for r in clean_records[:10]:
                pnl_color = "green" if r.pnl >= 0 else "red"
                rec_table.add_row(
                    r.trade_id,
                    r.symbol,
                    r.side,
                    f"{r.quantity:,.0f}",
                    f"${r.entry_price:.2f}",
                    f"${r.exit_price:.2f}",
                    f"[{pnl_color}]${r.pnl:,.2f}[/{pnl_color}]",
                    f"{r.return_pct * 100:.2f}%",
                )
        elif exp_t == "positions":
            rec_table.add_column("Portfolio", style="bold")
            rec_table.add_column("Symbol")
            rec_table.add_column("Side")
            rec_table.add_column("Qty", justify="right")
            rec_table.add_column("Current ($)", justify="right")
            rec_table.add_column("Market Value ($)", justify="right")
            rec_table.add_column("Unrealized PnL ($)", justify="right")
            for r in clean_records[:10]:
                pnl_color = "green" if r.unrealized_pnl >= 0 else "red"
                rec_table.add_row(
                    r.portfolio_id,
                    r.symbol,
                    r.side,
                    f"{r.quantity:,.0f}",
                    f"${r.current_price:.2f}",
                    f"${r.market_value:,.2f}",
                    f"[{pnl_color}]${r.unrealized_pnl:,.2f}[/{pnl_color}]",
                )
        else:  # equity
            rec_table.add_column("Portfolio", style="bold")
            rec_table.add_column("Timestamp")
            rec_table.add_column("NAV ($)", justify="right")
            rec_table.add_column("Cash ($)", justify="right")
            rec_table.add_column("Drawdown ($)", justify="right")
            rec_table.add_column("Drawdown (%)", justify="right")
            for r in clean_records[:10]:
                rec_table.add_row(
                    r.portfolio_id,
                    str(r.timestamp),
                    f"${r.portfolio_value:,.2f}",
                    f"${r.cash:,.2f}",
                    f"${r.drawdown:,.2f}",
                    f"{r.drawdown_pct * 100:.2f}%",
                )
        console.print(rec_table)

    if output:
        if output.endswith(".json"):
            with open(output, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        [r.model_dump() for r in clean_records],
                        indent=2,
                        default=str,
                    )
                )
        else:
            with open(output, "w", encoding="utf-8", newline="") as f:
                if clean_records:
                    fieldnames = list(clean_records[0].model_dump().keys())
                    dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
                    dict_writer.writeheader()
                    for rec in clean_records:
                        dict_writer.writerow(rec.model_dump())
        console.print(f"[bold green]Saved {len(clean_records)} clean records to {output}[/bold green]")


@cli.command("sql-generate")
@click.option(
    "--type",
    "-t",
    "export_type",
    type=click.Choice(["trades", "positions", "equity", "equity_curve"], case_sensitive=False),
    default="trades",
    help="Export type for SQL generation",
)
@click.option("--file", "-f", default=None, help="Path to raw CSV export file")
@click.option("--portfolio", "-p", default=None, help="Portfolio preset ID (for positions export)")
@click.option(
    "--copy-command",
    is_flag=True,
    help="Generate PostgreSQL COPY command instead of INSERTs",
)
@click.option("--output", "-o", default=None, help="Save SQL output to file path")
def sql_generate_cmd(
    export_type: str,
    file: Optional[str],
    portfolio: Optional[str],
    copy_command: bool,
    output: Optional[str],
):
    """Generate PostgreSQL staging COPY command or bulk INSERT SQL statements."""
    exp_t = export_type.lower()
    type_map = {
        "trades": ExportType.TRADES,
        "positions": ExportType.POSITIONS,
        "equity": ExportType.EQUITY_CURVE,
        "equity_curve": ExportType.EQUITY_CURVE,
    }
    exp_enum = type_map[exp_t]
    table_name = {
        ExportType.TRADES: "raw_trades_export",
        ExportType.POSITIONS: "raw_positions_export",
        ExportType.EQUITY_CURVE: "raw_equity_curve_export",
    }[exp_enum]

    if copy_command:
        file_path = file or f"/data/exports/{exp_enum.value}.csv"
        sql = PostgreSQLExportPipeline.generate_copy_command(table_name, file_path)
    else:
        if file:
            with open(file, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
        elif exp_enum == ExportType.POSITIONS:
            port_id = portfolio or "global-macro"
            port = PortfolioRepository.get_portfolio(port_id)
            if not port:
                console.print(f"[bold red]Error: Portfolio '{port_id}' not found.[/bold red]")
                raise click.Abort()
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
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
            ])
            for p in port.positions:
                writer.writerow([
                    port.id,
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
                ])
            records = list(csv.DictReader(io.StringIO(buf.getvalue())))
        elif exp_enum == ExportType.EQUITY_CURVE:
            csv_text = (
                "portfolio_id,timestamp,portfolio_value,cash,drawdown,drawdown_pct\n"
                "global-macro,2026-03-01T00:00:00,100000000.0,15000000.0,0.0,0.0\n"
                "global-macro,2026-03-02T00:00:00,100500000.0,14800000.0,0.0,0.0\n"
            )
            records = list(csv.DictReader(io.StringIO(csv_text)))
        else:  # trades
            csv_text = (
                "trade_id,symbol,side,quantity,entry_price,exit_price,entry_time,exit_time,pnl,return_pct,commission,holding_period_bars\n"
                "T-001,SPY,BUY,100,500.0,505.0,2026-03-01T09:30:00,2026-03-01T15:30:00,500.0,0.01,2.0,6\n"
                "T-002,AAPL,BUY,50,180.0,175.0,2026-03-02T10:00:00,2026-03-02T14:00:00,-250.0,-0.0278,1.5,4\n"
            )
            records = list(csv.DictReader(io.StringIO(csv_text)))

        sql = PostgreSQLExportPipeline.generate_staging_insert_sql(exp_enum, records)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(sql)
        console.print(f"[bold green]SQL written to {output}[/bold green]")
    else:
        console.print(sql)


if __name__ == "__main__":
    cli()
