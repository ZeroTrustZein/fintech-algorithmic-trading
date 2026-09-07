"""Rich-enabled command-line interface for institutional algorithmic trading risk engine."""

from __future__ import annotations

import json
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fintech_algorithmic_trading.engine.greeks import calculate_option_greeks
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.engine.risk_engine import PortfolioRiskEngine
from fintech_algorithmic_trading.engine.var import VaRCalculator
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.types import (
    DriftModel,
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
def list_portfolios_cmd():
    """List available institutional benchmark portfolios."""
    portfolios = PortfolioRepository.get_benchmark_portfolios()
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


if __name__ == "__main__":
    cli()
