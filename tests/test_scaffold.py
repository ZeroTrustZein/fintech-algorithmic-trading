"""Comprehensive scaffold verification test suite for fintech-algorithmic-trading."""

import numpy as np
from click.testing import CliRunner

from fintech_algorithmic_trading import __version__
from fintech_algorithmic_trading.cli.main import cli
from fintech_algorithmic_trading.engine.greeks import calculate_option_greeks
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.engine.risk_engine import PortfolioRiskEngine
from fintech_algorithmic_trading.engine.var import (
    VaRCalculator,
    calculate_cornish_fisher_var,
    calculate_historical_var,
    calculate_parametric_var,
)
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    DriftModel,
    MonteCarloConfig,
    OptionType,
    Portfolio,
    Position,
    PositionType,
    RiskLimits,
    VaRMethod,
)


def test_package_version():
    """Verify package version is set."""
    assert __version__ == "0.1.0"


def test_asset_and_position_models():
    """Test asset creation and position calculations for long and short sides."""
    asset = Asset(
        symbol="XYZ",
        name="XYZ Corp",
        asset_class=AssetClass.EQUITY,
        sector="Technology",
        current_price=150.0,
        volatility_annual=0.25,
    )
    long_pos = Position(
        asset=asset,
        quantity=10.0,
        entry_price=100.0,
        current_price=150.0,
        side=PositionType.LONG,
    )
    assert long_pos.market_value == 1500.0
    assert long_pos.cost_basis == 1000.0
    assert long_pos.unrealized_pnl == 500.0
    assert long_pos.unrealized_pnl_pct == 0.50

    short_pos = Position(
        asset=asset,
        quantity=10.0,
        entry_price=150.0,
        current_price=120.0,
        side=PositionType.SHORT,
    )
    assert short_pos.market_value == 1200.0
    assert short_pos.unrealized_pnl == 300.0


def test_portfolio_nav_and_exposure(simple_portfolio: Portfolio):
    """Test portfolio NAV, gross exposure, leverage, and weight distributions."""
    # Cash = 10000, pos1 (100 * 100 = 10000), pos2 (200 * 50 = 10000)
    assert simple_portfolio.total_positions_value == 20000.0
    assert simple_portfolio.net_asset_value == 30000.0
    assert simple_portfolio.gross_exposure == 20000.0
    assert round(simple_portfolio.leverage, 4) == round(20000.0 / 30000.0, 4)

    weights = simple_portfolio.asset_weights
    assert weights["AAA"] == 0.5
    assert weights["BBB"] == 0.5


def test_historical_var_calculation(sample_daily_returns: np.ndarray):
    """Test Historical VaR and CVaR calculations."""
    val = 1000000.0
    res_95 = calculate_historical_var(
        sample_daily_returns, portfolio_value=val, confidence_level=0.95
    )
    res_99 = calculate_historical_var(
        sample_daily_returns, portfolio_value=val, confidence_level=0.99
    )

    assert res_95.method == VaRMethod.HISTORICAL
    assert res_95.var_amount > 0
    assert res_95.cvar_amount >= res_95.var_amount
    assert res_99.var_amount >= res_95.var_amount
    assert res_95.confidence_level == 0.95


def test_parametric_var_calculation(sample_daily_returns: np.ndarray):
    """Test Gaussian parametric VaR and CVaR calculations."""
    val = 500000.0
    res_95 = calculate_parametric_var(
        sample_daily_returns, portfolio_value=val, confidence_level=0.95
    )
    res_99 = calculate_parametric_var(
        sample_daily_returns, portfolio_value=val, confidence_level=0.99
    )

    assert res_95.method == VaRMethod.PARAMETRIC
    assert res_95.var_amount > 0
    assert res_95.cvar_amount > res_95.var_amount
    assert res_99.var_amount > res_95.var_amount


def test_cornish_fisher_var_calculation(sample_daily_returns: np.ndarray):
    """Test Cornish-Fisher expansion VaR adjusting for higher moments."""
    val = 1000000.0
    res = calculate_cornish_fisher_var(
        sample_daily_returns, portfolio_value=val, confidence_level=0.95
    )

    assert res.method == VaRMethod.CORNISH_FISHER
    assert res.var_amount > 0
    assert res.skewness is not None
    assert res.excess_kurtosis is not None


def test_var_calculator_unified_evaluation(sample_daily_returns: np.ndarray):
    """Test unified VaR evaluation across all three methodologies."""
    res_map = VaRCalculator.evaluate_all(sample_daily_returns, portfolio_value=1000000.0)
    assert VaRMethod.HISTORICAL in res_map
    assert VaRMethod.PARAMETRIC in res_map
    assert VaRMethod.CORNISH_FISHER in res_map


def test_portfolio_parametric_var_with_covariance(simple_portfolio: Portfolio):
    """Test multi-asset portfolio parametric VaR with covariance matrix."""
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]], dtype=np.float64)
    symbols = ["AAA", "BBB"]
    res = VaRCalculator.calculate_portfolio_parametric_var(
        portfolio=simple_portfolio,
        asset_covariance_matrix=cov,
        symbols_order=symbols,
        confidence_level=0.95,
    )
    assert res.var_amount > 0
    assert res.portfolio_value == simple_portfolio.net_asset_value


def test_monte_carlo_engine_gbm(simple_portfolio: Portfolio):
    """Test Monte Carlo stochastic simulation with Geometric Brownian Motion."""
    cfg = MonteCarloConfig(
        n_simulations=500,
        horizon_days=10,
        time_steps=10,
        drift_model=DriftModel.GBM,
        random_seed=42,
    )
    engine = MonteCarloEngine(config=cfg)
    res = engine.simulate_portfolio(simple_portfolio)

    assert res.initial_portfolio_value == simple_portfolio.net_asset_value
    assert (
        res.min_terminal_value
        <= res.p5_terminal_value
        <= res.p50_terminal_value
        <= res.p95_terminal_value
        <= res.max_terminal_value
    )
    assert res.simulated_var_95_amount >= 0
    assert res.simulated_cvar_95_amount >= res.simulated_var_95_amount
    assert 0.0 <= res.max_drawdown_mean_pct <= 1.0


def test_monte_carlo_engine_jump_diffusion(simple_portfolio: Portfolio):
    """Test Monte Carlo simulation with Merton Jump Diffusion."""
    cfg = MonteCarloConfig(
        n_simulations=300,
        horizon_days=5,
        time_steps=5,
        drift_model=DriftModel.JUMP_DIFFUSION,
        jump_intensity=0.20,
        jump_mean=-0.08,
        random_seed=42,
    )
    engine = MonteCarloEngine(config=cfg)
    res = engine.simulate_portfolio(simple_portfolio)
    assert res.initial_portfolio_value == simple_portfolio.net_asset_value
    assert res.mean_terminal_value > 0


def test_monte_carlo_engine_mean_reverting(simple_portfolio: Portfolio):
    """Test Monte Carlo simulation with Mean Reverting drift."""
    cfg = MonteCarloConfig(
        n_simulations=300,
        horizon_days=5,
        time_steps=5,
        drift_model=DriftModel.MEAN_REVERTING,
        mean_reversion_speed=0.5,
        random_seed=42,
    )
    engine = MonteCarloEngine(config=cfg)
    res = engine.simulate_portfolio(simple_portfolio)
    assert res.initial_portfolio_value == simple_portfolio.net_asset_value


def test_black_scholes_option_greeks():
    """Test Black-Scholes analytical pricer and Greek risk sensitivities."""
    spot = 100.0
    strike = 100.0
    time_yr = 1.0
    vol = 0.20
    rate = 0.05

    call = calculate_option_greeks(
        OptionType.CALL,
        spot_price=spot,
        strike_price=strike,
        time_to_expiry_years=time_yr,
        volatility=vol,
        risk_free_rate=rate,
    )
    put = calculate_option_greeks(
        OptionType.PUT,
        spot_price=spot,
        strike_price=strike,
        time_to_expiry_years=time_yr,
        volatility=vol,
        risk_free_rate=rate,
    )

    # Put-Call Parity: C - P = S - K * exp(-r * T)
    parity_diff = (call.price - put.price) - (spot - strike * np.exp(-rate * time_yr))
    assert abs(parity_diff) < 0.01

    # Delta bounds
    assert 0.0 < call.delta < 1.0
    assert -1.0 < put.delta < 0.0
    assert abs((call.delta - put.delta) - 1.0) < 0.001

    # Gamma and Vega identical and positive
    assert call.gamma > 0
    assert abs(call.gamma - put.gamma) < 1e-5
    assert call.vega > 0
    assert abs(call.vega - put.vega) < 1e-4


def test_portfolio_risk_engine_compliance(sample_portfolio: Portfolio):
    """Test portfolio compliance limit checks and violation flagging."""
    engine = PortfolioRiskEngine()
    report = engine.check_compliance(sample_portfolio)
    assert report.portfolio_id == sample_portfolio.id
    assert isinstance(report.passed, bool)

    # Tight limits to trigger violations
    tight_limits = RiskLimits(
        max_leverage=0.5,  # sample portfolio leverage is higher
        max_single_position_pct=0.10,
        max_sector_pct=0.15,
    )
    strict_report = engine.check_compliance(sample_portfolio, custom_limits=tight_limits)
    assert strict_report.passed is False
    assert len(strict_report.violations) > 0


def test_portfolio_risk_engine_stress_testing(sample_portfolio: Portfolio):
    """Test macroeconomic crisis stress test impact."""
    scenarios = PortfolioRepository.get_stress_scenarios()
    lehman = scenarios["2008-lehman"]

    engine = PortfolioRiskEngine()
    result = engine.run_stress_test(sample_portfolio, lehman)

    assert result.portfolio_value_before == sample_portfolio.net_asset_value
    assert result.portfolio_value_after < result.portfolio_value_before
    assert result.absolute_impact < 0
    assert result.percentage_impact < 0
    assert len(result.asset_impacts) == len(sample_portfolio.positions)


def test_portfolio_repository_presets():
    """Verify built-in portfolios and stress scenarios in repository."""
    ports = PortfolioRepository.get_benchmark_portfolios()
    assert "global-macro" in ports
    assert "tech-momentum" in ports
    assert "multi-asset-balanced" in ports
    assert "crypto-fx-alpha" in ports

    for p in ports.values():
        assert p.net_asset_value > 0
        assert len(p.positions) > 0

    scenarios = PortfolioRepository.get_stress_scenarios()
    assert len(scenarios) >= 4


def test_cli_execution(cli_runner: CliRunner):
    """Test all CLI subcommands exit with code 0."""
    res = cli_runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "FinTech Algorithmic Trading Risk Engine" in res.output

    res = cli_runner.invoke(cli, ["list-portfolios"])
    assert res.exit_code == 0
    assert "Global" in res.output and "Macro" in res.output

    res = cli_runner.invoke(cli, ["analyze-var", "-p", "global-macro", "-c", "0.95", "-h", "1"])
    assert res.exit_code == 0
    assert "Value-at-Risk Analysis" in res.output

    res = cli_runner.invoke(cli, ["analyze-var", "-p", "global-macro", "--json-out"])
    assert res.exit_code == 0
    assert "HISTORICAL" in res.output

    res = cli_runner.invoke(
        cli, ["simulate", "-p", "tech-momentum", "-n", "150", "-h", "5", "--model", "GBM"]
    )
    assert res.exit_code == 0
    assert "Monte Carlo Risk Simulation" in res.output

    res = cli_runner.invoke(cli, ["stress-test", "-p", "multi-asset-balanced"])
    assert res.exit_code == 0
    assert "Stress Testing Scenarios" in res.output

    res = cli_runner.invoke(cli, ["check-limits", "-p", "global-macro"])
    assert res.exit_code == 0

    res = cli_runner.invoke(cli, ["greeks", "--type", "CALL", "-s", "100", "-k", "100"])
    assert res.exit_code == 0
    assert "Black-Scholes CALL Option Greeks" in res.output
