"""Comprehensive unit and integration test suite for Stage 3 Core Logic engines."""

import numpy as np
import pytest

from fintech_algorithmic_trading import (
    EVTEngine,
    KellyCriterionMode,
    OptimizationObjective,
    OptionContract,
    OptionHedgingEngine,
    OptionType,
    Portfolio,
    PortfolioOptimizer,
    PortfolioRiskEngine,
    PositionSizer,
    VaRMethod,
    calculate_evt_var,
)

# ============================================================================
# 1. Extreme Value Theory (EVT) Engine Tests
# ============================================================================


def test_evt_engine_fit_pot(sample_daily_returns: np.ndarray):
    """Test Generalized Pareto Distribution fitting on tail losses."""
    engine = EVTEngine(threshold_quantile=0.90)
    losses = -sample_daily_returns
    u, xi, beta, exceedances = engine.fit_pot(losses)

    assert u > 0.0
    assert beta > 0.0
    assert len(exceedances) >= 5
    assert np.all(exceedances > 0.0)


def test_evt_engine_invalid_threshold():
    """Verify invalid threshold quantile raises ValueError."""
    with pytest.raises(ValueError, match="threshold_quantile must be between"):
        EVTEngine(threshold_quantile=0.30)

    with pytest.raises(ValueError, match="threshold_quantile must be between"):
        EVTEngine(threshold_quantile=0.999)


def test_evt_engine_insufficient_data():
    """Verify error on insufficient data length."""
    engine = EVTEngine()
    short_returns = np.array([0.01, -0.01, 0.02, -0.02])
    with pytest.raises(ValueError, match="At least 20 return observations required"):
        engine.calculate_evt_var(short_returns, portfolio_value=100000.0)


def test_evt_var_calculation(sample_daily_returns: np.ndarray):
    """Test EVT Peaks-Over-Threshold VaR and CVaR calculations."""
    engine = EVTEngine(threshold_quantile=0.90)
    port_val = 1_000_000.0
    res_95 = engine.calculate_evt_var(
        sample_daily_returns, portfolio_value=port_val, confidence_level=0.95
    )
    res_99 = engine.calculate_evt_var(
        sample_daily_returns, portfolio_value=port_val, confidence_level=0.99
    )

    assert res_95.confidence_level == 0.95
    assert res_95.var_amount > 0.0
    assert res_95.cvar_amount >= res_95.var_amount
    assert res_99.var_amount >= res_95.var_amount
    assert res_95.exceedance_count > 0
    assert res_95.portfolio_value == port_val


def test_calculate_evt_var_wrapper(sample_daily_returns: np.ndarray):
    """Test standard VaRResult output wrapper for EVT."""
    port_val = 500_000.0
    var_res = calculate_evt_var(
        sample_daily_returns, portfolio_value=port_val, confidence_level=0.95, horizon_days=5
    )
    assert var_res.method == VaRMethod.EXTREME_VALUE_THEORY
    assert var_res.var_amount > 0.0
    assert var_res.cvar_amount >= var_res.var_amount
    assert var_res.horizon_days == 5


# ============================================================================
# 2. Dynamic Position Sizing & Kelly Criterion Tests
# ============================================================================


def test_continuous_kelly_criterion():
    """Test continuous Merton/Kelly leverage fraction calculations."""
    # Expected annual return 15%, vol 20%, risk-free 5% -> excess = 10%, var = 0.04
    # Full Kelly: 0.10 / 0.04 = 2.50
    res = PositionSizer.calculate_continuous_kelly(
        expected_return_annual=0.15,
        volatility_annual=0.20,
        risk_free_rate=0.05,
        portfolio_nav=200_000.0,
        fractional_factor=0.5,
        max_leverage_cap=2.0,
    )
    assert res.mode == KellyCriterionMode.CONTINUOUS
    assert res.full_kelly_fraction == 2.50
    assert res.half_kelly_fraction == 1.25
    assert res.quarter_kelly_fraction == 0.625
    assert res.recommended_fraction == 1.25
    assert res.recommended_position_size == 250_000.0
    assert res.expected_growth_rate > 0.0


def test_continuous_kelly_negative_edge():
    """Test that zero allocation is recommended when expected return <= risk free rate."""
    res = PositionSizer.calculate_continuous_kelly(
        expected_return_annual=0.03,
        volatility_annual=0.20,
        risk_free_rate=0.05,
        portfolio_nav=100_000.0,
    )
    assert res.full_kelly_fraction == 0.0
    assert res.recommended_fraction == 0.0
    assert res.recommended_position_size == 0.0


def test_continuous_kelly_input_validation():
    """Test input validations for continuous Kelly."""
    with pytest.raises(ValueError, match="Volatility must be positive"):
        PositionSizer.calculate_continuous_kelly(0.10, 0.0, 0.05, 100000.0)

    with pytest.raises(ValueError, match="Portfolio NAV must be positive"):
        PositionSizer.calculate_continuous_kelly(0.10, 0.20, 0.05, 0.0)


def test_discrete_kelly_criterion():
    """Test discrete win/loss Kelly betting formula."""
    # Win rate 55%, payoff 1.5 -> edge = 0.55 * 1.5 - 0.45 = 0.825 - 0.45 = 0.375
    # Full Kelly: 0.375 / 1.5 = 0.25 (25%)
    res = PositionSizer.calculate_discrete_kelly(
        win_rate=0.55,
        win_loss_ratio=1.5,
        portfolio_nav=100_000.0,
        fractional_factor=0.5,
        max_fraction_cap=0.30,
    )
    assert res.mode == KellyCriterionMode.DISCRETE
    assert res.full_kelly_fraction == 0.25
    assert res.half_kelly_fraction == 0.125
    assert res.recommended_fraction == 0.125
    assert res.recommended_position_size == 12_500.0
    assert res.expected_growth_rate > 0.0


def test_discrete_kelly_no_edge():
    """Test zero Kelly fraction when expected value is negative."""
    # Win rate 40%, payoff 1.0 -> edge = 0.4 - 0.6 = -0.2 < 0
    res = PositionSizer.calculate_discrete_kelly(
        win_rate=0.40,
        win_loss_ratio=1.0,
        portfolio_nav=100_000.0,
    )
    assert res.full_kelly_fraction == 0.0
    assert res.recommended_fraction == 0.0
    assert res.recommended_position_size == 0.0


def test_discrete_kelly_input_validation():
    """Test input validation for discrete Kelly."""
    with pytest.raises(ValueError, match="Win rate must be between 0 and 1 exclusive"):
        PositionSizer.calculate_discrete_kelly(1.5, 1.0, 100000.0)

    with pytest.raises(ValueError, match="Win/loss ratio must be positive"):
        PositionSizer.calculate_discrete_kelly(0.5, 0.0, 100000.0)


def test_volatility_target_allocation():
    """Test inversely volatility-weighted position sizing."""
    vols = {"ASSET_A": 0.10, "ASSET_B": 0.20, "ASSET_C": 0.40}
    weights = PositionSizer.calculate_volatility_target_allocation(
        target_vol_annual=0.15,
        asset_vols_annual=vols,
        portfolio_nav=1_000_000.0,
        max_weight_cap=0.40,
    )
    assert "ASSET_A" in weights
    assert "ASSET_B" in weights
    assert "ASSET_C" in weights
    # Lower vol asset receives highest weight
    assert weights["ASSET_A"] > weights["ASSET_B"] > weights["ASSET_C"]


def test_drawdown_scaled_size():
    """Test dynamic position downsizing during portfolio drawdowns."""
    base_size = 50_000.0
    # No drawdown: full size
    assert PositionSizer.calculate_drawdown_scaled_size(base_size, 0.0) == 50_000.0

    # 7.5% drawdown against 15% max limit: 50% scale
    half_size = PositionSizer.calculate_drawdown_scaled_size(base_size, 0.075, 0.15)
    assert half_size == 25_000.0

    # 15% or higher drawdown: zero size
    zero_size = PositionSizer.calculate_drawdown_scaled_size(base_size, 0.15, 0.15)
    assert zero_size == 0.0


def test_portfolio_risk_engine_kelly_integration():
    """Test that PortfolioRiskEngine delegates to PositionSizer."""
    res = PortfolioRiskEngine.calculate_kelly_bounds(
        expected_return_annual=0.12,
        volatility_annual=0.18,
        risk_free_rate=0.045,
        portfolio_nav=500_000.0,
    )
    assert res.mode == KellyCriterionMode.CONTINUOUS
    assert res.full_kelly_fraction > 0.0
    assert res.recommended_position_size > 0.0


# ============================================================================
# 3. Portfolio Optimizer (Markowitz & Risk Parity) Tests
# ============================================================================


def test_portfolio_optimizer_max_sharpe():
    """Test Markowitz maximum Sharpe ratio portfolio optimization."""
    symbols = ["EQUITY", "BOND", "COMMODITY"]
    mu = np.array([0.12, 0.05, 0.08])
    cov = np.array(
        [
            [0.040, 0.002, 0.005],
            [0.002, 0.006, 0.001],
            [0.005, 0.001, 0.025],
        ]
    )
    opt = PortfolioOptimizer(risk_free_rate=0.04)
    res = opt.optimize(symbols, mu, cov, objective=OptimizationObjective.MAX_SHARPE)

    assert res.objective == OptimizationObjective.MAX_SHARPE
    assert len(res.weights) == 3
    assert abs(sum(res.weights.values()) - 1.0) < 1e-3
    assert all(w >= 0 for w in res.weights.values())
    assert res.sharpe_ratio > 0.0
    assert res.volatility_annual > 0.0


def test_portfolio_optimizer_min_variance():
    """Test global minimum variance portfolio optimization."""
    symbols = ["HIGH_VOL", "LOW_VOL"]
    mu = np.array([0.15, 0.06])
    cov = np.array([[0.09, 0.00], [0.00, 0.01]])  # 30% vol vs 10% vol, 0 correlation
    opt = PortfolioOptimizer()
    res = opt.optimize(symbols, mu, cov, objective=OptimizationObjective.MIN_VARIANCE)

    assert res.objective == OptimizationObjective.MIN_VARIANCE
    assert abs(sum(res.weights.values()) - 1.0) < 1e-3
    # Low vol asset should receive substantially higher weight
    assert res.weights["LOW_VOL"] > res.weights["HIGH_VOL"]


def test_portfolio_optimizer_risk_parity():
    """Test Equal Risk Contribution (Risk Parity) optimization."""
    symbols = ["ASSET_1", "ASSET_2", "ASSET_3"]
    mu = np.array([0.08, 0.08, 0.08])
    # Asset 1 vol 10%, Asset 2 vol 20%, Asset 3 vol 30%
    cov = np.diag([0.01, 0.04, 0.09])
    opt = PortfolioOptimizer()
    res = opt.optimize(symbols, mu, cov, objective=OptimizationObjective.RISK_PARITY)

    assert res.objective == OptimizationObjective.RISK_PARITY
    assert abs(sum(res.weights.values()) - 1.0) < 1e-3
    # Lower vol asset needs higher capital weight to produce equal risk contribution
    assert res.weights["ASSET_1"] > res.weights["ASSET_2"] > res.weights["ASSET_3"]

    # Each asset risk contribution should be approximately 33.3%
    for rc in res.risk_contributions.values():
        assert abs(rc - 0.3333) < 0.05


def test_portfolio_optimizer_equal_weight():
    """Test 1/N equal weight allocation."""
    symbols = ["A", "B", "C", "D"]
    mu = np.array([0.1, 0.1, 0.1, 0.1])
    cov = np.eye(4) * 0.04
    opt = PortfolioOptimizer()
    res = opt.optimize(symbols, mu, cov, objective=OptimizationObjective.EQUAL_WEIGHT)

    for w in res.weights.values():
        assert w == 0.25


def test_portfolio_optimizer_from_portfolio_object(simple_portfolio: Portfolio):
    """Test optimization directly from Portfolio model instance."""
    opt = PortfolioOptimizer()
    res = opt.optimize_portfolio(simple_portfolio, objective=OptimizationObjective.MAX_SHARPE)
    assert len(res.weights) == len(simple_portfolio.positions)
    assert abs(sum(res.weights.values()) - 1.0) < 1e-3


def test_portfolio_optimizer_error_handling():
    """Test optimizer exceptions on invalid inputs."""
    opt = PortfolioOptimizer()
    # Less than 2 symbols
    with pytest.raises(ValueError, match="Optimization requires at least 2 assets"):
        opt.optimize(["A"], [0.1], np.array([[0.04]]))

    # Dimension mismatch
    with pytest.raises(ValueError, match="shape"):
        opt.optimize(["A", "B"], [0.1], np.eye(2))


# ============================================================================
# 4. Portfolio Greeks & Hedging Engine Tests
# ============================================================================


def test_portfolio_greeks_aggregation():
    """Test Greeks aggregation across multi-leg option positions."""
    c1 = OptionContract(
        symbol="SPY24C500",
        underlying_symbol="SPY",
        option_type=OptionType.CALL,
        spot_price=500.0,
        strike_price=500.0,
        time_to_expiry_years=0.25,
        volatility=0.20,
        quantity=10.0,  # Long 10 call contracts (1000 shares exposure)
        multiplier=100.0,
    )
    c2 = OptionContract(
        symbol="SPY24P500",
        underlying_symbol="SPY",
        option_type=OptionType.PUT,
        spot_price=500.0,
        strike_price=500.0,
        time_to_expiry_years=0.25,
        volatility=0.20,
        quantity=5.0,  # Long 5 put contracts
        multiplier=100.0,
    )

    greeks = OptionHedgingEngine.calculate_portfolio_greeks([c1, c2], underlying_shares=0.0)
    assert greeks.net_delta > 0.0  # Call delta exceeds put delta
    assert greeks.net_gamma > 0.0  # Both calls and puts have positive long gamma
    assert greeks.net_vega > 0.0  # Both long options have positive vega
    assert greeks.net_theta < 0.0  # Long options suffer negative theta decay
    assert greeks.total_market_value > 0.0


def test_portfolio_greeks_with_underlying_shares():
    """Test that underlying shares alter delta without affecting gamma or vega."""
    c = OptionContract(
        symbol="SPY24C500",
        underlying_symbol="SPY",
        option_type=OptionType.CALL,
        spot_price=500.0,
        strike_price=500.0,
        time_to_expiry_years=0.25,
        volatility=0.20,
        quantity=5.0,
        multiplier=100.0,
    )
    greeks_no_shares = OptionHedgingEngine.calculate_portfolio_greeks([c], underlying_shares=0.0)
    greeks_with_shares = OptionHedgingEngine.calculate_portfolio_greeks(
        [c], underlying_shares=200.0
    )

    # Delta shifts by exactly 200 shares
    assert round(greeks_with_shares.net_delta - greeks_no_shares.net_delta, 2) == 200.0
    # Gamma and Vega are identical
    assert greeks_with_shares.net_gamma == greeks_no_shares.net_gamma
    assert greeks_with_shares.net_vega == greeks_no_shares.net_vega


def test_delta_hedge_calculation():
    """Test calculation of shares needed for delta neutrality."""
    # Current net delta is +350.0
    hedge = OptionHedgingEngine.calculate_delta_hedge(
        current_net_delta=350.0, target_delta=0.0, underlying_symbol="SPY"
    )
    assert hedge.underlying_shares_needed == -350.0
    assert hedge.target_net_delta == 0.0


def test_delta_gamma_hedge_neutralization():
    """Test 2-instrument Delta-Gamma hedge neutralization."""
    # Portfolio of 10 long calls: positive delta and positive gamma
    contract = OptionContract(
        symbol="SPY24C500",
        underlying_symbol="SPY",
        option_type=OptionType.CALL,
        spot_price=500.0,
        strike_price=500.0,
        time_to_expiry_years=0.25,
        volatility=0.20,
        quantity=10.0,
        multiplier=100.0,
    )
    port_greeks = OptionHedgingEngine.calculate_portfolio_greeks([contract])

    # Hedge option: out-of-the-money put for hedging
    hedge_option = OptionContract(
        symbol="SPY24P490",
        underlying_symbol="SPY",
        option_type=OptionType.PUT,
        spot_price=500.0,
        strike_price=490.0,
        time_to_expiry_years=0.25,
        volatility=0.22,
        quantity=1.0,
        multiplier=100.0,
    )

    hedge_rec = OptionHedgingEngine.calculate_delta_gamma_hedge(
        portfolio_greeks=port_greeks,
        hedge_option=hedge_option,
        target_delta=0.0,
        target_gamma=0.0,
    )

    assert hedge_rec.hedge_option_symbol == "SPY24P490"
    assert hedge_rec.hedge_option_contracts_needed != 0.0
    assert hedge_rec.underlying_shares_needed != 0.0
    assert hedge_rec.post_hedge_delta == 0.0
    assert hedge_rec.post_hedge_gamma == 0.0
