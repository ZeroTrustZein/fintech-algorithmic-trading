"""Value-at-Risk (VaR) and Expected Shortfall (CVaR) quantitative analytics."""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import scipy.stats as stats

from fintech_algorithmic_trading.engine.evt import calculate_evt_var
from fintech_algorithmic_trading.types import Portfolio, VaRMethod, VaRResult


def calculate_historical_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult:
    """Compute Historical Simulation VaR and CVaR (Expected Shortfall).

    Args:
        returns: Historical return series (e.g. daily returns).
        portfolio_value: Current total portfolio valuation.
        confidence_level: Significance level (e.g. 0.95 for 95%).
        horizon_days: Holding period in days.

    Returns:
        VaRResult with absolute and percentage risk metrics.
    """
    start_t = time.perf_counter()
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 5:
        raise ValueError("At least 5 return observations are required for Historical VaR.")

    # Losses are negative returns: L = -R
    losses = -r
    # Quantile at confidence level: e.g. 95th percentile of loss
    percentile = confidence_level * 100.0
    var_1d_pct = float(np.percentile(losses, percentile))
    if var_1d_pct < 0:
        var_1d_pct = 0.0

    # Scale to horizon: square root of time rule
    scale = np.sqrt(float(horizon_days))
    var_pct = var_1d_pct * scale
    var_amount = var_pct * portfolio_value

    # CVaR: average of losses in tail >= 1d VaR
    tail_losses = losses[losses >= var_1d_pct]
    if len(tail_losses) > 0:
        cvar_1d_pct = float(np.mean(tail_losses))
    else:
        cvar_1d_pct = var_1d_pct

    cvar_pct = cvar_1d_pct * scale
    cvar_amount = cvar_pct * portfolio_value

    skewness = float(stats.skew(r))
    excess_kurt = float(stats.kurtosis(r, fisher=True))
    dur_ms = (time.perf_counter() - start_t) * 1000.0

    return VaRResult(
        method=VaRMethod.HISTORICAL,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
        var_amount=round(var_amount, 2),
        var_pct=round(var_pct, 6),
        cvar_amount=round(cvar_amount, 2),
        cvar_pct=round(cvar_pct, 6),
        portfolio_value=round(portfolio_value, 2),
        skewness=round(skewness, 4),
        excess_kurtosis=round(excess_kurt, 4),
        computation_time_ms=round(dur_ms, 3),
    )


def calculate_parametric_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult:
    """Compute Parametric (Variance-Covariance / Gaussian) VaR and CVaR.

    Args:
        returns: Historical return series.
        portfolio_value: Current total portfolio valuation.
        confidence_level: Significance level.
        horizon_days: Holding period in days.

    Returns:
        VaRResult with parametric metrics.
    """
    start_t = time.perf_counter()
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 3:
        raise ValueError("At least 3 return observations are required for Parametric VaR.")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = float(stats.norm.ppf(confidence_level))

    # Daily VaR % (loss): z * sigma - mu
    var_1d_pct = max(0.0, z * sigma - mu)
    scale = np.sqrt(float(horizon_days))
    var_pct = var_1d_pct * scale
    var_amount = var_pct * portfolio_value

    # Analytical CVaR for standard normal:
    # E[-R | -R >= VaR] = -mu + sigma * [pdf(z) / (1 - alpha)]
    pdf_z = float(stats.norm.pdf(z))
    cvar_1d_pct = max(var_1d_pct, -mu + sigma * (pdf_z / (1.0 - confidence_level)))
    cvar_pct = cvar_1d_pct * scale
    cvar_amount = cvar_pct * portfolio_value

    skewness = float(stats.skew(r))
    excess_kurt = float(stats.kurtosis(r, fisher=True))
    dur_ms = (time.perf_counter() - start_t) * 1000.0

    return VaRResult(
        method=VaRMethod.PARAMETRIC,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
        var_amount=round(var_amount, 2),
        var_pct=round(var_pct, 6),
        cvar_amount=round(cvar_amount, 2),
        cvar_pct=round(cvar_pct, 6),
        portfolio_value=round(portfolio_value, 2),
        skewness=round(skewness, 4),
        excess_kurtosis=round(excess_kurt, 4),
        computation_time_ms=round(dur_ms, 3),
    )


def calculate_cornish_fisher_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult:
    """Compute Cornish-Fisher Expansion VaR adjusting for skewness and fat-tailed excess kurtosis.

    The Cornish-Fisher quantile expansion:
    z_cf = z + (z^2 - 1) * S / 6 + (z^3 - 3z) * K / 24 - (2z^3 - 5z) * S^2 / 36
    """
    start_t = time.perf_counter()
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 10:
        raise ValueError("At least 10 return observations are required for Cornish-Fisher VaR.")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    s = float(stats.skew(r))
    k = float(stats.kurtosis(r, fisher=True))  # excess kurtosis

    z = float(stats.norm.ppf(confidence_level))

    # Cornish-Fisher adjusted z-score
    term1 = (z**2 - 1.0) * s / 6.0
    term2 = (z**3 - 3.0 * z) * k / 24.0
    term3 = (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    z_cf = z + term1 + term2 - term3

    var_1d_pct = max(0.0, z_cf * sigma - mu)
    scale = np.sqrt(float(horizon_days))
    var_pct = var_1d_pct * scale
    var_amount = var_pct * portfolio_value

    # Tail adjusted CVaR estimation
    pdf_z = float(stats.norm.pdf(z))
    base_cvar_ratio = (pdf_z / (1.0 - confidence_level)) / z if z > 0 else 1.2
    cvar_1d_pct = max(var_1d_pct, var_1d_pct * base_cvar_ratio)
    cvar_pct = cvar_1d_pct * scale
    cvar_amount = cvar_pct * portfolio_value

    dur_ms = (time.perf_counter() - start_t) * 1000.0

    return VaRResult(
        method=VaRMethod.CORNISH_FISHER,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
        var_amount=round(var_amount, 2),
        var_pct=round(var_pct, 6),
        cvar_amount=round(cvar_amount, 2),
        cvar_pct=round(cvar_pct, 6),
        portfolio_value=round(portfolio_value, 2),
        skewness=round(s, 4),
        excess_kurtosis=round(k, 4),
        computation_time_ms=round(dur_ms, 3),
    )


class VaRCalculator:
    """Unified engine to compute VaR across all methodologies."""

    @staticmethod
    def evaluate_all(
        returns: Sequence[float] | np.ndarray,
        portfolio_value: float,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        include_evt: bool = True,
    ) -> Dict[VaRMethod, VaRResult]:
        """Compute Historical, Parametric, Cornish-Fisher, and EVT VaR metrics simultaneously."""
        results: Dict[VaRMethod, VaRResult] = {
            VaRMethod.HISTORICAL: calculate_historical_var(
                returns, portfolio_value, confidence_level, horizon_days
            ),
            VaRMethod.PARAMETRIC: calculate_parametric_var(
                returns, portfolio_value, confidence_level, horizon_days
            ),
            VaRMethod.CORNISH_FISHER: calculate_cornish_fisher_var(
                returns, portfolio_value, confidence_level, horizon_days
            ),
        }
        if include_evt and len(returns) >= 20:
            try:
                results[VaRMethod.EXTREME_VALUE_THEORY] = calculate_evt_var(
                    returns, portfolio_value, confidence_level, horizon_days
                )
            except Exception:
                pass
        return results

    @staticmethod
    def calculate_portfolio_parametric_var(
        portfolio: Portfolio,
        asset_covariance_matrix: np.ndarray,
        symbols_order: list[str],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Calculate Parametric VaR for a multi-asset portfolio using weight vector and covariance matrix."""
        start_t = time.perf_counter()
        nav = portfolio.net_asset_value
        if nav <= 0:
            raise ValueError("Portfolio net asset value must be positive.")

        weights_map = portfolio.asset_weights
        w = np.array([weights_map.get(sym, 0.0) for sym in symbols_order], dtype=np.float64)

        cov = np.asarray(asset_covariance_matrix, dtype=np.float64)
        port_variance = float(np.dot(w.T, np.dot(cov, w)))
        port_vol_1d = float(np.sqrt(max(0.0, port_variance)))

        z = float(stats.norm.ppf(confidence_level))
        var_1d_pct = z * port_vol_1d
        scale = np.sqrt(float(horizon_days))
        var_pct = var_1d_pct * scale
        var_amount = var_pct * nav

        pdf_z = float(stats.norm.pdf(z))
        cvar_1d_pct = port_vol_1d * (pdf_z / (1.0 - confidence_level))
        cvar_pct = cvar_1d_pct * scale
        cvar_amount = cvar_pct * nav
        dur_ms = (time.perf_counter() - start_t) * 1000.0

        return VaRResult(
            method=VaRMethod.PARAMETRIC,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            var_amount=round(var_amount, 2),
            var_pct=round(var_pct, 6),
            cvar_amount=round(cvar_amount, 2),
            cvar_pct=round(cvar_pct, 6),
            portfolio_value=round(nav, 2),
            skewness=0.0,
            excess_kurtosis=0.0,
            computation_time_ms=round(dur_ms, 3),
        )
