"""Extreme Value Theory (EVT) Peaks-Over-Threshold (POT) tail risk analyzer."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import scipy.stats as stats

from fintech_algorithmic_trading.types import EVTResult, VaRMethod, VaRResult


class EVTEngine:
    """Extreme Value Theory tail-risk analyzer using Generalized Pareto Distribution (GPD)."""

    def __init__(self, threshold_quantile: float = 0.90):
        if not (0.50 < threshold_quantile < 0.99):
            raise ValueError("threshold_quantile must be between 0.50 and 0.99")
        self.threshold_quantile = threshold_quantile

    def fit_pot(
        self,
        losses: np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[float, float, float, np.ndarray]:
        """Fit Generalized Pareto Distribution to threshold exceedances.

        Returns:
            (u, xi, beta, exceedances)
        """
        if threshold is None:
            u = float(np.percentile(losses, self.threshold_quantile * 100.0))
        else:
            u = float(threshold)

        exceedances = losses[losses > u] - u
        if len(exceedances) < 5:
            raise ValueError(
                f"Insufficient exceedances ({len(exceedances)} < 5) above threshold {u:.5f}. "
                "Provide more return observations or lower threshold quantile."
            )

        # Fit GPD via maximum likelihood with loc=0 fixed
        c, _, beta = stats.genpareto.fit(exceedances, floc=0)
        xi = float(c)
        beta = float(max(1e-8, beta))

        return u, xi, beta, exceedances

    def calculate_evt_var(
        self,
        returns: Sequence[float] | np.ndarray,
        portfolio_value: float,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        threshold: Optional[float] = None,
    ) -> EVTResult:
        """Compute EVT Peaks-Over-Threshold VaR and CVaR."""
        start_t = time.perf_counter()
        r = np.asarray(returns, dtype=np.float64)
        if len(r) < 20:
            raise ValueError("At least 20 return observations required for EVT estimation.")

        losses = -r
        n_total = len(losses)
        u, xi, beta, exceedances = self.fit_pot(losses, threshold)
        n_u = len(exceedances)

        alpha = confidence_level
        tail_prob = 1.0 - alpha
        prob_ratio = (n_total / n_u) * tail_prob

        # Calculate 1-day VaR
        if abs(xi) > 1e-5:
            var_1d_pct = u + (beta / xi) * ((prob_ratio ** (-xi)) - 1.0)
        else:
            var_1d_pct = u - beta * np.log(prob_ratio)

        var_1d_pct = max(0.0, float(var_1d_pct))

        # Expected Shortfall (CVaR)
        if xi < 1.0:
            cvar_1d_pct = (var_1d_pct / (1.0 - xi)) + ((beta - xi * u) / (1.0 - xi))
            cvar_1d_pct = max(var_1d_pct, float(cvar_1d_pct))
        else:
            cvar_1d_pct = var_1d_pct * 1.5

        scale = np.sqrt(float(horizon_days))
        var_pct = var_1d_pct * scale
        var_amount = var_pct * portfolio_value
        cvar_pct = cvar_1d_pct * scale
        cvar_amount = cvar_pct * portfolio_value
        dur_ms = (time.perf_counter() - start_t) * 1000.0

        return EVTResult(
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            threshold_u=round(u, 6),
            exceedance_count=n_u,
            shape_parameter_xi=round(xi, 4),
            scale_parameter_beta=round(beta, 6),
            var_amount=round(var_amount, 2),
            var_pct=round(var_pct, 6),
            cvar_amount=round(cvar_amount, 2),
            cvar_pct=round(cvar_pct, 6),
            portfolio_value=round(portfolio_value, 2),
            computation_time_ms=round(dur_ms, 3),
        )


def calculate_evt_var(
    returns: Sequence[float] | np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    threshold_quantile: float = 0.90,
) -> VaRResult:
    """Compute Extreme Value Theory VaR formatted as standard VaRResult."""
    engine = EVTEngine(threshold_quantile=threshold_quantile)
    evt_res = engine.calculate_evt_var(
        returns=returns,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
    )
    r = np.asarray(returns, dtype=np.float64)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))

    return VaRResult(
        method=VaRMethod.EXTREME_VALUE_THEORY,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
        var_amount=evt_res.var_amount,
        var_pct=evt_res.var_pct,
        cvar_amount=evt_res.cvar_amount,
        cvar_pct=evt_res.cvar_pct,
        portfolio_value=evt_res.portfolio_value,
        skewness=round(skew, 4),
        excess_kurtosis=round(kurt, 4),
        computation_time_ms=evt_res.computation_time_ms,
    )
