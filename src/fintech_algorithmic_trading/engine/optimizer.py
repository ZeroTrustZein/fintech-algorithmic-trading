"""Portfolio optimization engine: Mean-Variance (Markowitz) and Equal Risk Contribution (Risk Parity)."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.optimize as sco

from fintech_algorithmic_trading.types import (
    OptimizationObjective,
    OptimizationResult,
    Portfolio,
)


class PortfolioOptimizer:
    """Quantitative asset allocation and portfolio optimization engine."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _portfolio_performance(
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> tuple[float, float]:
        """Compute portfolio expected annual return and volatility."""
        ret = float(np.dot(weights, expected_returns))
        var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
        vol = float(np.sqrt(max(1e-10, var)))
        return ret, vol

    @staticmethod
    def calculate_risk_contributions(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: Sequence[str],
    ) -> Dict[str, float]:
        """Calculate percentage risk contribution of each asset to portfolio volatility."""
        var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
        vol = np.sqrt(max(1e-10, var))
        # Marginal risk contribution: (Sigma * w) / vol
        marginal_contrib = np.dot(cov_matrix, weights) / vol
        # Component risk contribution: w * MRC
        component_risk = weights * marginal_contrib
        # Normalized percentage risk contribution
        total_risk = np.sum(component_risk)
        if total_risk <= 0:
            pct_contrib = np.full(len(weights), 1.0 / len(weights))
        else:
            pct_contrib = component_risk / total_risk

        return {sym: round(float(pct_contrib[i]), 4) for i, sym in enumerate(symbols)}

    def optimize(
        self,
        symbols: List[str],
        expected_returns: Sequence[float] | np.ndarray,
        covariance_matrix: np.ndarray,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        max_weight_cap: float = 1.0,
        min_weight_floor: float = 0.0,
    ) -> OptimizationResult:
        """Run optimization for specified objective."""
        start_t = time.perf_counter()
        n = len(symbols)
        if n < 2:
            raise ValueError("Optimization requires at least 2 assets.")

        mu = np.asarray(expected_returns, dtype=np.float64)
        cov = np.asarray(covariance_matrix, dtype=np.float64)

        if mu.shape != (n,):
            raise ValueError(f"expected_returns shape {mu.shape} does not match symbols count {n}.")
        if cov.shape != (n, n):
            raise ValueError(f"covariance_matrix shape {cov.shape} must be ({n}, {n}).")

        # Initial equal weights
        init_w = np.full(n, 1.0 / n, dtype=np.float64)
        bounds = tuple((min_weight_floor, max_weight_cap) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if objective == OptimizationObjective.EQUAL_WEIGHT:
            w_opt = init_w

        elif objective == OptimizationObjective.MIN_VARIANCE:
            def min_var_fn(w: np.ndarray) -> float:
                return float(np.dot(w.T, np.dot(cov, w)))

            res = sco.minimize(
                min_var_fn,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            w_opt = res.x if res.success else init_w

        elif objective == OptimizationObjective.MAX_SHARPE:
            def neg_sharpe_fn(w: np.ndarray) -> float:
                r, v = self._portfolio_performance(w, mu, cov)
                excess = r - self.risk_free_rate
                return -float(excess / v)

            res = sco.minimize(
                neg_sharpe_fn,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            w_opt = res.x if res.success else init_w

        elif objective == OptimizationObjective.RISK_PARITY:
            # Objective: sum of squared differences between relative risk contributions and 1/n
            def risk_parity_fn(w: np.ndarray) -> float:
                var = float(np.dot(w.T, np.dot(cov, w)))
                target = var / float(n)
                # w_i * (Sigma * w)_i should equal target
                marginal = np.dot(cov, w)
                risk_contrib = w * marginal
                return float(np.sum((risk_contrib - target) ** 2) * 1e6)

            res = sco.minimize(
                risk_parity_fn,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 800},
            )
            w_opt = res.x if res.success else init_w

        else:
            w_opt = init_w

        # Normalize weights to strictly sum to 1.0
        w_opt = np.maximum(0.0, w_opt)
        sum_w = np.sum(w_opt)
        if sum_w > 0:
            w_opt = w_opt / sum_w

        ret, vol = self._portfolio_performance(w_opt, mu, cov)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        risk_contribs = self.calculate_risk_contributions(w_opt, cov, symbols)
        weights_dict = {sym: round(float(w_opt[i]), 4) for i, sym in enumerate(symbols)}
        dur_ms = (time.perf_counter() - start_t) * 1000.0

        return OptimizationResult(
            objective=objective,
            weights=weights_dict,
            expected_return_annual=round(ret, 4),
            volatility_annual=round(vol, 4),
            sharpe_ratio=round(sharpe, 4),
            risk_contributions=risk_contribs,
            computation_time_ms=round(dur_ms, 3),
        )

    def optimize_portfolio(
        self,
        portfolio: Portfolio,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        covariance_matrix: Optional[np.ndarray] = None,
        max_weight_cap: float = 0.50,
    ) -> OptimizationResult:
        """Optimize an existing portfolio using its positions and estimated return parameters."""
        positions = portfolio.positions
        if len(positions) < 2:
            raise ValueError("Portfolio must contain at least 2 positions to optimize.")

        symbols = [p.asset.symbol for p in positions]
        n = len(symbols)

        # Estimate expected returns based on beta * 0.06 + risk_free_rate
        mu = np.array(
            [self.risk_free_rate + p.asset.beta_to_market * 0.06 for p in positions],
            dtype=np.float64,
        )

        # Build or use covariance matrix
        if covariance_matrix is not None:
            cov = np.asarray(covariance_matrix, dtype=np.float64)
        else:
            vols = np.array([p.asset.volatility_annual for p in positions], dtype=np.float64)
            # Default cross-correlation 0.35
            corr = np.full((n, n), 0.35, dtype=np.float64)
            np.fill_diagonal(corr, 1.0)
            cov = np.outer(vols, vols) * corr

        return self.optimize(
            symbols=symbols,
            expected_returns=mu,
            covariance_matrix=cov,
            objective=objective,
            max_weight_cap=max_weight_cap,
        )
