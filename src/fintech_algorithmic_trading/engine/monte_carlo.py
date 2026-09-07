"""Vectorized multi-asset stochastic Monte Carlo simulation engine."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from fintech_algorithmic_trading.types import (
    DriftModel,
    MonteCarloConfig,
    MonteCarloResult,
    Portfolio,
)


class MonteCarloEngine:
    """Institutional multi-asset Monte Carlo risk simulation engine."""

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.config = config or MonteCarloConfig()

    def simulate_portfolio(
        self,
        portfolio: Portfolio,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> MonteCarloResult:
        """Run multi-asset correlated Monte Carlo simulation for a given portfolio."""
        start_t = time.perf_counter()
        if not portfolio.positions:
            raise ValueError("Cannot simulate an empty portfolio.")

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        n_sims = self.config.n_simulations
        steps = self.config.time_steps
        horizon_days = self.config.horizon_days
        dt = (horizon_days / 252.0) / float(steps)

        positions = portfolio.positions
        n_assets = len(positions)
        s0 = np.array([p.current_price for p in positions], dtype=np.float64)
        quantities = np.array([p.quantity for p in positions], dtype=np.float64)
        vols = np.array([p.asset.volatility_annual for p in positions], dtype=np.float64)
        rf = self.config.risk_free_rate

        # Asset drift: default to risk-free rate + beta * equity risk premium (5%)
        drifts = np.array(
            [rf + p.asset.beta_to_market * 0.05 for p in positions],
            dtype=np.float64,
        )

        # Correlation matrix handling
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix, dtype=np.float64)
            if corr.shape != (n_assets, n_assets):
                raise ValueError(
                    f"Correlation matrix shape {corr.shape} must match ({n_assets}, {n_assets})"
                )
        else:
            # Identity correlation with 0.35 baseline pairwise correlation if multi-asset
            corr = np.full((n_assets, n_assets), 0.35, dtype=np.float64)
            np.fill_diagonal(corr, 1.0)

        # Ensure positive semi-definite and compute Cholesky decomposition
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Jitter diagonal if needed for numerical stability
            jitter = np.eye(n_assets) * 1e-6
            chol = np.linalg.cholesky(corr + jitter)

        # Array of simulated prices: shape (n_sims, steps + 1, n_assets)
        prices = np.zeros((n_sims, steps + 1, n_assets), dtype=np.float64)
        prices[:, 0, :] = s0

        # Jump diffusion parameters
        lam = self.config.jump_intensity
        jump_mu = self.config.jump_mean
        jump_std = self.config.jump_std
        # Compensator k for Merton jump
        k = np.exp(jump_mu + 0.5 * jump_std**2) - 1.0 if lam > 0 else 0.0

        for t in range(1, steps + 1):
            # Uncorrelated standard normal shocks: shape (n_sims, n_assets)
            z_uncorr = np.random.standard_normal(size=(n_sims, n_assets))
            # Correlated shocks: Z_corr = Z @ L.T
            z_corr = np.dot(z_uncorr, chol.T)

            current_p = prices[:, t - 1, :]

            if self.config.drift_model == DriftModel.GBM:
                drift_term = (drifts - 0.5 * vols**2) * dt
                diff_term = vols * np.sqrt(dt) * z_corr
                prices[:, t, :] = current_p * np.exp(drift_term + diff_term)

            elif self.config.drift_model == DriftModel.JUMP_DIFFUSION:
                drift_term = (drifts - 0.5 * vols**2 - lam * k) * dt
                diff_term = vols * np.sqrt(dt) * z_corr
                # Poisson jump arrivals per asset
                n_jumps = np.random.poisson(lam * dt, size=(n_sims, n_assets))
                jump_magnitudes = (
                    np.random.normal(jump_mu, jump_std, size=(n_sims, n_assets)) * n_jumps
                )
                prices[:, t, :] = current_p * np.exp(drift_term + diff_term + jump_magnitudes)

            elif self.config.drift_model == DriftModel.MEAN_REVERTING:
                kappa = self.config.mean_reversion_speed
                # Revert towards initial spot price s0
                reversion = kappa * (s0 - current_p) / np.maximum(1e-8, current_p) * dt
                diff_term = vols * np.sqrt(dt) * z_corr
                prices[:, t, :] = current_p * np.exp(reversion - 0.5 * (vols**2) * dt + diff_term)

        # Compute portfolio path values across time: shape (n_sims, steps + 1)
        # Cash is static across the horizon
        portfolio_paths = np.sum(prices * quantities, axis=2) + portfolio.cash

        initial_val = float(portfolio_paths[0, 0])
        terminal_vals = portfolio_paths[:, -1]

        # Drawdown calculation across each simulation trajectory
        # Running peaks: shape (n_sims, steps + 1)
        running_max = np.maximum.accumulate(portfolio_paths, axis=1)
        drawdowns = (running_max - portfolio_paths) / np.maximum(1e-8, running_max)
        max_drawdown_per_sim = np.max(drawdowns, axis=1)

        # Losses at terminal horizon: L = V0 - VT
        losses = initial_val - terminal_vals

        # Quantiles and percentiles
        p1 = float(np.percentile(terminal_vals, 1))
        p5 = float(np.percentile(terminal_vals, 5))
        p50 = float(np.percentile(terminal_vals, 50))
        p95 = float(np.percentile(terminal_vals, 95))
        p99 = float(np.percentile(terminal_vals, 99))

        # 95% VaR & CVaR
        var_95_amt = max(0.0, float(np.percentile(losses, 95)))
        tail_95 = losses[losses >= var_95_amt]
        cvar_95_amt = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95_amt

        # 99% VaR & CVaR
        var_99_amt = max(0.0, float(np.percentile(losses, 99)))
        tail_99 = losses[losses >= var_99_amt]
        cvar_99_amt = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99_amt

        prob_loss = float(np.mean(terminal_vals < initial_val))
        dur_ms = (time.perf_counter() - start_t) * 1000.0

        return MonteCarloResult(
            config=self.config,
            initial_portfolio_value=round(initial_val, 2),
            mean_terminal_value=round(float(np.mean(terminal_vals)), 2),
            median_terminal_value=round(p50, 2),
            min_terminal_value=round(float(np.min(terminal_vals)), 2),
            max_terminal_value=round(float(np.max(terminal_vals)), 2),
            p1_terminal_value=round(p1, 2),
            p5_terminal_value=round(p5, 2),
            p50_terminal_value=round(p50, 2),
            p95_terminal_value=round(p95, 2),
            p99_terminal_value=round(p99, 2),
            simulated_var_95_amount=round(var_95_amt, 2),
            simulated_var_95_pct=round(var_95_amt / initial_val, 6),
            simulated_cvar_95_amount=round(cvar_95_amt, 2),
            simulated_cvar_95_pct=round(cvar_95_amt / initial_val, 6),
            simulated_var_99_amount=round(var_99_amt, 2),
            simulated_var_99_pct=round(var_99_amt / initial_val, 6),
            simulated_cvar_99_amount=round(cvar_99_amt, 2),
            simulated_cvar_99_pct=round(cvar_99_amt / initial_val, 6),
            max_drawdown_mean_pct=round(float(np.mean(max_drawdown_per_sim)), 4),
            max_drawdown_p95_pct=round(float(np.percentile(max_drawdown_per_sim, 95)), 4),
            probability_of_loss=round(prob_loss, 4),
            computation_time_ms=round(dur_ms, 3),
        )
