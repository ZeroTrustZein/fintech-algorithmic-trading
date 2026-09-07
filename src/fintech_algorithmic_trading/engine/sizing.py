"""Dynamic position sizing, Kelly criterion optimization, and volatility targeting."""

from __future__ import annotations

import numpy as np

from fintech_algorithmic_trading.types import KellyCriterionMode, KellySizingResult


class PositionSizer:
    """Quantitative position sizing engine for algorithmic strategies."""

    @staticmethod
    def calculate_continuous_kelly(
        expected_return_annual: float,
        volatility_annual: float,
        risk_free_rate: float = 0.045,
        portfolio_nav: float = 100000.0,
        fractional_factor: float = 0.5,
        max_leverage_cap: float = 2.0,
    ) -> KellySizingResult:
        """Calculate continuous-time Merton/Kelly optimal leverage fraction.

        Formula: f* = (mu - rf) / sigma^2
        Growth rate: g(f) = rf + f * (mu - rf) - 0.5 * f^2 * sigma^2
        """
        if volatility_annual <= 0:
            raise ValueError("Volatility must be positive.")
        if portfolio_nav <= 0:
            raise ValueError("Portfolio NAV must be positive.")

        excess_return = expected_return_annual - risk_free_rate
        var = volatility_annual**2

        if excess_return <= 0:
            full_f = 0.0
        else:
            full_f = excess_return / var

        half_f = full_f * 0.5
        quarter_f = full_f * 0.25

        rec_f = min(max_leverage_cap, max(0.0, full_f * fractional_factor))
        rec_size = rec_f * portfolio_nav

        # Expected geometric growth rate at recommended fraction
        growth_rate = risk_free_rate + rec_f * excess_return - 0.5 * (rec_f**2) * var

        # Drawdown risk approximation: Prob(drawdown >= D) = D^(2 * f - 1)
        # Approximate 50% drawdown risk under chosen fraction
        max_dd_risk = 0.5 ** (1.0 / max(0.01, rec_f)) if rec_f > 0 else 0.0

        return KellySizingResult(
            mode=KellyCriterionMode.CONTINUOUS,
            full_kelly_fraction=round(full_f, 4),
            half_kelly_fraction=round(half_f, 4),
            quarter_kelly_fraction=round(quarter_f, 4),
            recommended_fraction=round(rec_f, 4),
            recommended_position_size=round(rec_size, 2),
            expected_growth_rate=round(growth_rate, 4),
            max_drawdown_risk_pct=round(min(1.0, max_dd_risk), 4),
        )

    @staticmethod
    def calculate_discrete_kelly(
        win_rate: float,
        win_loss_ratio: float,
        portfolio_nav: float = 100000.0,
        fractional_factor: float = 0.5,
        max_fraction_cap: float = 0.30,
    ) -> KellySizingResult:
        """Calculate discrete win/loss Kelly betting fraction.

        Formula: f* = p - (1 - p) / b = (p * b - q) / b
        where p = win_rate, q = 1 - p, b = payoff ratio (avg_win / avg_loss).
        """
        if not (0.0 < win_rate < 1.0):
            raise ValueError("Win rate must be between 0 and 1 exclusive.")
        if win_loss_ratio <= 0:
            raise ValueError("Win/loss ratio must be positive.")
        if portfolio_nav <= 0:
            raise ValueError("Portfolio NAV must be positive.")

        p = win_rate
        q = 1.0 - p
        b = win_loss_ratio

        edge = p * b - q
        if edge <= 0:
            full_f = 0.0
        else:
            full_f = edge / b

        half_f = full_f * 0.5
        quarter_f = full_f * 0.25

        rec_f = min(max_fraction_cap, max(0.0, full_f * fractional_factor))
        rec_size = rec_f * portfolio_nav

        # Discrete growth rate: E[log(1 + f * R)]
        # R_win = b, R_loss = -1
        if rec_f > 0 and (1.0 - rec_f) > 0:
            growth_rate = p * np.log(1.0 + rec_f * b) + q * np.log(1.0 - rec_f)
        else:
            growth_rate = 0.0

        max_dd_risk = 0.5 ** (1.0 / max(0.01, rec_f)) if rec_f > 0 else 0.0

        return KellySizingResult(
            mode=KellyCriterionMode.DISCRETE,
            full_kelly_fraction=round(full_f, 4),
            half_kelly_fraction=round(half_f, 4),
            quarter_kelly_fraction=round(quarter_f, 4),
            recommended_fraction=round(rec_f, 4),
            recommended_position_size=round(rec_size, 2),
            expected_growth_rate=round(float(growth_rate), 4),
            max_drawdown_risk_pct=round(min(1.0, max_dd_risk), 4),
        )

    @staticmethod
    def calculate_volatility_target_allocation(
        target_vol_annual: float,
        asset_vols_annual: dict[str, float],
        portfolio_nav: float,
        max_weight_cap: float = 0.35,
    ) -> dict[str, float]:
        """Compute inversely volatility-weighted position allocations."""
        if target_vol_annual <= 0 or portfolio_nav <= 0 or not asset_vols_annual:
            return {}

        inv_vols = {
            sym: (1.0 / max(1e-4, vol)) for sym, vol in asset_vols_annual.items() if vol > 0
        }
        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            return {}

        weights = {}
        for sym, inv in inv_vols.items():
            base_w = inv / total_inv
            scaled_w = min(max_weight_cap, base_w * (target_vol_annual / 0.15))
            weights[sym] = round(scaled_w, 4)

        return weights

    @staticmethod
    def calculate_drawdown_scaled_size(
        base_size: float,
        current_drawdown_pct: float,
        max_tolerated_drawdown_pct: float = 0.15,
    ) -> float:
        """Dynamically reduce position size when in drawdown to prevent breach of risk mandate."""
        if current_drawdown_pct <= 0:
            return base_size
        if max_tolerated_drawdown_pct <= 0:
            return 0.0

        dd_ratio = current_drawdown_pct / max_tolerated_drawdown_pct
        if dd_ratio >= 1.0:
            return 0.0

        scaling_multiplier = max(0.0, 1.0 - dd_ratio)
        return round(base_size * scaling_multiplier, 2)
