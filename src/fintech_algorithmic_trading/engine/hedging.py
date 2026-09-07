"""Portfolio Greeks aggregation and Delta-Gamma dynamic hedging engine."""

from __future__ import annotations

from typing import List

from fintech_algorithmic_trading.engine.greeks import calculate_option_greeks
from fintech_algorithmic_trading.types import (
    DeltaGammaHedge,
    OptionContract,
    PortfolioGreeks,
)


class OptionHedgingEngine:
    """Quantitative derivatives risk surveillance and Delta-Gamma hedging calculator."""

    @staticmethod
    def calculate_portfolio_greeks(
        contracts: List[OptionContract],
        underlying_shares: float = 0.0,
        risk_free_rate: float = 0.045,
    ) -> PortfolioGreeks:
        """Aggregate Greeks across multiple option legs and underlying share holdings."""
        total_delta = underlying_shares * 1.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        total_rho = 0.0
        total_val = 0.0

        for c in contracts:
            g = calculate_option_greeks(
                option_type=c.option_type,
                spot_price=c.spot_price,
                strike_price=c.strike_price,
                time_to_expiry_years=c.time_to_expiry_years,
                volatility=c.volatility,
                risk_free_rate=risk_free_rate,
            )
            factor = c.quantity * c.multiplier
            total_delta += g.delta * factor
            total_gamma += g.gamma * factor
            total_vega += g.vega * factor
            total_theta += g.theta * factor
            total_rho += g.rho * factor
            total_val += g.price * factor

        return PortfolioGreeks(
            net_delta=round(total_delta, 4),
            net_gamma=round(total_gamma, 6),
            net_vega=round(total_vega, 4),
            net_theta=round(total_theta, 4),
            net_rho=round(total_rho, 4),
            total_market_value=round(total_val, 2),
        )

    @staticmethod
    def calculate_delta_hedge(
        current_net_delta: float,
        target_delta: float = 0.0,
        underlying_symbol: str = "UNDERLYING",
    ) -> DeltaGammaHedge:
        """Determine underlying shares required to achieve delta-neutral or targeted delta posture."""
        shares_needed = target_delta - current_net_delta
        return DeltaGammaHedge(
            underlying_symbol=underlying_symbol,
            target_net_delta=round(target_delta, 4),
            target_net_gamma=0.0,
            underlying_shares_needed=round(shares_needed, 4),
            post_hedge_delta=round(target_delta, 4),
            post_hedge_gamma=0.0,
        )

    @staticmethod
    def calculate_delta_gamma_hedge(
        portfolio_greeks: PortfolioGreeks,
        hedge_option: OptionContract,
        target_delta: float = 0.0,
        target_gamma: float = 0.0,
        risk_free_rate: float = 0.045,
    ) -> DeltaGammaHedge:
        """Solve two-variable linear system for Delta and Gamma neutralization.

        Uses benchmark option to neutralize Gamma, then underlying shares to neutralize Delta.
        """
        # Calculate benchmark option unit Greeks
        opt_greeks = calculate_option_greeks(
            option_type=hedge_option.option_type,
            spot_price=hedge_option.spot_price,
            strike_price=hedge_option.strike_price,
            time_to_expiry_years=hedge_option.time_to_expiry_years,
            volatility=hedge_option.volatility,
            risk_free_rate=risk_free_rate,
        )

        gamma_per_contract = opt_greeks.gamma * hedge_option.multiplier
        if abs(gamma_per_contract) < 1e-9:
            raise ValueError("Hedge option has near-zero gamma; cannot be used for gamma hedging.")

        # Contracts of hedge option required to reach target gamma
        gamma_diff = target_gamma - portfolio_greeks.net_gamma
        n_hedge_contracts = gamma_diff / gamma_per_contract

        # Delta resulting from hedge option
        delta_from_hedge_option = n_hedge_contracts * opt_greeks.delta * hedge_option.multiplier
        interim_delta = portfolio_greeks.net_delta + delta_from_hedge_option

        # Underlying shares needed to reach target delta
        shares_needed = target_delta - interim_delta

        return DeltaGammaHedge(
            underlying_symbol=hedge_option.underlying_symbol,
            target_net_delta=round(target_delta, 4),
            target_net_gamma=round(target_gamma, 6),
            underlying_shares_needed=round(shares_needed, 4),
            hedge_option_symbol=hedge_option.symbol,
            hedge_option_contracts_needed=round(n_hedge_contracts, 4),
            post_hedge_delta=round(target_delta, 4),
            post_hedge_gamma=round(target_gamma, 6),
        )
