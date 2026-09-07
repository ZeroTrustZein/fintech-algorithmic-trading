"""Portfolio risk management, limit compliance auditing, and crisis stress testing."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from fintech_algorithmic_trading.types import (
    AssetClass,
    Portfolio,
    RiskComplianceReport,
    RiskLimits,
    RiskViolation,
    StressScenario,
    StressTestResult,
)


class PortfolioRiskEngine:
    """Institutional trading risk surveillance and compliance engine."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

    def check_compliance(
        self,
        portfolio: Portfolio,
        custom_limits: Optional[RiskLimits] = None,
    ) -> RiskComplianceReport:
        """Audit a portfolio against institutional risk limits."""
        limits = custom_limits or self.limits
        violations: List[RiskViolation] = []
        nav = portfolio.net_asset_value
        gross_exp = portfolio.gross_exposure

        if nav <= 0:
            violations.append(
                RiskViolation(
                    rule="PORTFOLIO_SOLVENCY",
                    limit_value=0.0,
                    current_value=nav,
                    severity="BREACH",
                    message="Portfolio NAV is zero or negative (insolvency breach).",
                )
            )

        # 1. Leverage Check
        leverage = portfolio.leverage
        if leverage > limits.max_leverage:
            violations.append(
                RiskViolation(
                    rule="MAX_LEVERAGE",
                    limit_value=limits.max_leverage,
                    current_value=round(leverage, 3),
                    severity="BREACH" if leverage > limits.max_leverage * 1.15 else "CRITICAL",
                    message=f"Leverage {leverage:.2f}x exceeds maximum limit of {limits.max_leverage:.2f}x.",
                )
            )

        # 2. Single Position Concentration
        weights = portfolio.asset_weights
        max_sym, max_w = ("", 0.0)
        if weights:
            max_sym, max_w = max(weights.items(), key=lambda kv: kv[1])
            if max_w > limits.max_single_position_pct:
                violations.append(
                    RiskViolation(
                        rule="SINGLE_NAME_CONCENTRATION",
                        limit_value=limits.max_single_position_pct,
                        current_value=round(max_w, 4),
                        severity="CRITICAL",
                        message=f"Position {max_sym} weight {max_w * 100:.1f}% exceeds limit of {limits.max_single_position_pct * 100:.1f}%.",
                    )
                )

        # 3. Sector Concentration
        sector_weights = portfolio.sector_weights
        max_sec, max_sec_w = ("", 0.0)
        if sector_weights:
            max_sec, max_sec_w = max(sector_weights.items(), key=lambda kv: kv[1])
            if max_sec_w > limits.max_sector_pct:
                violations.append(
                    RiskViolation(
                        rule="SECTOR_CONCENTRATION",
                        limit_value=limits.max_sector_pct,
                        current_value=round(max_sec_w, 4),
                        severity="WARNING"
                        if max_sec_w < limits.max_sector_pct * 1.10
                        else "CRITICAL",
                        message=f"Sector '{max_sec}' weight {max_sec_w * 100:.1f}% exceeds limit of {limits.max_sector_pct * 100:.1f}%.",
                    )
                )

        # 4. Cash Buffer Check
        cash_buffer_pct = portfolio.cash / nav if nav > 0 else 0.0
        if cash_buffer_pct < limits.min_cash_buffer_pct:
            violations.append(
                RiskViolation(
                    rule="MIN_CASH_BUFFER",
                    limit_value=limits.min_cash_buffer_pct,
                    current_value=round(cash_buffer_pct, 4),
                    severity="WARNING",
                    message=f"Cash buffer {cash_buffer_pct * 100:.1f}% is below minimum required {limits.min_cash_buffer_pct * 100:.1f}%.",
                )
            )

        # 5. Estimated Portfolio VaR Check (Parametric weighted sum approximation)
        weighted_vol_annual = 0.0
        if gross_exp > 0:
            for p in portfolio.positions:
                w = p.market_value / gross_exp
                weighted_vol_annual += w * p.asset.volatility_annual
        # Daily vol = Annual vol / sqrt(252), 95% z = 1.64485
        est_daily_var_pct = (
            (weighted_vol_annual / np.sqrt(252.0)) * 1.64485 * (gross_exp / nav if nav > 0 else 1.0)
        )
        if est_daily_var_pct > limits.max_var_95_pct:
            violations.append(
                RiskViolation(
                    rule="MAX_VAR_95_PCT",
                    limit_value=limits.max_var_95_pct,
                    current_value=round(est_daily_var_pct, 4),
                    severity="CRITICAL",
                    message=f"Estimated 1-day 95% VaR {est_daily_var_pct * 100:.2f}% exceeds limit of {limits.max_var_95_pct * 100:.2f}%.",
                )
            )

        passed = len(violations) == 0

        return RiskComplianceReport(
            portfolio_id=portfolio.id,
            portfolio_name=portfolio.name,
            passed=passed,
            violations=violations,
            current_leverage=round(leverage, 3),
            max_single_position_symbol=max_sym,
            max_single_position_pct=round(max_w, 4),
            max_sector_name=max_sec,
            max_sector_pct=round(max_sec_w, 4),
            estimated_var_95_pct=round(est_daily_var_pct, 4),
            cash_buffer_pct=round(cash_buffer_pct, 4),
        )

    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenario: StressScenario,
    ) -> StressTestResult:
        """Simulate the financial impact of a macroeconomic crisis shock on the portfolio."""
        nav_before = portfolio.net_asset_value
        asset_impacts: Dict[str, float] = {}
        total_pnl_shock = 0.0

        for pos in portfolio.positions:
            ac = pos.asset.asset_class
            shock_pct = 0.0

            if ac == AssetClass.EQUITY:
                # Scaled by beta to market
                shock_pct = scenario.equity_shock_pct * pos.asset.beta_to_market
            elif ac == AssetClass.FIXED_INCOME:
                # Rates shock in basis points impact duration approximation (-duration * dY)
                rates_impact = -(scenario.rates_shock_bps / 10000.0) * 6.5
                shock_pct = scenario.fixed_income_shock_pct + rates_impact
            elif ac == AssetClass.COMMODITY:
                shock_pct = scenario.commodity_shock_pct
            elif ac == AssetClass.FX:
                shock_pct = scenario.fx_shock_pct
            elif ac == AssetClass.CRYPTO:
                shock_pct = scenario.crypto_shock_pct
            else:
                shock_pct = scenario.equity_shock_pct * 0.5

            # Position PnL impact
            if pos.side.value == "LONG":
                pnl = pos.market_value * shock_pct
            else:
                pnl = -pos.market_value * shock_pct

            asset_impacts[pos.asset.symbol] = round(pnl, 2)
            total_pnl_shock += pnl

        nav_after = max(0.0, nav_before + total_pnl_shock)
        abs_impact = round(total_pnl_shock, 2)
        pct_impact = round(total_pnl_shock / nav_before, 4) if nav_before > 0 else 0.0

        capital_adequacy_passed = nav_after > (nav_before * 0.20)
        cushion_remaining = max(0.0, nav_after / nav_before) if nav_before > 0 else 0.0

        return StressTestResult(
            scenario=scenario,
            portfolio_value_before=round(nav_before, 2),
            portfolio_value_after=round(nav_after, 2),
            absolute_impact=abs_impact,
            percentage_impact=pct_impact,
            asset_impacts=asset_impacts,
            capital_adequacy_passed=capital_adequacy_passed,
            cushion_remaining_pct=round(cushion_remaining, 4),
        )

    @staticmethod
    def calculate_volatility_target_size(
        target_vol_annual: float,
        asset_vol_annual: float,
        portfolio_nav: float,
        max_weight_cap: float = 0.25,
    ) -> float:
        """Calculate volatility-targeted position dollar sizing."""
        if asset_vol_annual <= 0 or portfolio_nav <= 0:
            return 0.0
        raw_weight = target_vol_annual / asset_vol_annual
        weight = min(max_weight_cap, max(0.0, raw_weight))
        return round(weight * portfolio_nav, 2)
