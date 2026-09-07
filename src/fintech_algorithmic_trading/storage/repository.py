"""Market data, portfolio presets, and stress test repository."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    Portfolio,
    Position,
    PositionType,
    StressScenario,
)


class PortfolioRepository:
    """Repository of benchmark trading portfolios, market profiles, and historical stress scenarios."""

    @staticmethod
    def get_benchmark_portfolios() -> Dict[str, Portfolio]:
        """Return institutional benchmark portfolios."""
        portfolios = {}

        # 1. Global Macro Long/Short
        gm_positions = [
            Position(
                asset=Asset(
                    symbol="SPY",
                    name="SPDR S&P 500 ETF Trust",
                    asset_class=AssetClass.EQUITY,
                    sector="Broad Market",
                    current_price=540.0,
                    volatility_annual=0.16,
                    beta_to_market=1.0,
                ),
                quantity=1000,
                entry_price=520.0,
                current_price=540.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="QQQ",
                    name="Invesco QQQ Trust",
                    asset_class=AssetClass.EQUITY,
                    sector="Technology",
                    current_price=480.0,
                    volatility_annual=0.22,
                    beta_to_market=1.20,
                ),
                quantity=600,
                entry_price=450.0,
                current_price=480.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="TLT",
                    name="iShares 20+ Year Treasury Bond ETF",
                    asset_class=AssetClass.FIXED_INCOME,
                    sector="Sovereign Debt",
                    current_price=92.0,
                    volatility_annual=0.14,
                    beta_to_market=-0.25,
                ),
                quantity=2000,
                entry_price=95.0,
                current_price=92.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="GLD",
                    name="SPDR Gold Shares",
                    asset_class=AssetClass.COMMODITY,
                    sector="Precious Metals",
                    current_price=220.0,
                    volatility_annual=0.15,
                    beta_to_market=0.10,
                ),
                quantity=800,
                entry_price=210.0,
                current_price=220.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="USO",
                    name="United States Oil Fund",
                    asset_class=AssetClass.COMMODITY,
                    sector="Energy",
                    current_price=75.0,
                    volatility_annual=0.32,
                    beta_to_market=0.40,
                ),
                quantity=1200,
                entry_price=80.0,
                current_price=75.0,
                side=PositionType.SHORT,
            ),
        ]
        portfolios["global-macro"] = Portfolio(
            id="global-macro",
            name="Global Macro Long/Short Strategy",
            description="Multi-asset macro allocation across equities, treasuries, gold, and crude oil hedge.",
            cash=150000.0,
            positions=gm_positions,
        )

        # 2. Tech Growth & Momentum
        tech_positions = [
            Position(
                asset=Asset(
                    symbol="NVDA",
                    name="NVIDIA Corporation",
                    asset_class=AssetClass.EQUITY,
                    sector="Semiconductors",
                    current_price=125.0,
                    volatility_annual=0.45,
                    beta_to_market=1.85,
                ),
                quantity=3000,
                entry_price=110.0,
                current_price=125.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="MSFT",
                    name="Microsoft Corporation",
                    asset_class=AssetClass.EQUITY,
                    sector="Software",
                    current_price=430.0,
                    volatility_annual=0.24,
                    beta_to_market=1.15,
                ),
                quantity=800,
                entry_price=410.0,
                current_price=430.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="AAPL",
                    name="Apple Inc.",
                    asset_class=AssetClass.EQUITY,
                    sector="Consumer Electronics",
                    current_price=220.0,
                    volatility_annual=0.20,
                    beta_to_market=1.05,
                ),
                quantity=1200,
                entry_price=205.0,
                current_price=220.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="TSLA",
                    name="Tesla, Inc.",
                    asset_class=AssetClass.EQUITY,
                    sector="Automotive & Clean Energy",
                    current_price=240.0,
                    volatility_annual=0.55,
                    beta_to_market=2.10,
                ),
                quantity=1000,
                entry_price=220.0,
                current_price=240.0,
                side=PositionType.LONG,
            ),
        ]
        portfolios["tech-momentum"] = Portfolio(
            id="tech-momentum",
            name="Mega-Cap Tech Alpha Momentum",
            description="Aggressive high-beta semiconductor and cloud computing growth mandate.",
            cash=80000.0,
            positions=tech_positions,
        )

        # 3. Multi-Asset Balanced (60/40 Institutional)
        balanced_positions = [
            Position(
                asset=Asset(
                    symbol="VTI",
                    name="Vanguard Total Stock Market ETF",
                    asset_class=AssetClass.EQUITY,
                    sector="US Total Equities",
                    current_price=265.0,
                    volatility_annual=0.15,
                    beta_to_market=1.0,
                    dividend_yield=0.015,
                ),
                quantity=3000,
                entry_price=250.0,
                current_price=265.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="BND",
                    name="Vanguard Total Bond Market ETF",
                    asset_class=AssetClass.FIXED_INCOME,
                    sector="Broad Fixed Income",
                    current_price=72.0,
                    volatility_annual=0.07,
                    beta_to_market=0.05,
                    dividend_yield=0.038,
                ),
                quantity=7000,
                entry_price=73.0,
                current_price=72.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="IAU",
                    name="iShares Gold Trust",
                    asset_class=AssetClass.COMMODITY,
                    sector="Precious Metals",
                    current_price=46.0,
                    volatility_annual=0.15,
                    beta_to_market=0.08,
                ),
                quantity=2500,
                entry_price=42.0,
                current_price=46.0,
                side=PositionType.LONG,
            ),
        ]
        portfolios["multi-asset-balanced"] = Portfolio(
            id="multi-asset-balanced",
            name="Institutional Multi-Asset Balanced (60/40 + Gold)",
            description="Risk-managed asset allocation balancing equity upside, bond duration, and gold hedge.",
            cash=100000.0,
            positions=balanced_positions,
        )

        # 4. Crypto & FX Alpha
        crypto_positions = [
            Position(
                asset=Asset(
                    symbol="BTC",
                    name="Bitcoin",
                    asset_class=AssetClass.CRYPTO,
                    sector="Layer 1 Store of Value",
                    current_price=64000.0,
                    volatility_annual=0.60,
                    beta_to_market=0.85,
                ),
                quantity=5.0,
                entry_price=61000.0,
                current_price=64000.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="ETH",
                    name="Ethereum",
                    asset_class=AssetClass.CRYPTO,
                    sector="Smart Contract Platform",
                    current_price=3400.0,
                    volatility_annual=0.72,
                    beta_to_market=1.15,
                ),
                quantity=40.0,
                entry_price=3200.0,
                current_price=3400.0,
                side=PositionType.LONG,
            ),
            Position(
                asset=Asset(
                    symbol="EURUSD",
                    name="Euro / US Dollar FX",
                    asset_class=AssetClass.FX,
                    sector="G10 FX",
                    current_price=1.09,
                    volatility_annual=0.08,
                    beta_to_market=0.20,
                ),
                quantity=200000.0,
                entry_price=1.085,
                current_price=1.09,
                side=PositionType.LONG,
            ),
        ]
        portfolios["crypto-fx-alpha"] = Portfolio(
            id="crypto-fx-alpha",
            name="Liquid Digital Assets & FX Arbitrage",
            description="Systematic crypto momentum and major currency cross-rate portfolio.",
            cash=50000.0,
            positions=crypto_positions,
        )

        return portfolios

    @staticmethod
    def get_portfolio(portfolio_id: str) -> Optional[Portfolio]:
        """Fetch portfolio by ID."""
        benchmarks = PortfolioRepository.get_benchmark_portfolios()
        return benchmarks.get(portfolio_id)

    @staticmethod
    def get_stress_scenarios() -> Dict[str, StressScenario]:
        """Return historical crisis and macroeconomic shock scenarios."""
        return {
            "2008-lehman": StressScenario(
                id="2008-lehman",
                name="2008 Global Financial Crisis (Lehman Shock)",
                description="Severe credit freeze, equity liquidation, flight to long treasuries, commodity plunge.",
                equity_shock_pct=-0.45,
                fixed_income_shock_pct=0.12,
                rates_shock_bps=-300.0,
                commodity_shock_pct=-0.35,
                crypto_shock_pct=-0.70,
                volatility_multiplier=2.8,
            ),
            "2020-covid-crash": StressScenario(
                id="2020-covid-crash",
                name="March 2020 COVID-19 Liquidity Shock",
                description="Rapid cross-asset correlation spike to 1, crude oil collapse, rapid volatility surge.",
                equity_shock_pct=-0.34,
                fixed_income_shock_pct=0.05,
                rates_shock_bps=-150.0,
                commodity_shock_pct=-0.45,
                crypto_shock_pct=-0.50,
                volatility_multiplier=3.5,
            ),
            "2022-inflation-rates-hike": StressScenario(
                id="2022-inflation-rates-hike",
                name="2022 Aggressive Fed Rate Hikes & Tech De-rating",
                description="Simultaneous equity and bond drawdown with commodities surging on supply shock.",
                equity_shock_pct=-0.22,
                fixed_income_shock_pct=-0.16,
                rates_shock_bps=350.0,
                commodity_shock_pct=0.28,
                crypto_shock_pct=-0.65,
                volatility_multiplier=1.8,
            ),
            "flash-crash-liquidity-freeze": StressScenario(
                id="flash-crash-liquidity-freeze",
                name="Algorithmic Flash Crash & Liquidity Evaporation",
                description="Sudden cascade of stop-losses, bid-ask widening, and rapid intraday drawdowns.",
                equity_shock_pct=-0.14,
                fixed_income_shock_pct=0.02,
                rates_shock_bps=-40.0,
                commodity_shock_pct=-0.10,
                crypto_shock_pct=-0.28,
                volatility_multiplier=4.0,
            ),
        }

    @staticmethod
    def generate_synthetic_returns(
        n_days: int = 500,
        mean_daily: float = 0.0004,
        vol_daily: float = 0.012,
        fat_tailed: bool = True,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate realistic synthetic daily return series with Student's-t fat tails."""
        rng = np.random.default_rng(seed)
        if fat_tailed:
            # Student-t distribution with df=5 exhibits realistic financial kurtosis
            df = 5
            raw = rng.standard_t(df, size=n_days)
            # Scale variance to match vol_daily: Var(t) = df / (df - 2)
            scaled = raw * np.sqrt((df - 2) / df) * vol_daily + mean_daily
            return scaled
        else:
            return rng.normal(mean_daily, vol_daily, size=n_days)
