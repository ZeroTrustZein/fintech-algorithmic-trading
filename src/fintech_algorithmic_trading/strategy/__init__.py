"""Quantitative alpha signals and strategy pipelines."""

from fintech_algorithmic_trading.strategy.signals import (
    AlphaStrategy,
    CompositeAlphaAggregator,
    MeanRevertingBollingerStrategy,
    PairsStatArbStrategy,
    TrendFollowingMACDStrategy,
    VolatilityBreakoutStrategy,
)

__all__ = [
    "AlphaStrategy",
    "CompositeAlphaAggregator",
    "MeanRevertingBollingerStrategy",
    "PairsStatArbStrategy",
    "TrendFollowingMACDStrategy",
    "VolatilityBreakoutStrategy",
]
