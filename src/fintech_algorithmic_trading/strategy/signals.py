"""Quantitative alpha strategy pipelines, technical indicators, and signal aggregators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from fintech_algorithmic_trading.types import (
    Bar,
    SignalDirection,
    StrategySignal,
)


class AlphaStrategy(ABC):
    """Abstract base class for quantitative alpha generation algorithms."""

    def __init__(self, strategy_id: str, name: str) -> None:
        self.strategy_id = strategy_id
        self.name = name

    @abstractmethod
    def generate_signal(self, symbol: str, bars: List[Bar]) -> Optional[StrategySignal]:
        """Process price bars and emit trading signal."""
        pass


class TrendFollowingMACDStrategy(AlphaStrategy):
    """Dual EMA momentum and MACD histogram trend-following strategy."""

    def __init__(
        self,
        strategy_id: str = "trend-macd",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        super().__init__(strategy_id=strategy_id, name="Trend Following MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @staticmethod
    def _compute_ema(series: np.ndarray, span: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (span + 1.0)
        ema = np.empty_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
        return ema

    def generate_signal(self, symbol: str, bars: List[Bar]) -> Optional[StrategySignal]:
        min_bars = self.slow_period + self.signal_period
        if len(bars) < min_bars:
            return None

        closes = np.array([b.close for b in bars], dtype=np.float64)
        fast_ema = self._compute_ema(closes, self.fast_period)
        slow_ema = self._compute_ema(closes, self.slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = self._compute_ema(macd_line, self.signal_period)
        histogram = macd_line - signal_line

        latest_hist = float(histogram[-1])
        prev_hist = float(histogram[-2])
        latest_price = float(closes[-1])
        latest_bar = bars[-1]

        # Normalized histogram strength relative to price
        hist_norm = latest_hist / latest_price
        strength = float(np.clip(hist_norm * 50.0, -1.0, 1.0))

        if latest_hist > 0 and prev_hist <= 0:
            direction = SignalDirection.LONG
            confidence = min(1.0, max(0.5, abs(strength) * 2.0))
            rationale = (
                f"Bullish MACD crossover: hist surged from {prev_hist:.4f} to {latest_hist:.4f}"
            )
        elif latest_hist < 0 and prev_hist >= 0:
            direction = SignalDirection.SHORT
            confidence = min(1.0, max(0.5, abs(strength) * 2.0))
            rationale = (
                f"Bearish MACD crossover: hist fell from {prev_hist:.4f} to {latest_hist:.4f}"
            )
        elif latest_hist > 0:
            direction = SignalDirection.LONG
            confidence = min(1.0, max(0.3, abs(strength)))
            rationale = "Positive MACD trend continuation"
        elif latest_hist < 0:
            direction = SignalDirection.SHORT
            confidence = min(1.0, max(0.3, abs(strength)))
            rationale = "Negative MACD trend continuation"
        else:
            direction = SignalDirection.FLAT
            confidence = 0.0
            rationale = "Neutral MACD histogram"

        return StrategySignal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            timestamp=latest_bar.timestamp,
            rationale=rationale,
            metadata={
                "macd": round(float(macd_line[-1]), 4),
                "signal": round(float(signal_line[-1]), 4),
                "histogram": round(latest_hist, 4),
            },
        )


class MeanRevertingBollingerStrategy(AlphaStrategy):
    """Mean reversion strategy using Bollinger Bands and Price Z-Scores."""

    def __init__(
        self,
        strategy_id: str = "mean-revert-bb",
        window: int = 20,
        num_std: float = 2.0,
    ) -> None:
        super().__init__(strategy_id=strategy_id, name="Bollinger Mean Reversion")
        self.window = window
        self.num_std = num_std

    def generate_signal(self, symbol: str, bars: List[Bar]) -> Optional[StrategySignal]:
        if len(bars) < self.window:
            return None

        closes = np.array([b.close for b in bars[-self.window :]], dtype=np.float64)
        mean_px = float(np.mean(closes))
        std_px = float(np.std(closes))
        if std_px <= 1e-8:
            return None

        latest_px = float(closes[-1])
        z_score = (latest_px - mean_px) / std_px
        upper_band = mean_px + self.num_std * std_px
        lower_band = mean_px - self.num_std * std_px
        latest_bar = bars[-1]

        # Invert direction for mean reversion
        if z_score <= -self.num_std:
            direction = SignalDirection.LONG
            strength = float(np.clip((abs(z_score) - self.num_std) * 0.5 + 0.5, 0.0, 1.0))
            confidence = min(1.0, abs(z_score) / (self.num_std * 1.5))
            rationale = f"Oversold: price {latest_px:.2f} below lower band {lower_band:.2f} (z={z_score:.2f})"
        elif z_score >= self.num_std:
            direction = SignalDirection.SHORT
            strength = -float(np.clip((abs(z_score) - self.num_std) * 0.5 + 0.5, 0.0, 1.0))
            confidence = min(1.0, abs(z_score) / (self.num_std * 1.5))
            rationale = f"Overbought: price {latest_px:.2f} above upper band {upper_band:.2f} (z={z_score:.2f})"
        elif abs(z_score) < 0.5:
            direction = SignalDirection.FLAT
            strength = 0.0
            confidence = 0.8
            rationale = f"Mean reverted: price {latest_px:.2f} near moving average {mean_px:.2f}"
        else:
            direction = SignalDirection.LONG if z_score < 0 else SignalDirection.SHORT
            strength = -float(z_score / self.num_std)
            confidence = 0.4
            rationale = f"Mild mean-reversion drift (z={z_score:.2f})"

        return StrategySignal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            timestamp=latest_bar.timestamp,
            rationale=rationale,
            metadata={
                "z_score": round(z_score, 4),
                "upper_band": round(upper_band, 2),
                "lower_band": round(lower_band, 2),
                "mean": round(mean_px, 2),
            },
        )


class VolatilityBreakoutStrategy(AlphaStrategy):
    """Donchian channel breakout with Average True Range (ATR) volatility expansion."""

    def __init__(
        self,
        strategy_id: str = "vol-breakout",
        lookback_period: int = 20,
        atr_period: int = 14,
    ) -> None:
        super().__init__(strategy_id=strategy_id, name="Volatility Breakout")
        self.lookback_period = lookback_period
        self.atr_period = atr_period

    def generate_signal(self, symbol: str, bars: List[Bar]) -> Optional[StrategySignal]:
        req_bars = max(self.lookback_period, self.atr_period) + 1
        if len(bars) < req_bars:
            return None

        recent_bars = bars[-(self.lookback_period + 1) : -1]
        highest_high = max(b.high for b in recent_bars)
        lowest_low = min(b.low for b in recent_bars)
        cur_bar = bars[-1]

        # Calculate ATR
        tr_list = []
        for i in range(1, len(bars[-self.atr_period - 1 :])):
            b_prev = bars[-self.atr_period - 1 + i - 1]
            b_cur = bars[-self.atr_period - 1 + i]
            tr = max(
                b_cur.high - b_cur.low,
                abs(b_cur.high - b_prev.close),
                abs(b_cur.low - b_prev.close),
            )
            tr_list.append(tr)
        atr = float(np.mean(tr_list)) if tr_list else 1.0

        if cur_bar.close > highest_high:
            breakout_excess = (cur_bar.close - highest_high) / max(1e-4, atr)
            strength = float(np.clip(breakout_excess * 0.5, 0.2, 1.0))
            direction = SignalDirection.LONG
            confidence = min(1.0, 0.5 + breakout_excess * 0.25)
            rationale = f"Bullish breakout above {self.lookback_period}-bar high {highest_high:.2f} (ATR={atr:.2f})"
        elif cur_bar.close < lowest_low:
            breakout_excess = (lowest_low - cur_bar.close) / max(1e-4, atr)
            strength = -float(np.clip(breakout_excess * 0.5, 0.2, 1.0))
            direction = SignalDirection.SHORT
            confidence = min(1.0, 0.5 + breakout_excess * 0.25)
            rationale = f"Bearish breakdown below {self.lookback_period}-bar low {lowest_low:.2f} (ATR={atr:.2f})"
        else:
            direction = SignalDirection.FLAT
            strength = 0.0
            confidence = 0.2
            rationale = f"Price inside Donchian channel [{lowest_low:.2f}, {highest_high:.2f}]"

        return StrategySignal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            timestamp=cur_bar.timestamp,
            rationale=rationale,
            metadata={
                "atr": round(atr, 4),
                "highest_high": round(highest_high, 2),
                "lowest_low": round(lowest_low, 2),
            },
        )


class PairsStatArbStrategy:
    """Statistical Arbitrage pairs trading model on cointegrated spread."""

    def __init__(
        self,
        strategy_id: str = "pairs-statarb",
        symbol_a: str = "SPY",
        symbol_b: str = "QQQ",
        lookback_period: int = 30,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
    ) -> None:
        self.strategy_id = strategy_id
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score

    def generate_pair_signals(
        self,
        bars_a: List[Bar],
        bars_b: List[Bar],
    ) -> Tuple[Optional[StrategySignal], Optional[StrategySignal]]:
        """Calculate OLS hedge ratio, spread z-score, and emit paired trade signals."""
        min_len = min(len(bars_a), len(bars_b))
        if min_len < self.lookback_period:
            return None, None

        closes_a = np.array([b.close for b in bars_a[-self.lookback_period :]], dtype=np.float64)
        closes_b = np.array([b.close for b in bars_b[-self.lookback_period :]], dtype=np.float64)

        # OLS regression of A on B: A = alpha + beta * B
        cov = np.cov(closes_b, closes_a)[0, 1]
        var_b = np.var(closes_b)
        hedge_ratio = float(cov / var_b) if var_b > 1e-8 else 1.0

        spread = closes_a - hedge_ratio * closes_b
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        if spread_std <= 1e-8:
            return None, None

        cur_spread = float(spread[-1])
        z = (cur_spread - spread_mean) / spread_std
        ts = bars_a[-1].timestamp

        if z <= -self.entry_z_score:
            # Spread is too low -> Long A, Short B
            dir_a, dir_b = SignalDirection.LONG, SignalDirection.SHORT
            str_a = min(1.0, abs(z) / 3.0)
            str_b = -min(1.0, abs(z) / 3.0)
            rat = (
                f"Pair spread underpriced (z={z:.2f}): Long {self.symbol_a}, Short {self.symbol_b}"
            )
        elif z >= self.entry_z_score:
            # Spread is too high -> Short A, Long B
            dir_a, dir_b = SignalDirection.SHORT, SignalDirection.LONG
            str_a = -min(1.0, abs(z) / 3.0)
            str_b = min(1.0, abs(z) / 3.0)
            rat = f"Pair spread overpriced (z={z:.2f}): Short {self.symbol_a}, Long {self.symbol_b}"
        elif abs(z) <= self.exit_z_score:
            dir_a, dir_b = SignalDirection.FLAT, SignalDirection.FLAT
            str_a, str_b = 0.0, 0.0
            rat = f"Pair spread normalized (z={z:.2f}): Close positions"
        else:
            dir_a, dir_b = SignalDirection.FLAT, SignalDirection.FLAT
            str_a, str_b = 0.0, 0.0
            rat = f"Pair spread in no-trade band (z={z:.2f})"

        sig_a = StrategySignal(
            strategy_id=self.strategy_id,
            symbol=self.symbol_a,
            direction=dir_a,
            strength=round(str_a, 4),
            confidence=0.85,
            timestamp=ts,
            rationale=rat,
            metadata={"hedge_ratio": round(hedge_ratio, 4), "spread_z": round(z, 4)},
        )
        sig_b = StrategySignal(
            strategy_id=self.strategy_id,
            symbol=self.symbol_b,
            direction=dir_b,
            strength=round(str_b, 4),
            confidence=0.85,
            timestamp=ts,
            rationale=rat,
            metadata={"hedge_ratio": round(hedge_ratio, 4), "spread_z": round(z, 4)},
        )
        return sig_a, sig_b


class CompositeAlphaAggregator:
    """Ensemble blending of multiple alpha signals with strategic weighting."""

    def __init__(self, strategies: Optional[Dict[AlphaStrategy, float]] = None) -> None:
        self.strategies: Dict[AlphaStrategy, float] = strategies or {
            TrendFollowingMACDStrategy(): 0.40,
            MeanRevertingBollingerStrategy(): 0.35,
            VolatilityBreakoutStrategy(): 0.25,
        }

    def evaluate_composite_signal(self, symbol: str, bars: List[Bar]) -> StrategySignal:
        """Combine weighted alpha signals into single directional stance and strength."""
        if not bars:
            return StrategySignal(
                strategy_id="composite-ensemble",
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.0,
                rationale="Empty bar history",
            )

        weighted_strength = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        sub_signals = []

        for strat, weight in self.strategies.items():
            sig = strat.generate_signal(symbol, bars)
            if sig is not None:
                numeric_dir = (
                    1.0
                    if sig.direction == SignalDirection.LONG
                    else -1.0
                    if sig.direction == SignalDirection.SHORT
                    else 0.0
                )
                weighted_strength += numeric_dir * abs(sig.strength) * sig.confidence * weight
                weighted_confidence += sig.confidence * weight
                total_weight += weight
                sub_signals.append(f"{strat.name}: {sig.direction.value} (str={sig.strength:.2f})")

        if total_weight > 0:
            final_strength = weighted_strength / total_weight
            final_conf = weighted_confidence / total_weight
        else:
            final_strength = 0.0
            final_conf = 0.0

        if final_strength > 0.15:
            final_dir = SignalDirection.LONG
        elif final_strength < -0.15:
            final_dir = SignalDirection.SHORT
        else:
            final_dir = SignalDirection.FLAT

        latest_ts = bars[-1].timestamp if bars else datetime.utcnow()
        rationale = (
            " | ".join(sub_signals) if sub_signals else "No strategy generated active signal"
        )

        return StrategySignal(
            strategy_id="composite-ensemble",
            symbol=symbol,
            direction=final_dir,
            strength=round(float(np.clip(final_strength, -1.0, 1.0)), 4),
            confidence=round(float(np.clip(final_conf, 0.0, 1.0)), 4),
            timestamp=latest_ts,
            rationale=rationale,
            metadata={"sub_strategy_count": len(sub_signals)},
        )
