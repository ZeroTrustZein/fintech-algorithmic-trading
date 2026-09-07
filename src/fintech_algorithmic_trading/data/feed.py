"""Market data feed, order book depth simulator, and streaming bar processor."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np

from fintech_algorithmic_trading.types import (
    Bar,
    OrderBookDepth,
    OrderBookLevel,
    Quote,
)


class MarketDataFeed:
    """Institutional market data feed provider with bar storage, quotes, and order book depth."""

    def __init__(self) -> None:
        self._bars: Dict[str, List[Bar]] = {}
        self._quotes: Dict[str, Quote] = {}
        self._bar_subscribers: Dict[str, List[Callable[[Bar], None]]] = {}
        self._quote_subscribers: Dict[str, List[Callable[[Quote], None]]] = {}

    def add_bar(self, bar: Bar, publish: bool = True) -> None:
        """Register a new OHLCV bar and optionally dispatch to subscribers."""
        if bar.symbol not in self._bars:
            self._bars[bar.symbol] = []
        self._bars[bar.symbol].append(bar)

        # Update synthetic top-of-book quote based on close
        spread_half = bar.close * 0.00025  # 2.5 bps half-spread
        q = Quote(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            bid=round(bar.close - spread_half, 4),
            ask=round(bar.close + spread_half, 4),
            bid_size=1000.0,
            ask_size=1000.0,
        )
        self._quotes[bar.symbol] = q

        if publish:
            for cb in self._bar_subscribers.get(bar.symbol, []):
                cb(bar)
            for q_cb in self._quote_subscribers.get(bar.symbol, []):
                q_cb(q)

    def load_bars(self, symbol: str, bars: List[Bar]) -> None:
        """Bulk load historical bars without publishing live events."""
        if symbol not in self._bars:
            self._bars[symbol] = []
        self._bars[symbol].extend(bars)
        if bars:
            latest = bars[-1]
            spread_half = latest.close * 0.00025
            self._quotes[symbol] = Quote(
                symbol=symbol,
                timestamp=latest.timestamp,
                bid=round(latest.close - spread_half, 4),
                ask=round(latest.close + spread_half, 4),
                bid_size=1000.0,
                ask_size=1000.0,
            )

    def get_bars(self, symbol: str, n_bars: Optional[int] = None) -> List[Bar]:
        """Retrieve stored bars for a symbol."""
        bars = self._bars.get(symbol, [])
        if n_bars is not None and n_bars > 0:
            return bars[-n_bars:]
        return list(bars)

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get the most recent bar for a symbol."""
        bars = self._bars.get(symbol, [])
        return bars[-1] if bars else None

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get the current top-of-book quote."""
        return self._quotes.get(symbol)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the current price (mid-quote or latest close)."""
        q = self.get_latest_quote(symbol)
        if q:
            return q.mid_price
        b = self.get_latest_bar(symbol)
        return b.close if b else None

    def subscribe_bar(self, symbol: str, callback: Callable[[Bar], None]) -> None:
        """Register a callback for bar arrival."""
        if symbol not in self._bar_subscribers:
            self._bar_subscribers[symbol] = []
        self._bar_subscribers[symbol].append(callback)

    def subscribe_quote(self, symbol: str, callback: Callable[[Quote], None]) -> None:
        """Register a callback for quote updates."""
        if symbol not in self._quote_subscribers:
            self._quote_subscribers[symbol] = []
        self._quote_subscribers[symbol].append(callback)

    def generate_order_book(
        self,
        symbol: str,
        levels: int = 5,
        tick_size: float = 0.01,
        depth_scaling: float = 1.25,
    ) -> OrderBookDepth:
        """Simulate a realistic Level 2 order book snapshot based on current market price."""
        ref_price = self.get_latest_price(symbol) or 100.0
        q = self.get_latest_quote(symbol)
        best_bid = q.bid if q else round(ref_price - 0.01, 2)
        best_ask = q.ask if q else round(ref_price + 0.01, 2)

        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []

        base_size = 500.0
        for i in range(levels):
            bid_px = round(best_bid - i * tick_size * max(1, int(ref_price * 0.001)), 4)
            ask_px = round(best_ask + i * tick_size * max(1, int(ref_price * 0.001)), 4)
            size = round(base_size * (depth_scaling**i), 1)

            bids.append(OrderBookLevel(price=max(0.01, bid_px), size=size, order_count=3 + i))
            asks.append(OrderBookLevel(price=ask_px, size=size, order_count=3 + i))

        return OrderBookDepth(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
        )

    def generate_synthetic_history(
        self,
        symbol: str,
        start_price: float = 100.0,
        n_bars: int = 252,
        volatility_annual: float = 0.20,
        drift_annual: float = 0.08,
        seed: Optional[int] = 42,
        start_time: Optional[datetime] = None,
        freq_minutes: int = 1440,  # 1 day by default
    ) -> List[Bar]:
        """Generate realistic synthetic OHLCV bars via Geometric Brownian Motion with intraday noise."""
        rng = np.random.default_rng(seed)
        dt = freq_minutes / (252.0 * 1440.0)
        t0 = start_time or (datetime.utcnow() - timedelta(minutes=freq_minutes * n_bars))

        # Generate close prices
        returns = rng.normal(
            (drift_annual - 0.5 * volatility_annual**2) * dt,
            volatility_annual * np.sqrt(dt),
            size=n_bars,
        )
        closes = start_price * np.exp(np.cumsum(returns))

        bars: List[Bar] = []
        cur_open = start_price
        for i in range(n_bars):
            c = float(closes[i])
            o = float(cur_open)
            # Intraday fluctuation
            high_wiggle = abs(float(rng.normal(0, volatility_annual * np.sqrt(dt) * 0.5)))
            low_wiggle = abs(float(rng.normal(0, volatility_annual * np.sqrt(dt) * 0.5)))
            h = max(o, c) * (1.0 + high_wiggle)
            low_px = min(o, c) * (1.0 - low_wiggle)
            # Volume correlated with price volatility and volume baseline
            vol = float(max(1000.0, rng.lognormal(mean=11.5, sigma=0.4)))

            bar_ts = t0 + timedelta(minutes=freq_minutes * (i + 1))
            bar = Bar(
                symbol=symbol,
                timestamp=bar_ts,
                open=round(o, 4),
                high=round(h, 4),
                low=round(max(0.01, low_px), 4),
                close=round(c, 4),
                volume=round(vol, 0),
            )
            bars.append(bar)
            cur_open = c

        self.load_bars(symbol, bars)
        return bars

    def resample_bars(self, symbol: str, aggregation_factor: int = 5) -> List[Bar]:
        """Resample sequential bars into lower frequency bars (e.g. 5m to 25m)."""
        raw_bars = self.get_bars(symbol)
        if not raw_bars or aggregation_factor <= 1:
            return raw_bars

        resampled: List[Bar] = []
        for i in range(0, len(raw_bars), aggregation_factor):
            chunk = raw_bars[i : i + aggregation_factor]
            if not chunk:
                continue
            o = chunk[0].open
            h = max(b.high for b in chunk)
            low_px = min(b.low for b in chunk)
            c = chunk[-1].close
            v = sum(b.volume for b in chunk)
            ts = chunk[-1].timestamp
            resampled.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts,
                    open=round(o, 4),
                    high=round(h, 4),
                    low=round(low_px, 4),
                    close=round(c, 4),
                    volume=round(v, 0),
                )
            )
        return resampled
