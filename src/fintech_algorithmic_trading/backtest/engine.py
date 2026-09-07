"""Event-driven algorithmic trading backtesting engine and quantitative performance attribution."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Union

import numpy as np

from fintech_algorithmic_trading.engine.var import calculate_historical_var
from fintech_algorithmic_trading.execution.oms import OrderManagementSystem
from fintech_algorithmic_trading.strategy.signals import AlphaStrategy, CompositeAlphaAggregator
from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
    Bar,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    PositionType,
    SignalDirection,
    StrategySignal,
)


class BacktestEngine:
    """Institutional event-driven backtesting engine with realistic slippage, commissions, and risk limits."""

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        oms: Optional[OrderManagementSystem] = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.oms = oms or OrderManagementSystem(
            commission_bps=self.config.commission_bps,
            slippage_bps=self.config.slippage_bps,
        )

    def run(
        self,
        symbol: str,
        bars: List[Bar],
        strategy: Union[AlphaStrategy, CompositeAlphaAggregator],
        portfolio_id: str = "backtest-account",
        max_position_pct: float = 0.35,
    ) -> BacktestResult:
        """Execute historical simulation sequentially bar-by-bar."""
        start_t = time.perf_counter()
        if len(bars) < 10:
            raise ValueError("Backtest requires at least 10 bars.")

        portfolio = Portfolio(
            id=portfolio_id,
            name=f"Backtest Portfolio ({symbol})",
            description="Simulation trading portfolio",
            cash=self.config.initial_capital,
            positions=[],
        )

        equity_curve: List[float] = []
        daily_returns: List[float] = []
        closed_trades: List[BacktestTrade] = []
        rolling_var_95: List[float] = []

        # Trade tracking state
        active_trade_entry_price: Optional[float] = None
        active_trade_entry_time = None
        active_trade_side: Optional[OrderSide] = None
        active_trade_qty: float = 0.0
        active_trade_start_bar: int = 0

        prev_nav = self.config.initial_capital

        for i, bar in enumerate(bars):
            # 1. Mark to market open positions
            for pos in portfolio.positions:
                if pos.asset.symbol == symbol:
                    pos.current_price = bar.close

            cur_nav = portfolio.net_asset_value
            equity_curve.append(cur_nav)

            # Daily return tracking
            if i > 0:
                ret = (cur_nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
                daily_returns.append(ret)
            prev_nav = cur_nav

            # Rolling 20-bar VaR
            if len(daily_returns) >= 20:
                recent_rets = np.array(daily_returns[-20:], dtype=np.float64)
                var_res = calculate_historical_var(
                    returns=recent_rets,
                    portfolio_value=cur_nav,
                    confidence_level=0.95,
                    horizon_days=1,
                )
                rolling_var_95.append(var_res.var_amount)
            else:
                rolling_var_95.append(0.0)

            # 2. Strategy Signal Evaluation on historical slice up to current bar
            bars_slice = bars[: i + 1]
            sig: Optional[StrategySignal]
            if isinstance(strategy, CompositeAlphaAggregator):
                sig = strategy.evaluate_composite_signal(symbol, bars_slice)
            else:
                sig = strategy.generate_signal(symbol, bars_slice)

            if sig is None:
                continue

            # 3. Position Sizing & Target Rebalancing
            target_pos = next((p for p in portfolio.positions if p.asset.symbol == symbol), None)
            cur_qty = target_pos.quantity if target_pos else 0.0
            cur_side = target_pos.side if target_pos else None

            # Determine desired target shares
            nav = portfolio.net_asset_value
            max_capital = nav * max_position_pct * abs(sig.strength)
            desired_shares = math.floor(max_capital / bar.close) if bar.close > 0 else 0

            if sig.direction == SignalDirection.FLAT:
                desired_shares = 0
            elif sig.direction == SignalDirection.SHORT:
                desired_shares = -desired_shares

            # Calculate trade delta
            current_net_shares = (
                cur_qty
                if cur_side == PositionType.LONG
                else -cur_qty
                if cur_side == PositionType.SHORT
                else 0.0
            )
            delta_shares = desired_shares - current_net_shares

            if abs(delta_shares) >= 1:
                side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
                qty = abs(delta_shares)

                order = self.oms.create_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                )

                asset_meta = Asset(
                    symbol=symbol,
                    name=symbol,
                    asset_class=AssetClass.EQUITY,
                    current_price=bar.close,
                    volatility_annual=0.25,
                    beta_to_market=1.0,
                )

                rep = self.oms.execute_order(
                    order=order,
                    portfolio=portfolio,
                    market_price=bar.close,
                    volume_daily=bar.volume,
                    asset_metadata=asset_meta,
                )

                if (
                    rep.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
                    and rep.filled_quantity > 0
                ):
                    # Record trade entry / exit
                    if active_trade_entry_price is not None and active_trade_side != side:
                        # Position reduction or exit
                        exit_price = rep.avg_fill_price
                        close_qty = min(active_trade_qty, rep.filled_quantity)
                        if active_trade_side == OrderSide.BUY:
                            pnl = (
                                exit_price - active_trade_entry_price
                            ) * close_qty - rep.total_commission
                            ret_pct = (
                                exit_price - active_trade_entry_price
                            ) / active_trade_entry_price
                        else:
                            pnl = (
                                active_trade_entry_price - exit_price
                            ) * close_qty - rep.total_commission
                            ret_pct = (
                                active_trade_entry_price - exit_price
                            ) / active_trade_entry_price

                        trade_record = BacktestTrade(
                            symbol=symbol,
                            side=active_trade_side,
                            quantity=close_qty,
                            entry_price=round(active_trade_entry_price, 4),
                            exit_price=round(exit_price, 4),
                            entry_time=active_trade_entry_time or bar.timestamp,
                            exit_time=bar.timestamp,
                            pnl=round(pnl, 2),
                            return_pct=round(ret_pct, 4),
                            commission=round(rep.total_commission, 2),
                            holding_period_bars=i - active_trade_start_bar,
                        )
                        closed_trades.append(trade_record)

                        rem_qty = active_trade_qty - close_qty
                        if rem_qty <= 0:
                            active_trade_entry_price = None
                            active_trade_side = None
                            active_trade_qty = 0.0
                        else:
                            active_trade_qty = rem_qty
                    else:
                        # New position entry or add-on
                        active_trade_entry_price = rep.avg_fill_price
                        active_trade_entry_time = bar.timestamp
                        active_trade_side = side
                        active_trade_qty += rep.filled_quantity
                        active_trade_start_bar = i

        # Close any lingering open position at end of backtest for final performance attribution
        if (
            active_trade_entry_price is not None
            and active_trade_qty > 0
            and active_trade_side is not None
        ):
            last_bar = bars[-1]
            last_px = last_bar.close
            if active_trade_side == OrderSide.BUY:
                pnl = (last_px - active_trade_entry_price) * active_trade_qty
                ret_pct = (last_px - active_trade_entry_price) / active_trade_entry_price
            else:
                pnl = (active_trade_entry_price - last_px) * active_trade_qty
                ret_pct = (active_trade_entry_price - last_px) / active_trade_entry_price

            closed_trades.append(
                BacktestTrade(
                    symbol=symbol,
                    side=active_trade_side,
                    quantity=active_trade_qty,
                    entry_price=round(active_trade_entry_price, 4),
                    exit_price=round(last_px, 4),
                    entry_time=active_trade_entry_time or last_bar.timestamp,
                    exit_time=last_bar.timestamp,
                    pnl=round(pnl, 2),
                    return_pct=round(ret_pct, 4),
                    commission=0.0,
                    holding_period_bars=len(bars) - active_trade_start_bar,
                )
            )

        # 4. Compute Metrics
        metrics = self._calculate_metrics(
            initial_cap=self.config.initial_capital,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=closed_trades,
            annualization_factor=self.config.annualization_factor,
            rf=self.config.risk_free_rate,
        )

        dur_ms = (time.perf_counter() - start_t) * 1000.0
        return BacktestResult(
            config=self.config,
            initial_capital=self.config.initial_capital,
            final_capital=round(equity_curve[-1], 2),
            equity_curve=[round(v, 2) for v in equity_curve],
            daily_returns=[round(r, 6) for r in daily_returns],
            trades=closed_trades,
            metrics=metrics,
            rolling_var_95=[round(v, 2) for v in rolling_var_95],
            computation_time_ms=round(dur_ms, 3),
        )

    @staticmethod
    def _calculate_metrics(
        initial_cap: float,
        equity_curve: List[float],
        daily_returns: List[float],
        trades: List[BacktestTrade],
        annualization_factor: int = 252,
        rf: float = 0.045,
    ) -> BacktestMetrics:
        """Compute institutional performance KPIs."""
        final_cap = equity_curve[-1] if equity_curve else initial_cap
        tot_ret = (final_cap - initial_cap) / initial_cap if initial_cap > 0 else 0.0

        n_bars = max(1, len(equity_curve))
        years = n_bars / float(annualization_factor)
        ann_ret = (
            (1.0 + tot_ret) ** (1.0 / years) - 1.0 if (years > 0 and 1.0 + tot_ret > 0) else 0.0
        )

        if daily_returns:
            r_arr = np.array(daily_returns, dtype=np.float64)
            ann_vol = float(np.std(r_arr) * np.sqrt(annualization_factor))
            excess_return = ann_ret - rf
            sharpe = excess_return / ann_vol if ann_vol > 1e-8 else 0.0

            downside = r_arr[r_arr < 0]
            downside_vol = (
                float(np.std(downside) * np.sqrt(annualization_factor))
                if len(downside) > 1
                else 1e-4
            )
            sortino = excess_return / downside_vol if downside_vol > 1e-8 else 0.0
        else:
            ann_vol = 0.0
            sharpe = 0.0
            sortino = 0.0

        # Drawdown computation
        eq_arr = np.array(equity_curve, dtype=np.float64)
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (running_max - eq_arr) / running_max
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Drawdown duration in bars
        cur_dd_duration = 0
        max_dd_duration = 0
        for dd in drawdowns:
            if dd > 0.0001:
                cur_dd_duration += 1
                if cur_dd_duration > max_dd_duration:
                    max_dd_duration = cur_dd_duration
            else:
                cur_dd_duration = 0

        calmar = ann_ret / max_dd if max_dd > 1e-6 else 0.0

        # Trade metrics
        tot_trades = len(trades)
        profitable_trades = [t for t in trades if t.pnl > 0]
        loss_trades = [t for t in trades if t.pnl <= 0]
        n_prof = len(profitable_trades)
        n_loss = len(loss_trades)

        win_rate = n_prof / tot_trades if tot_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in profitable_trades)
        gross_loss = abs(sum(t.pnl for t in loss_trades))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (gross_profit if gross_profit > 0 else 1.0)
        )

        avg_win = gross_profit / n_prof if n_prof > 0 else 0.0
        avg_loss = gross_loss / n_loss if n_loss > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        return BacktestMetrics(
            total_return_pct=round(tot_ret, 4),
            annualized_return=round(ann_ret, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            max_drawdown_pct=round(max_dd, 4),
            max_drawdown_duration_bars=max_dd_duration,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            expectancy=round(expectancy, 2),
            total_trades=tot_trades,
            profitable_trades=n_prof,
            loss_making_trades=n_loss,
        )
