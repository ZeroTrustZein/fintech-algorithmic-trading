"""Order Management System (OMS), execution simulator, and pre-trade risk gating."""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    ExecutionReport,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionType,
    Quote,
    RiskLimits,
)


class SlippageModel:
    """Quantitative slippage estimation models."""

    @staticmethod
    def calculate_fixed_bps_slippage(
        price: float, side: OrderSide, slippage_bps: float = 2.5
    ) -> float:
        """Fixed basis points slippage."""
        slip_pct = slippage_bps / 10000.0
        return price * (1.0 + slip_pct) if side == OrderSide.BUY else price * (1.0 - slip_pct)

    @staticmethod
    def calculate_market_impact_slippage(
        price: float,
        side: OrderSide,
        quantity: float,
        daily_volume: float = 1_000_000.0,
        volatility_annual: float = 0.20,
        impact_factor: float = 0.10,
    ) -> Tuple[float, float]:
        """Square-root market impact model (Barra/Almgren-Chriss stylized).

        Returns:
            (executed_price, slippage_amount_per_share)
        """
        participation = max(1e-6, quantity / max(100.0, daily_volume))
        impact_pct = impact_factor * math.sqrt(participation) * (volatility_annual / math.sqrt(252))
        slip_amount = price * impact_pct

        executed_px = (
            price + slip_amount if side == OrderSide.BUY else max(0.01, price - slip_amount)
        )
        return round(executed_px, 4), round(slip_amount, 4)


class PreTradeRiskGatekeeper:
    """Institutional pre-trade risk policy validation filter."""

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        max_order_notional: float = 1_000_000.0,
    ) -> None:
        self.risk_limits = risk_limits or RiskLimits()
        self.max_order_notional = max_order_notional

    def evaluate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        reference_price: float,
    ) -> Tuple[bool, str, float]:
        """Validate order against leverage, position concentration, and order notional bounds.

        Returns:
            (is_approved, rejection_or_warning_reason, approved_quantity)
        """
        nav = max(1.0, portfolio.net_asset_value)
        order_notional = order.quantity * reference_price

        # 1. Single order notional limit
        if order_notional > self.max_order_notional:
            allowed_qty = math.floor(self.max_order_notional / reference_price)
            return (
                False,
                f"Order notional ${order_notional:,.2f} exceeds max order limit ${self.max_order_notional:,.2f}. Scaled to {allowed_qty} shares.",
                allowed_qty,
            )

        # 2. Buying power check for BUY orders
        if order.side == OrderSide.BUY and order_notional > portfolio.cash:
            # Allow partial fill up to cash if policy allows
            affordable_qty = math.floor(portfolio.cash / reference_price)
            if affordable_qty <= 0:
                return (
                    False,
                    f"Insufficient unencumbered cash (${portfolio.cash:,.2f}) for order notional (${order_notional:,.2f}).",
                    0.0,
                )

        # 3. Post-trade position concentration limit
        existing_pos = next(
            (p for p in portfolio.positions if p.asset.symbol == order.symbol), None
        )
        existing_val = existing_pos.market_value if existing_pos else 0.0
        # If buying to cover a short, position value decreases or flips
        if existing_pos and existing_pos.side == PositionType.SHORT and order.side == OrderSide.BUY:
            post_trade_val = abs(existing_val - order_notional)
        elif existing_pos and existing_pos.side == PositionType.LONG and order.side == OrderSide.SELL:
            post_trade_val = abs(existing_val - order_notional)
        else:
            post_trade_val = existing_val + order_notional

        post_trade_pct = post_trade_val / nav
        if post_trade_pct > self.risk_limits.max_single_position_pct:
            max_allowed_val = nav * self.risk_limits.max_single_position_pct
            room_val = max(0.0, max_allowed_val - existing_val)
            scaled_qty = math.floor(room_val / reference_price)
            return (
                False,
                f"Projected concentration {post_trade_pct * 100:.2f}% breaches limit {self.risk_limits.max_single_position_pct * 100:.1f}%.",
                scaled_qty,
            )

        # 4. Post-trade leverage limit
        if (existing_pos and existing_pos.side == PositionType.SHORT and order.side == OrderSide.BUY) or (
            existing_pos and existing_pos.side == PositionType.LONG and order.side == OrderSide.SELL
        ):
            post_gross = max(0.0, portfolio.gross_exposure - min(existing_val, order_notional))
        else:
            post_gross = portfolio.gross_exposure + order_notional
        post_leverage = post_gross / nav
        if post_leverage > self.risk_limits.max_leverage:
            return (
                False,
                f"Projected leverage {post_leverage:.2f}x breaches policy ceiling {self.risk_limits.max_leverage:.2f}x.",
                0.0,
            )

        return True, "Pre-trade risk audit passed", order.quantity


class OrderManagementSystem:
    """Order routing, execution simulation, and portfolio synchronization engine."""

    def __init__(
        self,
        gatekeeper: Optional[PreTradeRiskGatekeeper] = None,
        commission_bps: float = 5.0,  # 5 bps
        min_commission: float = 1.0,
        slippage_bps: float = 2.5,
    ) -> None:
        self.gatekeeper = gatekeeper or PreTradeRiskGatekeeper()
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, List[Fill]] = {}

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Construct and register a new order."""
        ord_id = f"ord-{uuid.uuid4().hex[:8]}"
        order = Order(
            id=ord_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=datetime.utcnow(),
            status=OrderStatus.PENDING,
        )
        self.orders[ord_id] = order
        self.fills[ord_id] = []
        return order

    def execute_order(
        self,
        order: Order,
        portfolio: Portfolio,
        market_price: float,
        quote: Optional[Quote] = None,
        volume_daily: float = 1_000_000.0,
        asset_metadata: Optional[Asset] = None,
    ) -> ExecutionReport:
        """Route order through pre-trade risk gating, simulate execution, and apply portfolio updates."""
        # 1. Pre-trade Risk Audit
        approved, reason, allowed_qty = self.gatekeeper.evaluate_order(
            order, portfolio, reference_price=market_price
        )
        if not approved and allowed_qty <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = reason
            return ExecutionReport(
                order_id=order.id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                remaining_quantity=order.quantity,
                avg_fill_price=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                rejection_reason=reason,
            )

        fill_quantity = order.quantity if approved else allowed_qty

        # 2. Check Order Conditionals
        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.side == OrderSide.BUY and market_price > order.price:
                # Limit not touched
                return ExecutionReport(
                    order_id=order.id,
                    symbol=order.symbol,
                    status=OrderStatus.PENDING,
                    filled_quantity=0.0,
                    remaining_quantity=order.quantity,
                    avg_fill_price=0.0,
                    total_commission=0.0,
                    total_slippage=0.0,
                    rejection_reason=f"Market price {market_price:.2f} above buy limit {order.price:.2f}",
                )
            elif order.side == OrderSide.SELL and market_price < order.price:
                return ExecutionReport(
                    order_id=order.id,
                    symbol=order.symbol,
                    status=OrderStatus.PENDING,
                    filled_quantity=0.0,
                    remaining_quantity=order.quantity,
                    avg_fill_price=0.0,
                    total_commission=0.0,
                    total_slippage=0.0,
                    rejection_reason=f"Market price {market_price:.2f} below sell limit {order.price:.2f}",
                )

        if order.order_type == OrderType.STOP_LOSS and order.stop_price is not None:
            if order.side == OrderSide.SELL and market_price > order.stop_price:
                return ExecutionReport(
                    order_id=order.id,
                    symbol=order.symbol,
                    status=OrderStatus.PENDING,
                    filled_quantity=0.0,
                    remaining_quantity=order.quantity,
                    avg_fill_price=0.0,
                    total_commission=0.0,
                    total_slippage=0.0,
                    rejection_reason="Stop price not triggered",
                )

        # 3. Simulate Slices (TWAP / VWAP) or Single Fill
        fills: List[Fill] = []
        n_slices = 4 if order.order_type in (OrderType.TWAP, OrderType.VWAP) else 1
        slice_qty = fill_quantity / n_slices

        for _ in range(n_slices):
            base_px = (
                quote.ask
                if (quote and order.side == OrderSide.BUY)
                else quote.bid
                if (quote and order.side == OrderSide.SELL)
                else market_price
            )
            exec_px, slip_val = SlippageModel.calculate_market_impact_slippage(
                price=base_px,
                side=order.side,
                quantity=slice_qty,
                daily_volume=volume_daily,
            )
            slice_notional = slice_qty * exec_px
            comm = max(self.min_commission, slice_notional * (self.commission_bps / 10000.0))

            f = Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=round(slice_qty, 4),
                price=round(exec_px, 4),
                commission=round(comm, 2),
                slippage=round(slip_val * slice_qty, 2),
                timestamp=datetime.utcnow(),
            )
            fills.append(f)

        if order.id not in self.fills:
            self.fills[order.id] = []
        self.fills[order.id].extend(fills)
        if order.id not in self.orders:
            self.orders[order.id] = order

        # Aggregate fill execution
        tot_qty = sum(f.quantity for f in fills)
        tot_cost = sum(f.quantity * f.price for f in fills)
        avg_px = tot_cost / tot_qty if tot_qty > 0 else market_price
        tot_comm = sum(f.commission for f in fills)
        tot_slip = sum(f.slippage for f in fills)

        order.filled_quantity = round(tot_qty, 4)
        order.avg_fill_price = round(avg_px, 4)
        order.status = (
            OrderStatus.FILLED if tot_qty >= order.quantity else OrderStatus.PARTIALLY_FILLED
        )

        # 4. Synchronize Portfolio Balances & Positions
        self._sync_portfolio(
            portfolio=portfolio,
            symbol=order.symbol,
            side=order.side,
            quantity=tot_qty,
            fill_price=avg_px,
            commission=tot_comm,
            asset_metadata=asset_metadata,
        )

        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=round(max(0.0, order.quantity - tot_qty), 4),
            avg_fill_price=order.avg_fill_price,
            total_commission=tot_comm,
            total_slippage=tot_slip,
            fills=fills,
            rejection_reason=None if approved else reason,
        )

    def _sync_portfolio(
        self,
        portfolio: Portfolio,
        symbol: str,
        side: OrderSide,
        quantity: float,
        fill_price: float,
        commission: float,
        asset_metadata: Optional[Asset] = None,
    ) -> None:
        """Apply executed fill to portfolio cash and open position inventory."""
        notional = quantity * fill_price
        target_pos = next((p for p in portfolio.positions if p.asset.symbol == symbol), None)

        if side == OrderSide.BUY:
            portfolio.cash = max(0.0, portfolio.cash - notional - commission)
            if target_pos is not None and target_pos.side == PositionType.LONG:
                # Increase existing long
                new_qty = target_pos.quantity + quantity
                new_entry = (
                    (target_pos.quantity * target_pos.entry_price + notional) / new_qty
                    if new_qty > 0
                    else fill_price
                )
                target_pos.quantity = round(new_qty, 4)
                target_pos.entry_price = round(new_entry, 4)
                target_pos.current_price = fill_price
            elif target_pos is not None and target_pos.side == PositionType.SHORT:
                # Cover existing short
                if quantity >= target_pos.quantity:
                    rem = quantity - target_pos.quantity
                    portfolio.positions.remove(target_pos)
                    if rem > 0:
                        # Flip to long
                        new_asset = target_pos.asset
                        new_p = Position(
                            asset=new_asset,
                            quantity=round(rem, 4),
                            entry_price=fill_price,
                            current_price=fill_price,
                            side=PositionType.LONG,
                        )
                        portfolio.positions.append(new_p)
                else:
                    target_pos.quantity = round(target_pos.quantity - quantity, 4)
                    target_pos.current_price = fill_price
            else:
                # Open new long position
                asset = asset_metadata or Asset(
                    symbol=symbol,
                    name=symbol,
                    asset_class=AssetClass.EQUITY,
                    current_price=fill_price,
                    volatility_annual=0.20,
                    beta_to_market=1.0,
                )
                asset.current_price = fill_price
                new_p = Position(
                    asset=asset,
                    quantity=round(quantity, 4),
                    entry_price=fill_price,
                    current_price=fill_price,
                    side=PositionType.LONG,
                )
                portfolio.positions.append(new_p)

        elif side == OrderSide.SELL:
            portfolio.cash = max(0.0, portfolio.cash + notional - commission)
            if target_pos is not None and target_pos.side == PositionType.LONG:
                if quantity >= target_pos.quantity:
                    rem = quantity - target_pos.quantity
                    portfolio.positions.remove(target_pos)
                    if rem > 0:
                        # Flip to short
                        new_p = Position(
                            asset=target_pos.asset,
                            quantity=round(rem, 4),
                            entry_price=fill_price,
                            current_price=fill_price,
                            side=PositionType.SHORT,
                        )
                        portfolio.positions.append(new_p)
                else:
                    target_pos.quantity = round(target_pos.quantity - quantity, 4)
                    target_pos.current_price = fill_price
            elif target_pos is not None and target_pos.side == PositionType.SHORT:
                # Increase existing short
                new_qty = target_pos.quantity + quantity
                new_entry = (
                    (target_pos.quantity * target_pos.entry_price + notional) / new_qty
                    if new_qty > 0
                    else fill_price
                )
                target_pos.quantity = round(new_qty, 4)
                target_pos.entry_price = round(new_entry, 4)
                target_pos.current_price = fill_price
            else:
                # Open new short position
                asset = asset_metadata or Asset(
                    symbol=symbol,
                    name=symbol,
                    asset_class=AssetClass.EQUITY,
                    current_price=fill_price,
                    volatility_annual=0.20,
                    beta_to_market=1.0,
                )
                asset.current_price = fill_price
                new_p = Position(
                    asset=asset,
                    quantity=round(quantity, 4),
                    entry_price=fill_price,
                    current_price=fill_price,
                    side=PositionType.SHORT,
                )
                portfolio.positions.append(new_p)
