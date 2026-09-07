"""Real-time risk surveillance, circuit breaker gating, and automated emergency kill-switch."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional

from fintech_algorithmic_trading.execution.oms import OrderManagementSystem
from fintech_algorithmic_trading.types import (
    AlertSeverity,
    CircuitBreakerState,
    OrderSide,
    OrderType,
    Portfolio,
    PositionType,
    RiskAlert,
    RiskLimits,
    SurveillanceReport,
)


class RiskSurveillanceMonitor:
    """Real-time portfolio surveillance monitor and automated circuit breaker controller."""

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        caution_drawdown_pct: float = 0.08,  # 8% drawdown -> CAUTION
        halt_drawdown_pct: float = 0.15,  # 15% drawdown -> HALT & KILL-SWITCH
        margin_warning_pct: float = 0.80,  # 80% margin utilization
        margin_critical_pct: float = 0.95,  # 95% margin utilization
    ) -> None:
        self.limits = risk_limits or RiskLimits()
        self.caution_drawdown_pct = caution_drawdown_pct
        self.halt_drawdown_pct = halt_drawdown_pct
        self.margin_warning_pct = margin_warning_pct
        self.margin_critical_pct = margin_critical_pct

        self.state: CircuitBreakerState = CircuitBreakerState.NORMAL
        self.alerts: List[RiskAlert] = []
        self.alert_listeners: List[Callable[[RiskAlert], None]] = []
        self._peak_nav_records: Dict[str, float] = {}

    def register_alert_listener(self, listener: Callable[[RiskAlert], None]) -> None:
        """Add listener callback for emitted risk alerts."""
        self.alert_listeners.append(listener)

    def _emit_alert(self, alert: RiskAlert) -> None:
        """Record and broadcast alert to subscribers."""
        self.alerts.append(alert)
        for listener in self.alert_listeners:
            listener(alert)

    def audit_portfolio(self, portfolio: Portfolio) -> SurveillanceReport:
        """Execute real-time surveillance checks and update circuit breaker state."""
        nav = portfolio.net_asset_value
        peak = max(self._peak_nav_records.get(portfolio.id, nav), nav)
        self._peak_nav_records[portfolio.id] = peak

        drawdown = (peak - nav) / peak if peak > 0 else 0.0
        leverage = portfolio.leverage
        gross = portfolio.gross_exposure

        # Estimate margin utilization: (gross / leverage_limit) / nav
        margin_used = (gross / self.limits.max_leverage) / nav if nav > 0 else 1.0
        margin_used = min(1.0, max(0.0, margin_used))

        new_alerts: List[RiskAlert] = []
        remedial_actions: List[str] = []
        kill_switch = False
        target_state = CircuitBreakerState.NORMAL

        # 1. Drawdown Threshold Checks
        if drawdown >= self.halt_drawdown_pct:
            target_state = CircuitBreakerState.HALTED
            kill_switch = True
            alert = RiskAlert(
                severity=AlertSeverity.EMERGENCY_KILL_SWITCH,
                component="SURVEILLANCE",
                rule="MAX_DRAWDOWN_CIRCUIT_BREAKER",
                message=f"Portfolio drawdown of {drawdown * 100:.2f}% exceeds halt ceiling {self.halt_drawdown_pct * 100:.1f}%. Immediate trading halt!",
                current_value=round(drawdown, 4),
                threshold_value=self.halt_drawdown_pct,
                action_taken="TRADING_HALTED_EMERGENCY_KILL_SWITCH",
            )
            new_alerts.append(alert)
            self._emit_alert(alert)
            remedial_actions.append("Flatten or de-risk high-volatility positions immediately.")

        elif drawdown >= self.caution_drawdown_pct:
            target_state = CircuitBreakerState.CAUTION
            alert = RiskAlert(
                severity=AlertSeverity.WARNING,
                component="SURVEILLANCE",
                rule="CAUTION_DRAWDOWN_WARNING",
                message=f"Portfolio drawdown of {drawdown * 100:.2f}% triggered CAUTION state.",
                current_value=round(drawdown, 4),
                threshold_value=self.caution_drawdown_pct,
                action_taken="REDUCE_POSITION_SIZES",
            )
            new_alerts.append(alert)
            self._emit_alert(alert)
            remedial_actions.append("Downscale sizing by 50% and disallow leverage increases.")

        # 2. Leverage Checks
        if leverage > self.limits.max_leverage:
            sev = (
                AlertSeverity.CRITICAL
                if target_state != CircuitBreakerState.HALTED
                else AlertSeverity.EMERGENCY_KILL_SWITCH
            )
            alert = RiskAlert(
                severity=sev,
                component="SURVEILLANCE",
                rule="LEVERAGE_CEILING_BREACH",
                message=f"Gross leverage {leverage:.2f}x breaches policy limit {self.limits.max_leverage:.2f}x.",
                current_value=round(leverage, 2),
                threshold_value=self.limits.max_leverage,
                action_taken="MARGIN_REDUCTION_REQUIRED",
            )
            new_alerts.append(alert)
            self._emit_alert(alert)
            remedial_actions.append(
                f"Trim positions to bring leverage under {self.limits.max_leverage:.2f}x."
            )

        # 3. Margin Utilization Checks
        if margin_used >= self.margin_critical_pct:
            alert = RiskAlert(
                severity=AlertSeverity.CRITICAL,
                component="SURVEILLANCE",
                rule="MARGIN_CALL_IMMINENT",
                message=f"Margin utilization at {margin_used * 100:.1f}% approaches liquidation trigger ({self.margin_critical_pct * 100:.1f}%).",
                current_value=round(margin_used, 4),
                threshold_value=self.margin_critical_pct,
                action_taken="MARGIN_CALL_WARNING",
            )
            new_alerts.append(alert)
            self._emit_alert(alert)
            remedial_actions.append("Deposit margin cash or liquidate long/short inventory.")

        # 4. Single Position Concentration
        for pos in portfolio.positions:
            conc = pos.market_value / nav if nav > 0 else 0.0
            if conc > self.limits.max_single_position_pct:
                alert = RiskAlert(
                    severity=AlertSeverity.WARNING,
                    component="SURVEILLANCE",
                    rule="CONCENTRATION_LIMIT",
                    message=f"Position {pos.asset.symbol} concentration of {conc * 100:.1f}% exceeds {self.limits.max_single_position_pct * 100:.1f}% limit.",
                    current_value=round(conc, 4),
                    threshold_value=self.limits.max_single_position_pct,
                    action_taken="CONCENTRATION_ALERT",
                )
                new_alerts.append(alert)
                self._emit_alert(alert)
                remedial_actions.append(
                    f"Trim {pos.asset.symbol} position to under {self.limits.max_single_position_pct * 100:.1f}%."
                )

        self.state = target_state

        return SurveillanceReport(
            timestamp=datetime.utcnow(),
            portfolio_id=portfolio.id,
            state=self.state,
            active_alerts=new_alerts,
            current_drawdown_pct=round(drawdown, 4),
            margin_utilization_pct=round(margin_used, 4),
            leverage=round(leverage, 2),
            kill_switch_active=kill_switch,
            remedial_instructions=remedial_actions,
        )

    def trigger_emergency_kill_switch(
        self,
        portfolio: Portfolio,
        oms: OrderManagementSystem,
        reason: str = "Manual kill-switch activation",
    ) -> List[str]:
        """Execute automated emergency position liquidation to restore cash solvency."""
        self.state = CircuitBreakerState.HALTED
        actions_taken = []

        # Sort positions by market value descending to liquidate largest risk first
        sorted_pos = sorted(portfolio.positions, key=lambda p: p.market_value, reverse=True)

        for pos in sorted_pos:
            side = OrderSide.SELL if pos.side == PositionType.LONG else OrderSide.BUY
            order = oms.create_order(
                symbol=pos.asset.symbol,
                side=side,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
            )
            rep = oms.execute_order(
                order=order,
                portfolio=portfolio,
                market_price=pos.current_price,
            )
            act_msg = f"Liquidated {pos.side.value} {pos.quantity} {pos.asset.symbol} @ {rep.avg_fill_price:.2f}"
            actions_taken.append(act_msg)

        alert = RiskAlert(
            severity=AlertSeverity.EMERGENCY_KILL_SWITCH,
            component="KILL_SWITCH",
            rule="EMERGENCY_DELEVERAGING",
            message=f"Kill-switch executed: {reason}. Closed {len(actions_taken)} positions.",
            current_value=0.0,
            threshold_value=0.0,
            action_taken="PORTFOLIO_FLATTENED",
        )
        self._emit_alert(alert)

        return actions_taken
