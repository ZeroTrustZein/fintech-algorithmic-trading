"""Black-Scholes analytical option pricing and Greek risk sensitivities."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from fintech_algorithmic_trading.types import OptionGreeks, OptionType


def calculate_option_greeks(
    option_type: OptionType,
    spot_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> OptionGreeks:
    """Compute Black-Scholes analytical price and primary Greeks (Delta, Gamma, Vega, Theta, Rho).

    Args:
        option_type: OptionType.CALL or OptionType.PUT.
        spot_price: Underlying current market price S.
        strike_price: Strike price K.
        time_to_expiry_years: Time to expiration T in years.
        volatility: Annualized implied volatility sigma.
        risk_free_rate: Annual continuously compounded risk-free rate r.
        dividend_yield: Continuous dividend yield q.

    Returns:
        OptionGreeks model with exact risk sensitivities.
    """
    if spot_price <= 0:
        raise ValueError("Spot price must be positive.")
    if strike_price <= 0:
        raise ValueError("Strike price must be positive.")
    if time_to_expiry_years <= 0:
        raise ValueError("Time to expiration must be positive.")
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")

    s = float(spot_price)
    k = float(strike_price)
    t = float(time_to_expiry_years)
    v = float(volatility)
    r = float(risk_free_rate)
    q = float(dividend_yield)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v**2) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    df_r = np.exp(-r * t)
    df_q = np.exp(-q * t)
    pdf_d1 = stats.norm.pdf(d1)

    # Common Greeks: Gamma & Vega are identical for call and put
    gamma = float(df_q * pdf_d1 / (s * v * sqrt_t))
    # Vega: sensitivity per 1% change in vol (0.01)
    vega = float(s * df_q * sqrt_t * pdf_d1 * 0.01)

    if option_type == OptionType.CALL:
        nd1 = float(stats.norm.cdf(d1))
        nd2 = float(stats.norm.cdf(d2))
        price = float(s * df_q * nd1 - k * df_r * nd2)
        delta = float(df_q * nd1)
        # Theta annualized, divided by 365 for daily theta
        theta_annual = (
            -(s * df_q * pdf_d1 * v) / (2.0 * sqrt_t) - r * k * df_r * nd2 + q * s * df_q * nd1
        )
        theta_daily = float(theta_annual / 365.0)
        # Rho per 100 bps (1%) rate change
        rho = float(k * t * df_r * nd2 * 0.01)
    else:
        n_neg_d1 = float(stats.norm.cdf(-d1))
        n_neg_d2 = float(stats.norm.cdf(-d2))
        price = float(k * df_r * n_neg_d2 - s * df_q * n_neg_d1)
        delta = float(-df_q * n_neg_d1)
        theta_annual = (
            -(s * df_q * pdf_d1 * v) / (2.0 * sqrt_t)
            + r * k * df_r * n_neg_d2
            - q * s * df_q * n_neg_d1
        )
        theta_daily = float(theta_annual / 365.0)
        rho = float(-k * t * df_r * n_neg_d2 * 0.01)

    return OptionGreeks(
        option_type=option_type,
        spot_price=round(s, 2),
        strike_price=round(k, 2),
        time_to_expiry_years=round(t, 4),
        volatility=round(v, 4),
        risk_free_rate=round(r, 4),
        price=round(price, 4),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        vega=round(vega, 4),
        theta=round(theta_daily, 4),
        rho=round(rho, 4),
    )
