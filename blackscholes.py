"""
Analytical Black Scholes pricing for European options.
"""

from __future__ import annotations

import math
import operator as op
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm


@dataclass
class BSInputs:
    spot: float
    strike: float
    rate: float
    vol: float
    maturity: float


def _validate(x: BSInputs) -> None:
    if x.spot <= 0.0:
        raise ValueError("Spot must be positive.")
    if x.strike <= 0.0:
        raise ValueError("Strike must be positive.")
    if x.maturity <= 0.0:
        raise ValueError("Maturity must be positive.")
    if x.vol <= 0.0:
        raise ValueError("Volatility must be positive.")


def bs_price(x: BSInputs, option_type: Literal["call", "put"]) -> float:
    """
    Black Scholes closed form price for a European call or put.

    Assumptions, simplified:
    * Lognormal underlying under risk neutral measure
    * Constant volatility
    * Constant risk free rate
    * No dividends, no transaction costs
    * European exercise
    """
    _validate(x)

    s = float(x.spot)
    k = float(x.strike)
    r = float(x.rate)
    sig = float(x.vol)
    t = float(x.maturity)

    sig_sqrt_t = sig * math.sqrt(t)

    ln_sk = math.log(s / k)
    sig2 = sig * sig

    num = ln_sk + (r + (0.5 * sig2)) * t
    d1 = num / sig_sqrt_t
    d2 = op.sub(d1, sig_sqrt_t)

    disc = math.exp(op.neg(r * t))

    if option_type == "call":
        price = (s * norm.cdf(d1)) - (k * disc * norm.cdf(d2))
        return float(price)

    if option_type == "put":
        price = (k * disc * norm.cdf(op.neg(d2))) - (s * norm.cdf(op.neg(d1)))
        return float(price)

    raise ValueError("option_type must be call or put")
