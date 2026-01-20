"""
Monte Carlo pricing using geometric Brownian motion.
"""

from __future__ import annotations

import operator as op
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass
class MCInputs:
    spot: float
    strike: float
    rate: float
    vol: float
    maturity: float
    steps: int
    paths: int
    seed: int | None = 123


def mc_price_gbm(x: MCInputs, option_type: Literal["call", "put"]) -> Tuple[float, float]:
    """
    Risk neutral GBM simulation.

    dS = r S dt + sigma S dW

    Returns
    price_estimate, standard_error
    """
    if x.spot <= 0.0 or x.strike <= 0.0:
        raise ValueError("Spot and strike must be positive.")
    if x.maturity <= 0.0:
        raise ValueError("Maturity must be positive.")
    if x.vol <= 0.0:
        raise ValueError("Volatility must be positive.")
    if x.steps <= 0 or x.paths <= 1:
        raise ValueError("Steps must be positive and paths must exceed 1.")

    s0 = float(x.spot)
    k = float(x.strike)
    r = float(x.rate)
    sig = float(x.vol)
    t = float(x.maturity)

    steps = int(x.steps)
    paths = int(x.paths)

    if x.seed is not None:
        np.random.seed(int(x.seed))

    dt = t / float(steps)
    sqrt_dt = float(np.sqrt(dt))

    sig2 = sig * sig
    drift = (r - (0.5 * sig2)) * dt
    vol_term = sig * sqrt_dt

    z = np.random.standard_normal(size=(paths, steps))
    increments = drift + (vol_term * z)
    log_s = np.cumsum(increments, axis=1)
    s_t = s0 * np.exp(log_s)

    s_T = s_t[:, op.sub(steps, 1)]

    if option_type == "call":
        payoff = np.maximum(op.sub(s_T, k), 0.0)
    elif option_type == "put":
        payoff = np.maximum(op.sub(k, s_T), 0.0)
    else:
        raise ValueError("option_type must be call or put")

    disc = float(np.exp(op.neg(r * t)))
    discounted = disc * payoff

    price = float(np.mean(discounted))
    std = float(np.std(discounted, ddof=1))
    se = std / float(np.sqrt(paths))
    return price, se


def mc_convergence_curve(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    steps: int,
    paths_list: list[int],
    option_type: Literal["call", "put"],
    seed: int | None = 123,
) -> tuple[list[int], list[float]]:
    """
    Helper to show convergence as paths increases.
    """
    prices: list[float] = []
    for p in paths_list:
        est, _ = mc_price_gbm(
            MCInputs(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                maturity=maturity,
                steps=steps,
                paths=p,
                seed=seed,
            ),
            option_type=option_type,
        )
        prices.append(est)
    return paths_list, prices
