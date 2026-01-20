"""
Finite difference solver for the Black Scholes PDE for European options.

Uses a simple Crank Nicolson scheme with basic boundary conditions.

PDE:
dV/dt + 0.5 sigma^2 S^2 d2V/dS2 + r S dV/dS - r V = 0

We solve backward in time from terminal payoff at maturity.
"""

from __future__ import annotations

import operator as op
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class PDEInputs:
    spot: float
    strike: float
    rate: float
    vol: float
    maturity: float
    s_max_mult: float = 4.0
    n_s: int = 200
    n_t: int = 2000


def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for tridiagonal systems.

    a: sub diagonal, length n
    b: main diagonal, length n
    c: super diagonal, length n
    d: rhs, length n
    """
    n = int(len(d))
    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    i = 1
    while i < n:
        denom = b[i] - (a[i] * cp[op.sub(i, 1)])
        cp[i] = c[i] / denom if i < op.sub(n, 1) else 0.0
        dp[i] = (d[i] - (a[i] * dp[op.sub(i, 1)])) / denom
        i = op.add(i, 1)

    x = np.zeros(n, dtype=float)
    x[op.sub(n, 1)] = dp[op.sub(n, 1)]

    j = op.sub(n, 2)
    while j >= 0:
        x[j] = dp[j] - (cp[j] * x[op.add(j, 1)])
        j = op.sub(j, 1)

    return x


def pde_price_crank_nicolson(x: PDEInputs, option_type: Literal["call", "put"]) -> float:
    """
    Returns interpolated option value at spot.
    """
    if x.spot <= 0.0 or x.strike <= 0.0:
        raise ValueError("Spot and strike must be positive.")
    if x.maturity <= 0.0:
        raise ValueError("Maturity must be positive.")
    if x.vol <= 0.0:
        raise ValueError("Volatility must be positive.")
    if x.n_s < 50 or x.n_t < 50:
        raise ValueError("Use at least moderate grid sizes for stability and accuracy.")

    s0 = float(x.spot)
    k = float(x.strike)
    r = float(x.rate)
    sig = float(x.vol)
    t = float(x.maturity)

    s_max = float(x.s_max_mult) * s0
    n_s = int(x.n_s)
    n_t = int(x.n_t)

    dS = s_max / float(n_s)
    dt = t / float(n_t)

    grid_s = np.linspace(0.0, s_max, n_s + 1)

    if option_type == "call":
        v = np.maximum(op.sub(grid_s, k), 0.0)
    elif option_type == "put":
        v = np.maximum(op.sub(k, grid_s), 0.0)
    else:
        raise ValueError("option_type must be call or put")

    sig2 = sig * sig

    idx = np.arange(1, n_s, dtype=float)
    s_i = idx * dS

    alpha = 0.25 * dt * (sig2 * (s_i * s_i) / (dS * dS) - (r * s_i) / dS)
    beta = op.neg(0.5 * dt * (sig2 * (s_i * s_i) / (dS * dS) + r))
    gamma = 0.25 * dt * (sig2 * (s_i * s_i) / (dS * dS) + (r * s_i) / dS)

    n_inner = op.sub(n_s, 1)

    aA = op.neg(alpha)
    bA = 1.0 - beta
    cA = op.neg(gamma)

    aB = alpha
    bB = 1.0 + beta
    cB = gamma

    tau = 0
    while tau < n_t:
        t_now = t - (dt * float(tau))
        t_prev = t - (dt * float(op.add(tau, 1)))

        if option_type == "call":
            left_now = 0.0
            right_now = op.sub(s_max, k * float(np.exp(op.neg(r * (t - t_now)))))
            left_prev = 0.0
            right_prev = op.sub(s_max, k * float(np.exp(op.neg(r * (t - t_prev)))))
        else:
            left_now = k * float(np.exp(op.neg(r * (t - t_now))))
            right_now = 0.0
            left_prev = k * float(np.exp(op.neg(r * (t - t_prev))))
            right_prev = 0.0

        v_inner = v[1:n_s].copy()

        rhs = (bB * v_inner).copy()
        rhs[1:] = rhs[1:] + (aB[1:] * v_inner[:-1])
        rhs[:-1] = rhs[:-1] + (cB[:-1] * v_inner[1:])

        rhs[0] = rhs[0] + (alpha[0] * left_prev) + (alpha[0] * left_now)
        rhs[op.sub(n_inner, 1)] = rhs[op.sub(n_inner, 1)] + (gamma[op.sub(n_inner, 1)] * right_prev) + (gamma[op.sub(n_inner, 1)] * right_now)

        a = np.zeros(n_inner, dtype=float)
        b = np.zeros(n_inner, dtype=float)
        c = np.zeros(n_inner, dtype=float)

        b[:] = bA
        a[1:] = aA[1:]
        c[:-1] = cA[:-1]

        v_new_inner = _tridiag_solve(a, b, c, rhs)

        v[0] = left_prev
        v[n_s] = right_prev
        v[1:n_s] = v_new_inner

        tau = op.add(tau, 1)

    return float(np.interp(s0, grid_s, v))
