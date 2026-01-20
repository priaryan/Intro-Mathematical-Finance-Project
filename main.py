"""
Main script.

Workflow:
1 Fetch market data
2 Estimate volatility from historical log returns
3 Price European options using
   Analytical Black Scholes
   Monte Carlo simulation
   Finite difference PDE solver
4 Compare results and produce simple plots

Outputs are saved to an outputs folder.
"""

from __future__ import annotations

import os
import math
import operator as op
from datetime import date, timedelta

import numpy as np
import matplotlib.pyplot as plt

from data_loader import fetch_daily_close
from preprocessing import compute_log_returns, estimate_annualized_volatility
from black_scholes import BSInputs, bs_price
from monte_carlo import MCInputs, mc_price_gbm, mc_convergence_curve
from pde_solver import PDEInputs, pde_price_crank_nicolson


def choose_strike_from_spot(spot: float) -> float:
    """
    Simple strike choice for a demo: round spot to nearest 5.
    """
    step = 5.0
    return float(step * round(spot / step))


def ensure_outputs_dir() -> str:
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main() -> None:
    out_dir = ensure_outputs_dir()

    ticker = "AAPL"
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 3)

    series = fetch_daily_close(ticker=ticker, start_date=start_date, end_date=end_date)
    close = series.close

    log_rets = compute_log_returns(close)
    sigma = estimate_annualized_volatility(log_rets)

    spot = float(close.iloc[op.sub(len(close), 1)])
    strike = choose_strike_from_spot(spot)

    maturity_days = 30
    maturity = float(maturity_days) / 365.0

    rate = 0.03

    print("Asset", ticker)
    print("Date range", start_date, "to", end_date)
    print("Last close", round(spot, 4))
    print("Strike", round(strike, 4))
    print("Maturity years", round(maturity, 6))
    print("Risk free rate", rate)
    print("Estimated vol", round(sigma, 6))

    bs_in = BSInputs(spot=spot, strike=strike, rate=rate, vol=sigma, maturity=maturity)

    bs_call = bs_price(bs_in, "call")
    bs_put = bs_price(bs_in, "put")

    mc_in = MCInputs(
        spot=spot,
        strike=strike,
        rate=rate,
        vol=sigma,
        maturity=maturity,
        steps=100,
        paths=50000,
        seed=123,
    )
    mc_call, mc_call_se = mc_price_gbm(mc_in, "call")
    mc_put, mc_put_se = mc_price_gbm(mc_in, "put")

    pde_in = PDEInputs(
        spot=spot,
        strike=strike,
        rate=rate,
        vol=sigma,
        maturity=maturity,
        s_max_mult=4.0,
        n_s=200,
        n_t=2000,
    )
    pde_call = pde_price_crank_nicolson(pde_in, "call")
    pde_put = pde_price_crank_nicolson(pde_in, "put")

    def abs_rel_err(approx: float, ref: float) -> tuple[float, float]:
        abs_e = float(abs(approx - ref))
        rel_e = abs_e / float(ref) if ref != 0.0 else float("nan")
        return abs_e, rel_e

    call_mc_abs, call_mc_rel = abs_rel_err(mc_call, bs_call)
    call_pde_abs, call_pde_rel = abs_rel_err(pde_call, bs_call)
    put_mc_abs, put_mc_rel = abs_rel_err(mc_put, bs_put)
    put_pde_abs, put_pde_rel = abs_rel_err(pde_put, bs_put)

    print()
    print("Call prices")
    print("Analytical", round(bs_call, 6))
    print("Monte Carlo", round(mc_call, 6), "SE", round(mc_call_se, 6))
    print("PDE", round(pde_call, 6))
    print("Call Monte Carlo abs err", round(call_mc_abs, 6), "rel err", round(call_mc_rel, 6))
    print("Call PDE abs err", round(call_pde_abs, 6), "rel err", round(call_pde_rel, 6))

    print()
    print("Put prices")
    print("Analytical", round(bs_put, 6))
    print("Monte Carlo", round(mc_put, 6), "SE", round(mc_put_se, 6))
    print("PDE", round(pde_put, 6))
    print("Put Monte Carlo abs err", round(put_mc_abs, 6), "rel err", round(put_mc_rel, 6))
    print("Put PDE abs err", round(put_pde_abs, 6), "rel err", round(put_pde_rel, 6))

    paths_list = [500, 2000, 8000, 20000, 50000, 100000]
    xs, ys = mc_convergence_curve(
        spot=spot,
        strike=strike,
        rate=rate,
        vol=sigma,
        maturity=maturity,
        steps=100,
        paths_list=paths_list,
        option_type="call",
        seed=123,
    )

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.axhline(bs_call, linestyle=":")
    plt.xscale("log")
    plt.xlabel("Paths")
    plt.ylabel("Call price estimate")
    plt.title("Monte Carlo convergence for call")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_convergence_call.png"), dpi=150)
    plt.close()

    s_grid = np.linspace(0.2 * spot, 1.8 * spot, 60)
    bs_vals = []
    pde_vals = []

    for s in s_grid:
        bs_vals.append(bs_price(BSInputs(spot=float(s), strike=strike, rate=rate, vol=sigma, maturity=maturity), "call"))
        pde_vals.append(pde_price_crank_nicolson(PDEInputs(spot=float(s), strike=strike, rate=rate, vol=sigma, maturity=maturity, s_max_mult=4.0, n_s=200, n_t=2000), "call"))

    plt.figure()
    plt.plot(s_grid, bs_vals, marker="o", markersize=3)
    plt.plot(s_grid, pde_vals, marker="x", markersize=3)
    plt.xlabel("Underlying price")
    plt.ylabel("Call value")
    plt.title("Option value vs underlying price")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_vs_spot_call.png"), dpi=150)
    plt.close()

    grid_list = [80, 120, 200, 300]
    call_errs = []
    for n_s in grid_list:
        pde_val = pde_price_crank_nicolson(PDEInputs(spot=spot, strike=strike, rate=rate, vol=sigma, maturity=maturity, s_max_mult=4.0, n_s=int(n_s), n_t=2000), "call")
        call_errs.append(float(abs(pde_val - bs_call)))

    plt.figure()
    plt.plot(grid_list, call_errs, marker="o")
    plt.xlabel("Spatial grid points")
    plt.ylabel("Absolute error vs analytical")
    plt.title("PDE error vs grid resolution for call")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pde_error_vs_grid_call.png"), dpi=150)
    plt.close()

    print()
    print("Saved plots to", out_dir)


if __name__ == "__main__":
    main()
