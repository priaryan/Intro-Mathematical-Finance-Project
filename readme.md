# Black Scholes option pricing project

## Motivation
This is a personal project taken for the sole purpose of learning some basics of financial modeling by building European option price three ways and comparing results.

I attempt to connect:
* Theory:  analytical Black Scholes formula.
* Simulation: Monte Carlo under geometric Brownian motion
* Numerical methods: Finite difference solver for the Black Scholes PDE

The emphasis is introducing myself to certain concepts, understanding the implications of different models, and guaging certain tradeoffs.

## Data used
We use daily close prices for a single liquid US equity, default AAPL.

Data source:
* Fetched in Python using yfinance
* Daily close prices over roughly the last three years


## Preprocessing
From daily closes we compute daily log returns:
r_t = log(S_t / S_{t-1})

Volatility:
* Estimate daily standard deviation of log returns
* Annualize using sqrt(252)

Risk free rate:
* Use a constant rate, default 0.03

Assumptions are intentionally simplified:
* No dividends
* Constant volatility
* Constant rate
* European exercise only

## Models implemented

### 1 Analytical Black Scholes
I compute European call and put prices using the closed form formula based on d1 and d2.

I attempted to learn:
* The role of volatility and discounting
* The meaning of risk neutral pricing
* The limits of constant volatility and rate

### 2 Monte Carlo simulation
We simulate the underlying price under risk neutral geometric Brownian motion and estimate the discounted expected payoff.

I attempted to learn:
* How convergence improves as paths increase
* How random sampling introduces sampling error, summarized with a standard error estimate
* How numerical methods that I had learnt in engineering class translate to a financial model.

### 3 PDE solver
We solve the Black Scholes PDE backward in time with a Crank Nicolson finite difference scheme.

After researching time stepping methods for the Black Scholes PDE, I chose the Crank Nicolson scheme because it balances stability, accuracy, and simplicity. An explicit method like forward Euler is easy but can become unstable unless the time step is extremely small. A fully implicit method like backward Euler is stable but more diffusive and only first order in time. Crank Nicolson is second order and typically stable, while still leading to a simple tridiagonal solve each step.Higher order methods add complexity and tuning, and payoff kinks plus stiffness limit practical gains for this project. They also require extra solves and debugging time.

Why not use higher order methods with more complexity?
* Higher order methods can be more accurate per time step when the solution is smooth, so I would probably need fewer time steps for a target error.
* Black Scholes terminal payoffs have a kink at the strike, which reduces effective convergence order and can cause oscillations for higher order schemes
* After spatial discretization the system is stiff because of the diffusion term, so explicit higher order Runge Kutta is limited by stability and still needs tiny time steps
* Implicit higher order methods handle stiffness but require more work per step, often multiple linear solves or stages, which increases runtime and code complexity
* In practice, spatial discretization and boundary condition errors may dominate.


## Validation and comparison
We compare call and put prices from said models.

We compute absolute and relative errors against the analytical price.

We also include plots:
* Monte Carlo price convergence as paths increase
* Option value vs underlying price
* PDE absolute error vs grid resolution

## How to run
1 Install dependencies
   pip install -r requirements.txt

2 Run
   python main.py

Plots are saved into an outputs folder.

## Limitations
This is a basic learning project in my opinion. It does not include:
* Dividends
* Early exercise
* Stochastic volatility
* Term structures for rates or volatility
* Calibration to an option surface

Just my journey into introducing myself to mathematical finance.
