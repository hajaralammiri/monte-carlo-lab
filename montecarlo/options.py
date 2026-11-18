"""Price a European call option by Monte Carlo, and check it against Black-Scholes.

Model (Black-Scholes): the stock price follows a geometric Brownian motion

    S_T = S_0 * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)

A call option pays max(S_T - K, 0) at time T. Its fair price is the
discounted expected payoff under the risk-neutral measure:

    price = exp(-r T) * E[max(S_T - K, 0)]

Monte Carlo estimates this expectation by simulating many values of Z.
Black-Scholes gives the exact answer, so we can measure the Monte Carlo error.
"""

from __future__ import annotations

import math

import numpy as np

from .core import Estimate, estimate


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form Black-Scholes price of a European call."""
    d1 = (math.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)


def simulate_terminal_prices(S0, r, sigma, T, n, rng=None, antithetic=False):
    """Simulate n values of S_T. With antithetic=True, use Z and -Z in pairs."""
    rng = rng or np.random.default_rng()
    if antithetic:
        z = rng.standard_normal(n // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n)
    return S0 * np.exp((r - sigma ** 2 / 2) * T + sigma * math.sqrt(T) * z)


def monte_carlo_call(S0, K, r, sigma, T, n, rng=None, antithetic=False) -> Estimate:
    """Monte Carlo price of a European call (optionally with antithetic variates)."""
    rng = rng or np.random.default_rng()
    if antithetic:
        # average each pair (Z, -Z) first so the samples stay independent
        z = rng.standard_normal(n // 2)
        drift, vol = (r - sigma ** 2 / 2) * T, sigma * math.sqrt(T)
        p1 = np.maximum(S0 * np.exp(drift + vol * z) - K, 0)
        p2 = np.maximum(S0 * np.exp(drift - vol * z) - K, 0)
        payoffs = (p1 + p2) / 2
        est = estimate(math.exp(-r * T) * payoffs)
        est.n = 2 * len(z)  # number of simulated prices
        return est
    else:
        ST = simulate_terminal_prices(S0, r, sigma, T, n, rng)
        payoffs = np.maximum(ST - K, 0)
    return estimate(math.exp(-r * T) * payoffs)


def simulate_paths(S0, r, sigma, T, steps, n_paths, rng=None) -> np.ndarray:
    """Full price paths, shape (n_paths, steps + 1), for plotting."""
    rng = rng or np.random.default_rng()
    dt = T / steps
    z = rng.standard_normal((n_paths, steps))
    increments = (r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)
