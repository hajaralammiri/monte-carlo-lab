import math

import numpy as np
import pytest

from montecarlo.options import black_scholes_call, monte_carlo_call, normal_cdf
from montecarlo.pi import estimate_pi


def test_normal_cdf():
    assert normal_cdf(0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-4)


def test_black_scholes_known_value():
    # textbook example: S0=100, K=100, r=5%, sigma=20%, T=1  ->  10.4506
    assert black_scholes_call(100, 100, 0.05, 0.2, 1) == pytest.approx(10.4506, abs=1e-4)


def test_call_price_bounds():
    # a call is worth at least S0 - K e^{-rT} and at most S0
    c = black_scholes_call(100, 90, 0.03, 0.25, 2)
    assert 100 - 90 * math.exp(-0.06) <= c <= 100


def test_pi_estimate_in_confidence_interval():
    est = estimate_pi(1_000_000, np.random.default_rng(0))
    lo, hi = est.ci95
    assert lo - 1e-3 < math.pi < hi + 1e-3


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_matches_black_scholes(antithetic):
    exact = black_scholes_call(100, 100, 0.03, 0.2, 1)
    est = monte_carlo_call(100, 100, 0.03, 0.2, 1, 1_000_000, np.random.default_rng(1), antithetic)
    assert abs(est.value - exact) < 4 * est.std_error


def test_antithetic_reduces_variance():
    rng = np.random.default_rng(2)
    plain = monte_carlo_call(100, 100, 0.03, 0.2, 1, 200_000, rng)
    anti = monte_carlo_call(100, 100, 0.03, 0.2, 1, 200_000, rng, antithetic=True)
    assert anti.std_error < plain.std_error
