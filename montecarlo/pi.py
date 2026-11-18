"""Estimate pi by throwing random points in a square.

A point (x, y) uniform in [0, 1]^2 falls inside the quarter disc
x^2 + y^2 <= 1 with probability pi / 4. So pi = 4 * P(inside), and
4 * (fraction of points inside) is an unbiased estimator of pi.
"""

from __future__ import annotations

import numpy as np

from .core import Estimate, estimate


def estimate_pi(n: int, rng: np.random.Generator | None = None) -> Estimate:
    rng = rng or np.random.default_rng()
    x, y = rng.random(n), rng.random(n)
    inside = (x * x + y * y <= 1.0).astype(float)
    return estimate(4.0 * inside)


def running_estimate(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Estimate of pi after 1, 2, ..., n points (for convergence plots)."""
    rng = rng or np.random.default_rng()
    x, y = rng.random(n), rng.random(n)
    inside = (x * x + y * y <= 1.0)
    return 4.0 * np.cumsum(inside) / np.arange(1, n + 1)
