"""Monte Carlo estimation with confidence intervals.

The idea behind every Monte Carlo method in this project:

    E[X] ≈ (X_1 + ... + X_n) / n

By the law of large numbers the average converges to E[X], and by the
central limit theorem the error is about 1.96 * sigma / sqrt(n) with
95% confidence. So to get one more correct digit, you need 100x more samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    value: float
    std_error: float
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        """95% confidence interval from the central limit theorem."""
        return self.value - 1.96 * self.std_error, self.value + 1.96 * self.std_error

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"{self.value:.5f}  (95% CI [{lo:.5f}, {hi:.5f}], n={self.n:,})"


def estimate(samples: np.ndarray) -> Estimate:
    """Sample mean and its standard error sigma / sqrt(n)."""
    n = len(samples)
    return Estimate(float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(n)), n)
