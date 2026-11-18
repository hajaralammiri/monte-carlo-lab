"""Run every experiment, print the results and save the figures.

Usage:
    python run.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from montecarlo.options import black_scholes_call, monte_carlo_call, simulate_paths
from montecarlo.pi import estimate_pi, running_estimate

FIG = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREY = "#2f5d8a", "#d9822b", "#888888"

# Option used in every experiment: stock at 100, strike 100, 1 year,
# interest rate 3%, volatility 20%.
S0, K, R, SIGMA, T = 100.0, 100.0, 0.03, 0.20, 1.0


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def pi_experiment(rng):
    print("1) Estimating pi")
    for n in (1_000, 100_000, 10_000_000):
        print(f"   n = {n:>10,}:  {estimate_pi(n, rng)}")

    n = 100_000
    est = running_estimate(n, rng)
    k = np.arange(1, n + 1)
    band = 1.96 * np.sqrt(np.pi * (4 - np.pi)) / np.sqrt(k)  # sigma of 4*Bernoulli(pi/4)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, est, color=BLUE, lw=1, label="Monte Carlo estimate")
    ax.fill_between(k, np.pi - band, np.pi + band, color=GREY, alpha=0.25, label="95% band: ±1.96·σ/√n")
    ax.axhline(np.pi, color=ORANGE, lw=1.2, label="π")
    ax.set_xscale("log")
    ax.set_ylim(2.6, 3.7)
    ax.set_xlabel("number of random points (log scale)")
    ax.set_ylabel("estimate")
    ax.set_title("Estimating π: the error shrinks like 1/√n")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "pi_convergence.png", dpi=130)
    plt.close(fig)


def paths_figure(rng):
    paths = simulate_paths(S0, R, SIGMA, T, steps=252, n_paths=30, rng=rng)
    t = np.linspace(0, T, paths.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, paths.T, lw=0.8, alpha=0.7, color=BLUE)
    ax.axhline(K, color=ORANGE, lw=1.2, ls="--", label=f"strike K = {K:.0f}")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("stock price")
    ax.set_title("30 simulated stock price paths (geometric Brownian motion)")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "gbm_paths.png", dpi=130)
    plt.close(fig)


def option_experiment(rng):
    exact = black_scholes_call(S0, K, R, SIGMA, T)
    print(f"\n2) Pricing a European call (S0={S0:.0f}, K={K:.0f}, r={R}, sigma={SIGMA}, T={T})")
    print(f"   Black-Scholes exact price: {exact:.5f}")
    for n in (1_000, 100_000, 1_000_000):
        plain = monte_carlo_call(S0, K, R, SIGMA, T, n, rng)
        anti = monte_carlo_call(S0, K, R, SIGMA, T, n, rng, antithetic=True)
        print(f"   n = {n:>9,}  plain:      {plain}")
        print(f"   {'':13s}antithetic: {anti}")

    # error vs n, averaged over repeated runs
    ns = np.unique(np.logspace(2, 6, 13).astype(int))
    reps = 30
    err_plain, err_anti = [], []
    for n in ns:
        e1 = [abs(monte_carlo_call(S0, K, R, SIGMA, T, n, rng).value - exact) for _ in range(reps)]
        e2 = [abs(monte_carlo_call(S0, K, R, SIGMA, T, n, rng, antithetic=True).value - exact) for _ in range(reps)]
        err_plain.append(np.mean(e1))
        err_anti.append(np.mean(e2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ns, err_plain, "o-", color=BLUE, label="plain Monte Carlo")
    ax.loglog(ns, err_anti, "o-", color=ORANGE, label="antithetic variates")
    ref = err_plain[0] * np.sqrt(ns[0] / ns)
    ax.loglog(ns, ref, "--", color=GREY, label="slope −1/2 (1/√n)")
    ax.set_xlabel("number of simulations n")
    ax.set_ylabel("average absolute error")
    ax.set_title("Option price error vs number of simulations")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "option_convergence.png", dpi=130)
    plt.close(fig)
    ratio = np.mean(np.array(err_plain) / np.array(err_anti))
    print(f"   Antithetic variates cut the error by about {ratio:.1f}x for the same n")


def main():
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)  # fixed seed: results are reproducible
    pi_experiment(rng)
    paths_figure(rng)
    option_experiment(rng)


if __name__ == "__main__":
    main()
