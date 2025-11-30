"""
Analyze alternative benchmark definitions and market conditions
that could lead to significantly better performance.

Key question: What if we change the rules of the game?
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.gbm import generate_gbm_paths

np.random.seed(42)

n_sims = 10000
n_days = 125
S0 = 100
total_usd = 1_000_000_000
min_dur = 75
max_dur = 125
target_dur = 100


def execute_with_benchmark(prices_2d, total_usd, max_duration, benchmark_type='expanding'):
    """
    Execute with different benchmark definitions.

    benchmark_type options:
    - 'expanding': Standard expanding mean (what we use now)
    - 'fixed_day1': Benchmark is Day 1 price only
    - 'twap': Time-weighted average (known upfront via simulation)
    - 'vwap_uniform': What VWAP would be with uniform execution
    """
    n_paths, n_days_total = prices_2d.shape
    n_exec_days = min(n_days_total, max_duration)

    # Pre-calculate benchmarks based on type
    if benchmark_type == 'expanding':
        # Current approach: benchmark at day d is mean of prices[0:d+1]
        benchmarks = np.cumsum(prices_2d[:, :n_exec_days], axis=1) / np.arange(1, n_exec_days + 1)
    elif benchmark_type == 'fixed_day1':
        # Benchmark is always Day 1 price
        benchmarks = np.tile(prices_2d[:, 0:1], (1, n_exec_days))
    elif benchmark_type == 'twap':
        # TWAP over execution period (we "know" all prices - cheating but instructive)
        twap = np.mean(prices_2d[:, :n_exec_days], axis=1, keepdims=True)
        benchmarks = np.tile(twap, (1, n_exec_days))
    elif benchmark_type == 'vwap_uniform':
        # What would VWAP be with perfectly uniform execution
        uniform_vwap = np.mean(prices_2d[:, :target_dur], axis=1, keepdims=True)
        benchmarks = np.tile(uniform_vwap, (1, n_exec_days))

    # Execute using Strategy 2 logic but with custom benchmark
    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    for day in range(n_exec_days):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day
        base_daily = remaining / np.maximum(days_remaining, 1)

        deviation = (benchmark - price) / benchmark
        multiplier = np.exp(50 * deviation)

        time_pct = day / max_duration
        if time_pct > 0.6:
            urgency = 1 + ((time_pct - 0.6) / 0.4) ** 2
            multiplier *= urgency

        multiplier = np.clip(multiplier, 0.1, 8.0)

        usd_today = np.minimum(base_daily * multiplier, remaining)
        shares_today = usd_today / price

        remaining_usd[active] -= usd_today
        total_shares[active] += shares_today

        newly_completed = remaining_usd < 1.0
        durations[newly_completed & ~completed] = day + 1
        completed = newly_completed

    # Force completion
    still_active = ~completed
    if np.any(still_active):
        final_day = n_exec_days - 1
        price = prices_2d[still_active, final_day]
        total_shares[still_active] += remaining_usd[still_active] / price
        durations[still_active] = final_day + 1

    # Calculate final performance using SAME benchmark definition
    vwap = total_usd / total_shares

    if benchmark_type == 'expanding':
        final_benchmarks = np.array([
            np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
        ])
    elif benchmark_type == 'fixed_day1':
        final_benchmarks = prices_2d[:, 0]
    elif benchmark_type == 'twap':
        final_benchmarks = np.mean(prices_2d[:, :n_exec_days], axis=1)
    elif benchmark_type == 'vwap_uniform':
        final_benchmarks = np.mean(prices_2d[:, :target_dur], axis=1)

    performance = (final_benchmarks - vwap) / final_benchmarks * 10000

    return performance, durations, vwap, final_benchmarks


print("=" * 70)
print("BENCHMARK ALTERNATIVE ANALYSIS")
print("=" * 70)
print()

# Test with different volatilities
for sigma in [0.15, 0.25, 0.35, 0.50]:
    print(f"\nVolatility: {sigma*100:.0f}%")
    print("-" * 50)

    prices = generate_gbm_paths(S0, 0.0, sigma, n_days, n_sims, seed=42)

    for bench_type in ['expanding', 'fixed_day1', 'twap', 'vwap_uniform']:
        perf, dur, _, _ = execute_with_benchmark(prices, total_usd, max_dur, bench_type)
        print(f"  {bench_type:15s}: {np.mean(perf):8.2f} bps (std: {np.std(perf):6.2f})")


print()
print("=" * 70)
print("DRIFT IMPACT ANALYSIS")
print("=" * 70)
print()

# Test with different drifts
for mu in [-0.30, -0.15, 0.0, 0.15, 0.30]:
    print(f"\nDrift: {mu*100:+.0f}%")
    print("-" * 50)

    prices = generate_gbm_paths(S0, mu, 0.25, n_days, n_sims, seed=42)

    for bench_type in ['expanding', 'fixed_day1']:
        perf, dur, _, _ = execute_with_benchmark(prices, total_usd, max_dur, bench_type)
        print(f"  {bench_type:15s}: {np.mean(perf):8.2f} bps (std: {np.std(perf):6.2f})")


print()
print("=" * 70)
print("THEORETICAL ANALYSIS: Why is improvement limited?")
print("=" * 70)
print("""
FUNDAMENTAL INSIGHT:

With an expanding mean benchmark and GBM prices:

1. The benchmark B(t) = mean(P[0:t]) is a FUNCTION of past prices
2. If we buy more when P < B, we're just exploiting past randomness
3. Future prices are INDEPENDENT of past (Markov property)
4. The benchmark itself moves with our execution

MATHEMATICAL LIMIT:

For any adaptive strategy vs uniform:
- E[Savings] = E[sum(shares_i * (B - P_i))]
- This is bounded by the variance of prices around the mean
- With more aggressive strategies, variance of savings increases
  but MEAN savings has diminishing returns

WHY STRATEGY 2/4 ARE NEAR-OPTIMAL:

1. They exploit deviation when P < B (positive edge)
2. They don't over-concentrate execution (diversification)
3. They complete on time (no emergency buying penalties)

TO SIGNIFICANTLY BEAT THIS, YOU NEED:

1. PREDICTIVE SIGNAL: Know future prices (impossible with pure GBM)
2. DIFFERENT BENCHMARK: Fixed benchmark allows more exploitation
3. MARKET IMPACT: Real markets have price impact (Almgren-Chriss)
4. MEAN REVERSION: Non-GBM price processes (Ornstein-Uhlenbeck)
""")

# Demonstrate with mean-reverting process
print()
print("=" * 70)
print("MEAN-REVERTING PRICES (Not GBM)")
print("=" * 70)


def generate_ou_paths(S0, theta, mu_long, sigma, n_days, n_sims, seed=None):
    """
    Generate Ornstein-Uhlenbeck (mean-reverting) price paths.

    dS = theta * (mu_long - S) * dt + sigma * dW

    theta: speed of mean reversion
    mu_long: long-term mean price
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1 / 252
    paths = np.zeros((n_sims, n_days))
    paths[:, 0] = S0

    for t in range(1, n_days):
        dW = np.random.randn(n_sims) * np.sqrt(dt)
        paths[:, t] = (paths[:, t-1] +
                       theta * (mu_long - paths[:, t-1]) * dt +
                       sigma * paths[:, t-1] * dW)
        paths[:, t] = np.maximum(paths[:, t], 1)  # Floor at 1

    return paths


print("\nMean-Reverting Process (OU) vs GBM:")
print("-" * 50)

# GBM baseline
prices_gbm = generate_gbm_paths(S0, 0.0, 0.25, n_days, n_sims, seed=42)
perf_gbm, _, _, _ = execute_with_benchmark(prices_gbm, total_usd, max_dur, 'expanding')
print(f"GBM (random walk):      {np.mean(perf_gbm):8.2f} bps")

# Mean-reverting with different reversion speeds
for theta in [1.0, 5.0, 10.0]:
    prices_ou = generate_ou_paths(S0, theta, S0, 0.25, n_days, n_sims, seed=42)
    perf_ou, _, _, _ = execute_with_benchmark(prices_ou, total_usd, max_dur, 'expanding')
    print(f"OU (theta={theta:4.1f}):         {np.mean(perf_ou):8.2f} bps")

print("""
CONCLUSION:

With mean-reverting prices, our strategy performs MUCH better because:
- When price drops below mean, it's likely to revert UP
- We buy more when cheap, knowing prices will recover
- This creates consistent positive alpha

In pure GBM, prices don't "want" to go anywhere - they're random.
Our strategies are already near the theoretical optimum for random walks.
""")
