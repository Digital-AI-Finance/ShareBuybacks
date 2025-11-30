"""
Partial completion v2: Keep Strategy 4's core logic,
only modify the forced-completion behavior at deadline.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.gbm import generate_gbm_paths
from modules.strategies_vectorized import precompute_benchmarks, strategy_2_vectorized, strategy_4_vectorized
from modules.config import S4_BETA, S4_GAMMA

np.random.seed(42)

n_sims = 10000
n_days = 125
S0 = 100
sigma = 0.25
mu = 0.0
total_usd = 1_000_000_000
min_dur = 75
max_dur = 125
target_dur = 100

prices = generate_gbm_paths(S0, mu, sigma, n_days, n_sims, seed=42)
benchmarks = precompute_benchmarks(prices)


def strategy_4_partial_completion(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    min_completion_pct=100.0,  # Minimum completion required
    beta=S4_BETA,
    gamma=S4_GAMMA,
):
    """
    Strategy 4 with optional partial completion.

    Same as Strategy 4, but at deadline:
    - If min_completion_pct is reached AND price is unfavorable, STOP buying
    - Don't force-buy remaining at bad prices
    """
    n_paths, n_days_total = prices_2d.shape
    n_exec_days = min(n_days_total, max_duration)

    min_usd = total_usd * (min_completion_pct / 100.0)

    # Track execution
    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    cumulative_usd = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    for day in range(n_exec_days):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day
        base_daily = remaining / np.maximum(days_remaining, 1)

        # Deviation from benchmark
        deviation = (benchmark - price) / benchmark

        # Convex scaling (Strategy 4 core)
        convex_mult = np.exp(beta * deviation)

        # Time urgency
        time_pct = day / max_duration
        urgency = 1 + gamma * (time_pct ** 2)

        multiplier = convex_mult * urgency
        multiplier = np.clip(multiplier, 0.1, 8.0)

        # Calculate USD to spend
        usd_today = np.minimum(base_daily * multiplier, remaining)
        shares_today = usd_today / price

        remaining_usd[active] -= usd_today
        total_shares[active] += shares_today
        cumulative_usd[active] += usd_today

        # Check completion
        fully_done = remaining_usd < 1.0
        durations[fully_done & ~completed] = day + 1
        completed = fully_done

    # KEY DIFFERENCE: At deadline, decide whether to force-complete
    still_active = ~completed
    if np.any(still_active):
        final_day = n_exec_days - 1
        final_price = prices_2d[still_active, final_day]
        final_benchmark = benchmarks_2d[still_active, final_day]
        final_deviation = (final_benchmark - final_price) / final_benchmark

        remaining = remaining_usd[still_active]
        executed = cumulative_usd[still_active]

        # For each path: decide whether to complete
        indices = np.where(still_active)[0]

        for idx, (i, rem, exe, dev) in enumerate(zip(indices, remaining, executed, final_deviation)):
            hit_minimum = exe >= min_usd
            price_unfavorable = dev < -0.01  # Price 1% above benchmark

            if hit_minimum and price_unfavorable:
                # Don't buy more - accept partial completion
                pass
            else:
                # Force complete (buy remaining at current price)
                total_shares[i] += rem / prices_2d[i, final_day]
                cumulative_usd[i] += rem
                remaining_usd[i] = 0

            durations[i] = final_day + 1

    # Calculate metrics
    actual_executed = cumulative_usd
    vwap = actual_executed / np.maximum(total_shares, 1e-10)

    final_benchmarks = np.array([
        np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
    ])

    performance = (final_benchmarks - vwap) / final_benchmarks * 10000
    completion_pct = (actual_executed / total_usd) * 100

    return performance, durations, vwap, final_benchmarks, completion_pct


def strategy_2_partial_completion(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    min_completion_pct=100.0,
):
    """
    Strategy 2 with optional partial completion.
    """
    n_paths, n_days_total = prices_2d.shape
    n_exec_days = min(n_days_total, max_duration)

    min_usd = total_usd * (min_completion_pct / 100.0)
    base_daily = total_usd / target_duration

    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    cumulative_usd = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    initial_period = 10
    max_speedup = 5.0

    for day in range(n_exec_days):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day

        if day < initial_period:
            usd_today = np.minimum(np.full_like(price, base_daily), remaining)
        else:
            # Adaptive logic
            deviation = (benchmark - price) / benchmark

            # Speedup when favorable
            multiplier = np.ones_like(price)
            favorable = deviation > 0
            multiplier[favorable] = 1 + (max_speedup - 1) * np.minimum(deviation[favorable] * 10, 1)

            # Slowdown when unfavorable
            unfavorable = deviation <= 0
            multiplier[unfavorable] = 0.5

            # Urgency near end
            if day >= min_duration:
                urgency = remaining / np.maximum(days_remaining, 1) / base_daily
                multiplier = np.maximum(multiplier, urgency)

            usd_today = np.minimum(base_daily * multiplier, remaining)

        shares_today = usd_today / price

        remaining_usd[active] -= usd_today
        total_shares[active] += shares_today
        cumulative_usd[active] += usd_today

        fully_done = remaining_usd < 1.0
        durations[fully_done & ~completed] = day + 1
        completed = fully_done

    # Partial completion logic at deadline
    still_active = ~completed
    if np.any(still_active):
        final_day = n_exec_days - 1
        final_price = prices_2d[still_active, final_day]
        final_benchmark = benchmarks_2d[still_active, final_day]
        final_deviation = (final_benchmark - final_price) / final_benchmark

        remaining = remaining_usd[still_active]
        executed = cumulative_usd[still_active]

        indices = np.where(still_active)[0]

        for idx, (i, rem, exe, dev) in enumerate(zip(indices, remaining, executed, final_deviation)):
            hit_minimum = exe >= min_usd
            price_unfavorable = dev < -0.01

            if hit_minimum and price_unfavorable:
                pass  # Accept partial
            else:
                total_shares[i] += rem / prices_2d[i, final_day]
                cumulative_usd[i] += rem
                remaining_usd[i] = 0

            durations[i] = final_day + 1

    actual_executed = cumulative_usd
    vwap = actual_executed / np.maximum(total_shares, 1e-10)

    final_benchmarks = np.array([
        np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
    ])

    performance = (final_benchmarks - vwap) / final_benchmarks * 10000
    completion_pct = (actual_executed / total_usd) * 100

    return performance, durations, vwap, final_benchmarks, completion_pct


print("=" * 70)
print("PARTIAL COMPLETION ANALYSIS V2")
print("=" * 70)
print(f"Using Strategy 2/4 core logic, only modifying deadline behavior")
print(f"Simulations: {n_sims:,}, Volatility: {sigma*100:.0f}%")
print()

# Baseline
print("BASELINE (100% completion):")
print("-" * 70)
perf2, dur2, _, _ = strategy_2_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur)
perf4, dur4, _, _ = strategy_4_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur)
print(f"Strategy 2:          {np.mean(perf2):8.2f} bps (std: {np.std(perf2):6.2f})")
print(f"Strategy 4:          {np.mean(perf4):8.2f} bps (std: {np.std(perf4):6.2f})")
print()

# Strategy 2 with partial completion
print("STRATEGY 2 + PARTIAL COMPLETION:")
print("-" * 70)
print(f"{'Min %':<8} {'Performance':>12} {'Std':>10} {'Completion':>12} {'Improvement':>12}")
print("-" * 70)

for min_pct in [100, 99, 98, 95, 90, 85]:
    perf, dur, vwap, bench, comp = strategy_2_partial_completion(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        min_completion_pct=min_pct
    )
    improvement = np.mean(perf) - np.mean(perf2)
    print(f"{min_pct:>3}%     {np.mean(perf):>10.2f} bps {np.std(perf):>10.2f} {np.mean(comp):>10.2f}% {improvement:>+10.2f} bps")

print()

# Strategy 4 with partial completion
print("STRATEGY 4 + PARTIAL COMPLETION:")
print("-" * 70)
print(f"{'Min %':<8} {'Performance':>12} {'Std':>10} {'Completion':>12} {'Improvement':>12}")
print("-" * 70)

for min_pct in [100, 99, 98, 95, 90, 85]:
    perf, dur, vwap, bench, comp = strategy_4_partial_completion(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        min_completion_pct=min_pct
    )
    improvement = np.mean(perf) - np.mean(perf4)
    print(f"{min_pct:>3}%     {np.mean(perf):>10.2f} bps {np.std(perf):>10.2f} {np.mean(comp):>10.2f}% {improvement:>+10.2f} bps")

print()
print("=" * 70)
print("DOLLAR VALUE OF FLEXIBILITY:")
print("=" * 70)

# Best partial completion result
perf_s4_95, _, _, _, comp_s4_95 = strategy_4_partial_completion(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
    min_completion_pct=95
)

improvement_bps = np.mean(perf_s4_95) - np.mean(perf4)
dollar_savings = improvement_bps / 10000 * total_usd

print(f"\nStrategy 4 baseline: {np.mean(perf4):.2f} bps")
print(f"Strategy 4 + 95% min: {np.mean(perf_s4_95):.2f} bps")
print(f"Improvement: {improvement_bps:+.2f} bps")
print(f"\nFor ${total_usd/1e9:.1f}B program:")
print(f"Dollar savings: ${dollar_savings/1e6:+.2f}M")
print(f"Average completion: {np.mean(comp_s4_95):.2f}%")

# Analysis of when partial helps
print()
print("=" * 70)
print("WHEN DOES PARTIAL COMPLETION HELP?")
print("=" * 70)

# Compare performance in paths where price ended unfavorably
perf_full, dur_full, vwap_full, bench_full = strategy_4_vectorized(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur
)
perf_part, dur_part, vwap_part, bench_part, comp_part = strategy_4_partial_completion(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
    min_completion_pct=95
)

# Paths where partial completion was used
partial_used = comp_part < 99.9
n_partial = np.sum(partial_used)

print(f"\nPaths using partial completion: {n_partial} ({n_partial/n_sims*100:.1f}%)")

if n_partial > 0:
    perf_full_partial = perf_full[partial_used]
    perf_part_partial = perf_part[partial_used]

    print(f"In those paths:")
    print(f"  Full completion avg: {np.mean(perf_full_partial):.2f} bps")
    print(f"  Partial (95%) avg:   {np.mean(perf_part_partial):.2f} bps")
    print(f"  Improvement:         {np.mean(perf_part_partial) - np.mean(perf_full_partial):+.2f} bps")
