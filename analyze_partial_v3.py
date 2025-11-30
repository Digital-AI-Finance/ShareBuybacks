"""
Partial completion v3: Cleaner analysis.

Strategy: Run normal Strategy 2/4, but at the FINAL forced completion step,
decide whether to complete based on price favorability.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.gbm import generate_gbm_paths
from modules.strategies_vectorized import precompute_benchmarks
from modules.config import S4_BETA, S4_GAMMA, INITIAL_PERIOD_DAYS, MAX_SPEEDUP_MULTIPLIER

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


def strategy_2_with_partial(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    min_completion_pct=100.0,
    unfavorable_threshold=-0.01,  # Stop if price is X% above benchmark
):
    """
    Strategy 2 with partial completion option.
    Exact same logic as Strategy 2, but at deadline can choose not to complete.
    """
    n_paths, n_days_total = prices_2d.shape
    base_daily = total_usd / target_duration

    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    cumulative_usd = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    min_usd = total_usd * (min_completion_pct / 100.0)

    for day in range(min(n_days_total, max_duration)):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day

        if day < INITIAL_PERIOD_DAYS:
            # Initial period: constant buying
            usd_today = np.full_like(price, base_daily)
        else:
            # Adaptive phase
            deviation = (benchmark - price) / benchmark

            multiplier = np.ones_like(price)

            # Speedup when favorable
            favorable = deviation > 0
            multiplier[favorable] = 1 + (MAX_SPEEDUP_MULTIPLIER - 1) * np.minimum(deviation[favorable] * 10, 1)

            # Slowdown when unfavorable
            unfavorable = deviation <= 0
            multiplier[unfavorable] = 0.5

            # Urgency near end
            if day >= min_duration:
                urgency = remaining / np.maximum(days_remaining, 1) / base_daily
                multiplier = np.maximum(multiplier, urgency * 0.8)

            usd_today = base_daily * multiplier

        usd_today = np.minimum(usd_today, remaining)
        shares_today = usd_today / price

        remaining_usd[active] -= usd_today
        total_shares[active] += shares_today
        cumulative_usd[active] += usd_today

        fully_done = remaining_usd < 1.0
        durations[fully_done & ~completed] = day + 1
        completed = fully_done

    # Handle paths that haven't completed
    still_active = ~completed
    if np.any(still_active):
        final_day = min(n_days_total, max_duration) - 1
        indices = np.where(still_active)[0]

        for i in indices:
            final_price = prices_2d[i, final_day]
            final_benchmark = benchmarks_2d[i, final_day]
            deviation = (final_benchmark - final_price) / final_benchmark

            remaining = remaining_usd[i]
            executed = cumulative_usd[i]

            # Decision: complete or accept partial?
            hit_minimum = executed >= min_usd
            price_very_unfavorable = deviation < unfavorable_threshold

            if hit_minimum and price_very_unfavorable:
                # Accept partial - don't buy more
                pass
            else:
                # Force complete
                total_shares[i] += remaining / final_price
                cumulative_usd[i] += remaining
                remaining_usd[i] = 0

            durations[i] = final_day + 1

    # Calculate metrics
    vwap = cumulative_usd / np.maximum(total_shares, 1e-10)
    final_benchmarks = np.array([
        np.mean(prices_2d[i, :max(durations[i], 1)]) for i in range(n_paths)
    ])
    performance = (final_benchmarks - vwap) / final_benchmarks * 10000
    completion_pct = cumulative_usd / total_usd * 100

    return performance, durations, vwap, final_benchmarks, completion_pct


def strategy_4_with_partial(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    min_completion_pct=100.0,
    unfavorable_threshold=-0.01,
    beta=S4_BETA,
    gamma=S4_GAMMA,
):
    """
    Strategy 4 with partial completion option.
    """
    n_paths, n_days_total = prices_2d.shape

    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    cumulative_usd = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    min_usd = total_usd * (min_completion_pct / 100.0)

    for day in range(min(n_days_total, max_duration)):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day
        base_daily = remaining / np.maximum(days_remaining, 1)

        # Deviation
        deviation = (benchmark - price) / benchmark

        # Convex scaling
        convex_mult = np.exp(beta * deviation)

        # Time urgency
        time_pct = day / max_duration
        urgency = 1 + gamma * (time_pct ** 2)

        multiplier = convex_mult * urgency
        multiplier = np.clip(multiplier, 0.1, 8.0)

        usd_today = np.minimum(base_daily * multiplier, remaining)
        shares_today = usd_today / price

        remaining_usd[active] -= usd_today
        total_shares[active] += shares_today
        cumulative_usd[active] += usd_today

        fully_done = remaining_usd < 1.0
        durations[fully_done & ~completed] = day + 1
        completed = fully_done

    # Handle incomplete paths
    still_active = ~completed
    if np.any(still_active):
        final_day = min(n_days_total, max_duration) - 1
        indices = np.where(still_active)[0]

        for i in indices:
            final_price = prices_2d[i, final_day]
            final_benchmark = benchmarks_2d[i, final_day]
            deviation = (final_benchmark - final_price) / final_benchmark

            remaining = remaining_usd[i]
            executed = cumulative_usd[i]

            hit_minimum = executed >= min_usd
            price_very_unfavorable = deviation < unfavorable_threshold

            if hit_minimum and price_very_unfavorable:
                pass  # Accept partial
            else:
                total_shares[i] += remaining / final_price
                cumulative_usd[i] += remaining
                remaining_usd[i] = 0

            durations[i] = final_day + 1

    vwap = cumulative_usd / np.maximum(total_shares, 1e-10)
    final_benchmarks = np.array([
        np.mean(prices_2d[i, :max(durations[i], 1)]) for i in range(n_paths)
    ])
    performance = (final_benchmarks - vwap) / final_benchmarks * 10000
    completion_pct = cumulative_usd / total_usd * 100

    return performance, durations, vwap, final_benchmarks, completion_pct


print("=" * 70)
print("PARTIAL COMPLETION ANALYSIS V3")
print("=" * 70)
print(f"Simulations: {n_sims:,}, Volatility: {sigma*100:.0f}%")
print()

# Test Strategy 2
print("STRATEGY 2 COMPARISON:")
print("-" * 70)

perf_s2_100, dur_s2_100, _, _, comp_s2_100 = strategy_2_with_partial(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
    min_completion_pct=100
)
print(f"100% completion: {np.mean(perf_s2_100):8.2f} bps, avg completion: {np.mean(comp_s2_100):.2f}%")

for min_pct in [99, 98, 95, 90, 85]:
    perf, dur, _, _, comp = strategy_2_with_partial(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        min_completion_pct=min_pct
    )
    diff = np.mean(perf) - np.mean(perf_s2_100)
    partial_used = np.sum(comp < 99.9)
    print(f"{min_pct:3}% minimum:  {np.mean(perf):8.2f} bps ({diff:+6.2f}), avg completion: {np.mean(comp):.2f}%, partial in {partial_used} paths")

print()

# Test Strategy 4
print("STRATEGY 4 COMPARISON:")
print("-" * 70)

perf_s4_100, dur_s4_100, _, _, comp_s4_100 = strategy_4_with_partial(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
    min_completion_pct=100
)
print(f"100% completion: {np.mean(perf_s4_100):8.2f} bps, avg completion: {np.mean(comp_s4_100):.2f}%")

for min_pct in [99, 98, 95, 90, 85]:
    perf, dur, _, _, comp = strategy_4_with_partial(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        min_completion_pct=min_pct
    )
    diff = np.mean(perf) - np.mean(perf_s4_100)
    partial_used = np.sum(comp < 99.9)
    print(f"{min_pct:3}% minimum:  {np.mean(perf):8.2f} bps ({diff:+6.2f}), avg completion: {np.mean(comp):.2f}%, partial in {partial_used} paths")

print()

# Analyze best result
print("=" * 70)
print("DETAILED ANALYSIS: Strategy 4 with 95% minimum")
print("=" * 70)

perf_full, _, _, _, _ = strategy_4_with_partial(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
    min_completion_pct=100
)
perf_95, _, _, _, comp_95 = strategy_4_with_partial(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
    min_completion_pct=95
)

partial_paths = comp_95 < 99.9
full_paths = ~partial_paths

print(f"\nTotal paths: {n_sims}")
print(f"Paths that completed 100%: {np.sum(full_paths)} ({np.sum(full_paths)/n_sims*100:.1f}%)")
print(f"Paths with partial completion: {np.sum(partial_paths)} ({np.sum(partial_paths)/n_sims*100:.1f}%)")

if np.sum(partial_paths) > 0:
    print(f"\nPerformance breakdown:")
    print(f"  Full paths - Full strategy: {np.mean(perf_full[full_paths]):.2f} bps")
    print(f"  Full paths - 95% strategy:  {np.mean(perf_95[full_paths]):.2f} bps")
    print(f"  Partial paths - Full strategy: {np.mean(perf_full[partial_paths]):.2f} bps")
    print(f"  Partial paths - 95% strategy:  {np.mean(perf_95[partial_paths]):.2f} bps")
    print(f"  Improvement in partial paths:  {np.mean(perf_95[partial_paths]) - np.mean(perf_full[partial_paths]):+.2f} bps")

# Dollar value
print()
print("=" * 70)
print("DOLLAR VALUE:")
print("=" * 70)
improvement_bps = np.mean(perf_95) - np.mean(perf_full)
dollar_value = improvement_bps / 10000 * total_usd
print(f"\nImprovement: {improvement_bps:+.2f} bps")
print(f"For ${total_usd/1e9:.1f}B program: ${dollar_value/1e6:+.2f}M")
print(f"Average completion: {np.mean(comp_95):.2f}%")
print(f"Unexecuted amount: ${total_usd * (1 - np.mean(comp_95)/100) / 1e6:.2f}M")
