"""
Analyze what drives execution performance and explore more aggressive strategies.

Key insight: With GBM (random walk), we can't predict future prices.
The only edge is WHEN we buy relative to the backward-looking benchmark.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.gbm import generate_gbm_paths
from modules.strategies_vectorized import (
    precompute_benchmarks,
    strategy_1_vectorized,
    strategy_2_vectorized,
    strategy_4_vectorized,
)
from modules.config import S4_BETA, S4_GAMMA

np.random.seed(42)

# Generate test data
n_sims = 10000
n_days = 125
S0 = 100
sigma = 0.25
mu = 0.0

prices = generate_gbm_paths(S0, mu, sigma, n_days, n_sims, seed=42)
benchmarks = precompute_benchmarks(prices)

total_usd = 1_000_000_000
min_dur = 75
max_dur = 125
target_dur = 100


def strategy_threshold_vectorized(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    threshold_pct=2.0,  # Only buy when price is X% below benchmark
    patience_mult=0.05,  # Multiplier when above benchmark (nearly stop buying)
    urgency_start=0.7,   # Start emergency buying at this fraction of max_duration
):
    """
    Threshold-based opportunistic strategy.

    Core idea: Be patient. Only buy aggressively when price is significantly
    below benchmark. Near deadline, accelerate to ensure completion.
    """
    n_paths, n_days = prices_2d.shape

    # Initialize tracking arrays
    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    cumulative_usd = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    threshold = threshold_pct / 100.0

    for day in range(min(n_days, max_duration)):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        # Calculate deviation from benchmark
        deviation = (benchmark - price) / benchmark  # Positive = price below benchmark

        # Time progress
        time_pct = day / max_duration
        days_remaining = max_duration - day

        # Base daily amount (to finish by max_duration)
        base_daily = remaining / np.maximum(days_remaining, 1)

        # Threshold logic
        multiplier = np.ones_like(price)

        # Below threshold: buy aggressively (exponential scaling)
        below_threshold = deviation > threshold
        multiplier[below_threshold] = np.exp(50 * deviation[below_threshold])

        # Above benchmark but within threshold: minimal buying
        within_threshold = (deviation <= threshold) & (deviation >= 0)
        multiplier[within_threshold] = patience_mult + deviation[within_threshold] * 5

        # Above benchmark (unfavorable): nearly stop
        above_benchmark = deviation < 0
        multiplier[above_benchmark] = patience_mult

        # Emergency urgency near deadline
        if time_pct > urgency_start:
            urgency_factor = 1 + 3 * ((time_pct - urgency_start) / (1 - urgency_start)) ** 2
            multiplier *= urgency_factor

        # Clip multiplier
        multiplier = np.clip(multiplier, 0.05, 15.0)

        # Calculate USD to spend
        usd_today = np.minimum(base_daily * multiplier, remaining)
        shares_today = usd_today / price

        # Update tracking
        remaining_usd[active] -= usd_today
        total_shares[active] += shares_today
        cumulative_usd[active] += usd_today

        # Check completion
        newly_completed = remaining_usd < 1.0
        durations[newly_completed & ~completed] = day + 1
        completed = newly_completed

    # Force completion for any remaining
    still_active = ~completed
    if np.any(still_active):
        final_day = min(n_days - 1, max_duration - 1)
        price = prices_2d[still_active, final_day]
        remaining = remaining_usd[still_active]
        shares = remaining / price
        total_shares[still_active] += shares
        cumulative_usd[still_active] += remaining
        remaining_usd[still_active] = 0
        durations[still_active] = final_day + 1

    # Calculate metrics
    vwap = total_usd / total_shares
    final_benchmarks = np.array([
        np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
    ])
    performance = (final_benchmarks - vwap) / final_benchmarks * 10000

    return performance, durations, vwap, final_benchmarks


def strategy_momentum_vectorized(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    lookback=5,          # Days to look back for momentum
    momentum_weight=2.0,  # Weight for momentum signal
):
    """
    Momentum-enhanced strategy.

    Core idea: If price is dropping (negative momentum), buy more aggressively
    as prices are likely to be lower than benchmark soon.
    """
    n_paths, n_days = prices_2d.shape

    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    for day in range(min(n_days, max_duration)):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day
        base_daily = remaining / np.maximum(days_remaining, 1)

        # Price deviation
        deviation = (benchmark - price) / benchmark

        # Momentum: compare current price to lookback-day average
        if day >= lookback:
            past_avg = np.mean(prices_2d[active, day-lookback:day], axis=1)
            momentum = (past_avg - price) / past_avg  # Positive = price falling
        else:
            momentum = np.zeros_like(price)

        # Combined signal
        signal = deviation + momentum_weight * momentum

        # Convex scaling on combined signal
        multiplier = np.exp(40 * signal)

        # Time urgency
        time_pct = day / max_duration
        if time_pct > 0.6:
            urgency = 1 + 2 * ((time_pct - 0.6) / 0.4) ** 2
            multiplier *= urgency

        multiplier = np.clip(multiplier, 0.1, 10.0)

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
        final_day = min(n_days - 1, max_duration - 1)
        price = prices_2d[still_active, final_day]
        remaining = remaining_usd[still_active]
        total_shares[still_active] += remaining / price
        remaining_usd[still_active] = 0
        durations[still_active] = final_day + 1

    vwap = total_usd / total_shares
    final_benchmarks = np.array([
        np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
    ])
    performance = (final_benchmarks - vwap) / final_benchmarks * 10000

    return performance, durations, vwap, final_benchmarks


def strategy_extreme_patience_vectorized(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    buy_threshold_pct=3.0,  # Only buy when price X% below benchmark
    min_buy_pct=0.01,       # Minimum buying even when unfavorable (1% of daily)
):
    """
    Extreme patience strategy.

    Core idea: Almost never buy unless price is significantly below benchmark.
    This maximizes the "discount" captured but risks running out of time.
    """
    n_paths, n_days = prices_2d.shape

    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    threshold = buy_threshold_pct / 100.0

    for day in range(min(n_days, max_duration)):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day
        base_daily = remaining / np.maximum(days_remaining, 1)

        deviation = (benchmark - price) / benchmark

        # Very aggressive threshold-based buying
        multiplier = np.full_like(price, min_buy_pct)

        # Only buy significantly when below threshold
        below_threshold = deviation > threshold
        # Exponential scaling for significant discounts
        multiplier[below_threshold] = np.exp(60 * (deviation[below_threshold] - threshold))

        # Moderate buying when slightly below benchmark
        moderate = (deviation > 0) & (deviation <= threshold)
        multiplier[moderate] = min_buy_pct + (deviation[moderate] / threshold) * 0.5

        # Emergency buying in final days
        if days_remaining <= 10:
            emergency_factor = (11 - days_remaining) / 5
            multiplier = np.maximum(multiplier, emergency_factor)

        multiplier = np.clip(multiplier, min_buy_pct, 20.0)

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
        final_day = min(n_days - 1, max_duration - 1)
        price = prices_2d[still_active, final_day]
        remaining = remaining_usd[still_active]
        total_shares[still_active] += remaining / price
        remaining_usd[still_active] = 0
        durations[still_active] = final_day + 1

    vwap = total_usd / total_shares
    final_benchmarks = np.array([
        np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
    ])
    performance = (final_benchmarks - vwap) / final_benchmarks * 10000

    return performance, durations, vwap, final_benchmarks


def strategy_volatility_aware_vectorized(
    prices_2d, benchmarks_2d, total_usd, min_duration, max_duration, target_duration,
    vol_window=10,
):
    """
    Volatility-aware strategy.

    Core idea: In high volatility periods, be more aggressive on discounts
    because bigger opportunities exist. In low vol, be patient.
    """
    n_paths, n_days = prices_2d.shape

    remaining_usd = np.full(n_paths, total_usd, dtype=np.float64)
    total_shares = np.zeros(n_paths, dtype=np.float64)
    durations = np.zeros(n_paths, dtype=np.int32)
    completed = np.zeros(n_paths, dtype=bool)

    # Precompute rolling volatility
    returns = np.diff(np.log(prices_2d), axis=1)

    for day in range(min(n_days, max_duration)):
        active = ~completed
        if not np.any(active):
            break

        price = prices_2d[active, day]
        benchmark = benchmarks_2d[active, day]
        remaining = remaining_usd[active]

        days_remaining = max_duration - day
        base_daily = remaining / np.maximum(days_remaining, 1)

        deviation = (benchmark - price) / benchmark

        # Calculate recent volatility
        if day >= vol_window:
            recent_vol = np.std(returns[active, day-vol_window:day], axis=1)
        else:
            recent_vol = np.full(np.sum(active), 0.01)

        # Normalize volatility (higher vol = more aggressive)
        vol_factor = recent_vol / 0.015  # Normalize to ~1 for typical daily vol
        vol_factor = np.clip(vol_factor, 0.5, 3.0)

        # Base convex scaling
        base_mult = np.exp(40 * deviation)

        # Adjust by volatility: high vol + positive deviation = very aggressive
        multiplier = base_mult * (1 + (vol_factor - 1) * np.maximum(deviation, 0) * 10)

        # Time urgency
        time_pct = day / max_duration
        if time_pct > 0.7:
            urgency = 1 + 2 * ((time_pct - 0.7) / 0.3) ** 2
            multiplier *= urgency

        multiplier = np.clip(multiplier, 0.1, 12.0)

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
        final_day = min(n_days - 1, max_duration - 1)
        price = prices_2d[still_active, final_day]
        remaining = remaining_usd[still_active]
        total_shares[still_active] += remaining / price
        remaining_usd[still_active] = 0
        durations[still_active] = final_day + 1

    vwap = total_usd / total_shares
    final_benchmarks = np.array([
        np.mean(prices_2d[i, :durations[i]]) for i in range(n_paths)
    ])
    performance = (final_benchmarks - vwap) / final_benchmarks * 10000

    return performance, durations, vwap, final_benchmarks


print("=" * 70)
print("PERFORMANCE DRIVER ANALYSIS")
print("=" * 70)
print(f"Simulations: {n_sims:,}")
print(f"Days: {n_days}, Duration: {min_dur}-{max_dur}, Target: {target_dur}")
print(f"Volatility: {sigma*100:.0f}%")
print()

# Baseline strategies
print("BASELINE STRATEGIES:")
print("-" * 70)

perf1, dur1, _, _ = strategy_1_vectorized(prices, total_usd, target_dur)
print(f"Strategy 1 (Uniform):      {np.mean(perf1):8.2f} bps  (std: {np.std(perf1):.2f})")

perf2, dur2, _, _ = strategy_2_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur)
print(f"Strategy 2 (Adaptive):     {np.mean(perf2):8.2f} bps  (std: {np.std(perf2):.2f})")

perf4, dur4, _, _ = strategy_4_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur)
print(f"Strategy 4 (Convex):       {np.mean(perf4):8.2f} bps  (std: {np.std(perf4):.2f})")

print()
print("EXPERIMENTAL STRATEGIES:")
print("-" * 70)

# Test threshold strategy with different parameters
for threshold in [1.0, 2.0, 3.0, 5.0]:
    perf_t, dur_t, _, _ = strategy_threshold_vectorized(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        threshold_pct=threshold
    )
    print(f"Threshold ({threshold}%):           {np.mean(perf_t):8.2f} bps  (std: {np.std(perf_t):.2f}, avg_dur: {np.mean(dur_t):.1f})")

print()

# Momentum strategy
for lookback in [3, 5, 10]:
    perf_m, dur_m, _, _ = strategy_momentum_vectorized(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        lookback=lookback
    )
    print(f"Momentum (lb={lookback}):          {np.mean(perf_m):8.2f} bps  (std: {np.std(perf_m):.2f}, avg_dur: {np.mean(dur_m):.1f})")

print()

# Extreme patience
for threshold in [2.0, 3.0, 5.0]:
    perf_p, dur_p, _, _ = strategy_extreme_patience_vectorized(
        prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
        buy_threshold_pct=threshold
    )
    print(f"Extreme Patience ({threshold}%):   {np.mean(perf_p):8.2f} bps  (std: {np.std(perf_p):.2f}, avg_dur: {np.mean(dur_p):.1f})")

print()

# Volatility aware
perf_v, dur_v, _, _ = strategy_volatility_aware_vectorized(
    prices, benchmarks, total_usd, min_dur, max_dur, target_dur
)
print(f"Volatility Aware:          {np.mean(perf_v):8.2f} bps  (std: {np.std(perf_v):.2f}, avg_dur: {np.mean(dur_v):.1f})")

print()
print("=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
1. With GBM (random walk), future prices are UNPREDICTABLE
2. The ONLY edge comes from buying when price < benchmark (backward-looking)
3. More aggressive thresholds can capture bigger discounts BUT:
   - Risk running out of time (emergency buying at end negates gains)
   - Higher variance in outcomes
4. The fundamental limit is that benchmark UPDATES as we execute
   - If we wait for big discounts, benchmark drops too
   - This creates diminishing returns for patience
5. Momentum signals have limited value in pure random walks
""")

# Find best configuration
print()
print("BEST EXPERIMENTAL RESULT:")
all_means = {
    'Threshold 1%': np.mean(strategy_threshold_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur, threshold_pct=1.0)[0]),
    'Threshold 2%': np.mean(strategy_threshold_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur, threshold_pct=2.0)[0]),
    'Threshold 3%': np.mean(strategy_threshold_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur, threshold_pct=3.0)[0]),
    'Extreme Patience 3%': np.mean(strategy_extreme_patience_vectorized(prices, benchmarks, total_usd, min_dur, max_dur, target_dur, buy_threshold_pct=3.0)[0]),
}
best_name = max(all_means, key=all_means.get)
print(f"{best_name}: {all_means[best_name]:.2f} bps")
print(f"Improvement over Strategy 4: {all_means[best_name] - np.mean(perf4):.2f} bps")
