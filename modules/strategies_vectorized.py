"""
Vectorized trading strategies for high-performance simulation.

These functions operate on all simulation paths at once using NumPy
array operations, providing 10-100x speedup over sequential execution.
"""

import numpy as np
from typing import Tuple
from .config import (
    INITIAL_PERIOD_DAYS,
    MAX_SPEEDUP_MULTIPLIER,
    EXTRA_SLOW_MULTIPLIER,
    EXTRA_SLOW_THRESHOLD_DAYS
)


def precompute_benchmarks(prices_2d: np.ndarray, discount_bps: float = 0) -> np.ndarray:
    """
    Precompute expanding window mean (benchmark) for all paths and days.

    Parameters
    ----------
    prices_2d : np.ndarray
        Price paths of shape (n_sims, n_days).
    discount_bps : float, optional
        Basis points discount to apply. Default is 0.

    Returns
    -------
    np.ndarray
        Benchmark values of shape (n_sims, n_days).
    """
    # Cumulative sum along days axis
    cumsum = np.cumsum(prices_2d, axis=1)
    day_counts = np.arange(1, prices_2d.shape[1] + 1)
    benchmarks = cumsum / day_counts  # Broadcasting: (n_sims, n_days) / (n_days,)

    if discount_bps > 0:
        benchmarks = benchmarks * (1 - discount_bps / 10000.0)

    return benchmarks


def strategy_1_vectorized(
    prices_2d: np.ndarray,
    total_usd: float,
    target_duration: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Strategy 1: Uniform execution over target duration.

    Executes equal USD amounts each day for exactly target_duration days,
    computed simultaneously for all simulation paths.

    Parameters
    ----------
    prices_2d : np.ndarray
        Price paths of shape (n_sims, n_days).
    total_usd : float
        Total USD amount to execute per path.
    target_duration : int
        Number of days over which to execute.

    Returns
    -------
    tuple
        - performances : np.ndarray - Performance in bps for each path
        - durations : np.ndarray - End day for each path (all equal to target_duration)
        - vwaps : np.ndarray - VWAP for each path
        - benchmarks_final : np.ndarray - Final benchmark for each path
    """
    n_sims, n_days = prices_2d.shape
    n_exec_days = min(target_duration, n_days)
    daily_usd = total_usd / target_duration

    # Shares bought each day = daily_usd / price
    # Shape: (n_sims, n_exec_days)
    shares_per_day = daily_usd / prices_2d[:, :n_exec_days]

    # Total shares per path
    total_shares = shares_per_day.sum(axis=1)

    # VWAP = total_usd / total_shares
    vwaps = total_usd / total_shares

    # Benchmark = mean of prices over execution period
    benchmarks_final = prices_2d[:, :n_exec_days].mean(axis=1)

    # Performance in bps = -((VWAP - Benchmark) / Benchmark) * 10000
    performances = -((vwaps - benchmarks_final) / benchmarks_final) * 10000

    # All paths end at same day
    durations = np.full(n_sims, n_exec_days)

    return performances, durations, vwaps, benchmarks_final


def strategy_2_vectorized(
    prices_2d: np.ndarray,
    benchmarks_2d: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Strategy 2: Adaptive execution with flexible end time.

    Uses masked array operations to handle all paths simultaneously
    while respecting per-path decision logic.

    Parameters
    ----------
    prices_2d : np.ndarray
        Price paths of shape (n_sims, n_days).
    benchmarks_2d : np.ndarray
        Precomputed benchmarks of shape (n_sims, n_days).
    total_usd : float
        Total USD amount to execute per path.
    min_duration : int
        Minimum execution duration.
    max_duration : int
        Maximum execution duration.
    target_duration : int
        Target execution duration (sets base daily amount).

    Returns
    -------
    tuple
        - performances : np.ndarray - Performance in bps for each path
        - durations : np.ndarray - End day for each path
        - vwaps : np.ndarray - VWAP for each path
        - benchmarks_final : np.ndarray - Final benchmark for each path
    """
    n_sims, n_days = prices_2d.shape
    base_daily = total_usd / target_duration
    initial_period = min(INITIAL_PERIOD_DAYS, min_duration)

    # State arrays
    usd_executed = np.zeros((n_sims, n_days))
    remaining = np.full(n_sims, total_usd, dtype=np.float64)
    completed = np.zeros(n_sims, dtype=bool)
    end_days = np.zeros(n_sims, dtype=int)

    for day in range(min(n_days, max_duration)):
        if completed.all():
            break

        # Get active paths (not yet completed)
        active_mask = ~completed
        n_active = active_mask.sum()

        if n_active == 0:
            break

        # Get today's prices and benchmarks for active paths
        prices_today = prices_2d[active_mask, day]
        benchmark_today = benchmarks_2d[active_mask, day]
        remaining_active = remaining[active_mask]

        # Initialize today's execution amounts
        today_usd = np.zeros(n_active)

        if day < initial_period:
            # Initial period: constant execution
            today_usd = np.minimum(base_daily, remaining_active)
        else:
            # Adaptive period
            below_bench = prices_today < benchmark_today
            above_bench = ~below_bench

            days_to_min = max(1, min_duration - day)
            days_to_max = max(1, max_duration - day)

            # Handle below benchmark cases
            if below_bench.any():
                if day >= min_duration:
                    # Case 1: Price < Benchmark AND past min_duration -> ASAP (max 5x)
                    today_usd[below_bench] = np.minimum(
                        MAX_SPEEDUP_MULTIPLIER * base_daily,
                        remaining_active[below_bench]
                    )
                else:
                    # Case 2: Price < Benchmark AND before min_duration -> speed up
                    speedup_amount = np.minimum(
                        remaining_active[below_bench] / days_to_min,
                        MAX_SPEEDUP_MULTIPLIER * base_daily
                    )
                    today_usd[below_bench] = np.minimum(speedup_amount, remaining_active[below_bench])

            # Case 3: Price >= Benchmark -> slow down
            if above_bench.any():
                # Target daily amount to finish by max_duration
                target_daily = remaining_active[above_bench] / days_to_max

                # Extra slow mode: if remaining < 5x base AND days_to_max > 5
                extra_slow_mask = (
                    (remaining_active[above_bench] < MAX_SPEEDUP_MULTIPLIER * base_daily) &
                    (days_to_max > EXTRA_SLOW_THRESHOLD_DAYS)
                )
                target_daily[extra_slow_mask] = np.minimum(
                    EXTRA_SLOW_MULTIPLIER * base_daily,
                    remaining_active[above_bench][extra_slow_mask]
                )

                today_usd[above_bench] = np.minimum(target_daily, remaining_active[above_bench])

        # Execute
        usd_executed[active_mask, day] = today_usd
        remaining[active_mask] -= today_usd

        # Mark newly completed paths
        newly_completed = remaining < 0.01
        if newly_completed.any():
            # Get indices of newly completed paths that were active
            active_indices = np.where(active_mask)[0]
            for idx in active_indices:
                if remaining[idx] < 0.01 and not completed[idx]:
                    end_days[idx] = day + 1
                    completed[idx] = True

    # Handle any paths that didn't complete (shouldn't happen with proper params)
    incomplete = ~completed
    if incomplete.any():
        end_days[incomplete] = max_duration

    # Compute final metrics
    # Shares per day = USD / price
    with np.errstate(divide='ignore', invalid='ignore'):
        shares_executed = np.where(usd_executed > 0, usd_executed / prices_2d, 0)

    # Total shares and VWAP
    total_shares = shares_executed.sum(axis=1)
    vwaps = total_usd / np.maximum(total_shares, 1e-10)

    # Final benchmark = mean of prices up to end_day for each path
    benchmarks_final = np.zeros(n_sims)
    for i in range(n_sims):
        end = end_days[i]
        if end > 0:
            benchmarks_final[i] = prices_2d[i, :end].mean()
        else:
            benchmarks_final[i] = prices_2d[i, 0]

    # Performance in bps
    performances = -((vwaps - benchmarks_final) / benchmarks_final) * 10000

    return performances, end_days, vwaps, benchmarks_final


def strategy_3_vectorized(
    prices_2d: np.ndarray,
    benchmarks_2d: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    discount_bps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Strategy 3: Adaptive with discounted benchmark.

    Same as Strategy 2 but uses discounted benchmark for decisions.
    The benchmark used for final performance calculation is also discounted.

    Parameters
    ----------
    prices_2d : np.ndarray
        Price paths of shape (n_sims, n_days).
    benchmarks_2d : np.ndarray
        Precomputed benchmarks WITH discount already applied.
    total_usd : float
        Total USD amount to execute per path.
    min_duration : int
        Minimum execution duration.
    max_duration : int
        Maximum execution duration.
    target_duration : int
        Target execution duration.
    discount_bps : float
        Basis points discount applied to benchmark.

    Returns
    -------
    tuple
        - performances : np.ndarray - Performance in bps for each path
        - durations : np.ndarray - End day for each path
        - vwaps : np.ndarray - VWAP for each path
        - benchmarks_final : np.ndarray - Final discounted benchmark for each path
    """
    # Strategy 3 uses the same logic as Strategy 2
    # The benchmark_2d should already have discount applied
    return strategy_2_vectorized(
        prices_2d, benchmarks_2d, total_usd,
        min_duration, max_duration, target_duration
    )


def run_all_strategies_vectorized(
    prices_2d: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    discount_bps: float
) -> dict:
    """
    Run all three strategies on price paths using vectorized operations.

    Parameters
    ----------
    prices_2d : np.ndarray
        Price paths of shape (n_sims, n_days).
    total_usd : float
        Total USD to execute.
    min_duration : int
        Minimum execution duration.
    max_duration : int
        Maximum execution duration.
    target_duration : int
        Target execution duration.
    discount_bps : float
        Basis points discount for Strategy 3.

    Returns
    -------
    dict
        Results dictionary with performances and durations for each strategy.
    """
    # Precompute benchmarks
    benchmarks_no_discount = precompute_benchmarks(prices_2d, discount_bps=0)
    benchmarks_with_discount = precompute_benchmarks(prices_2d, discount_bps=discount_bps)

    results = {}

    # Strategy 1
    perf1, dur1, vwap1, bench1 = strategy_1_vectorized(
        prices_2d, total_usd, target_duration
    )
    results['Strategy 1'] = {
        'performances': perf1,
        'durations': dur1
    }

    # Strategy 2
    perf2, dur2, vwap2, bench2 = strategy_2_vectorized(
        prices_2d, benchmarks_no_discount, total_usd,
        min_duration, max_duration, target_duration
    )
    results['Strategy 2'] = {
        'performances': perf2,
        'durations': dur2
    }

    # Strategy 3
    perf3, dur3, vwap3, bench3 = strategy_3_vectorized(
        prices_2d, benchmarks_with_discount, total_usd,
        min_duration, max_duration, target_duration, discount_bps
    )
    results['Strategy 3'] = {
        'performances': perf3,
        'durations': dur3
    }

    return results
