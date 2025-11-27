"""
Trading strategies for fixed-notional share buyback execution.

Three strategies are implemented:
1. Strategy 1: Simple uniform execution over target duration
2. Strategy 2: Adaptive execution with flexible end time (no discount)
3. Strategy 3: Adaptive execution with discounted benchmark
"""

import numpy as np
from typing import Tuple
from .config import (
    INITIAL_PERIOD_DAYS,
    MAX_SPEEDUP_MULTIPLIER,
    EXTRA_SLOW_MULTIPLIER,
    EXTRA_SLOW_THRESHOLD_DAYS
)


def strategy_1(
    prices: np.ndarray,
    total_usd: float,
    target_duration: int
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Strategy 1: Simple uniform execution.

    Executes equal USD amounts each day for exactly target_duration days.

    Parameters
    ----------
    prices : np.ndarray
        1D array of stock prices for each trading day.
    total_usd : float
        Total USD amount to execute.
    target_duration : int
        Number of days over which to execute (days 1 through target_duration).

    Returns
    -------
    tuple
        - usd_per_day : np.ndarray - USD executed each day
        - shares_per_day : np.ndarray - Shares bought each day
        - total_shares : float - Total shares acquired
        - end_day : int - Day on which execution completed (equals target_duration)

    Examples
    --------
    >>> prices = np.array([100, 102, 98, 101, 100])
    >>> usd, shares, total, end = strategy_1(prices, 10000, 5)
    >>> end
    5
    >>> np.allclose(usd, 2000)  # Equal USD each day
    True
    """
    n_days = min(target_duration, len(prices))
    daily_usd = total_usd / target_duration

    usd_per_day = np.zeros(len(prices))
    shares_per_day = np.zeros(len(prices))

    for day in range(n_days):
        usd_per_day[day] = daily_usd
        shares_per_day[day] = daily_usd / prices[day]

    total_shares = np.sum(shares_per_day)
    end_day = n_days

    return usd_per_day, shares_per_day, total_shares, end_day


def strategy_2(
    prices: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    discount_bps: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Strategy 2: Adaptive execution with flexible end time.

    The strategy adapts daily execution based on current price vs. benchmark:
    - If price < benchmark: Speed up execution (buy more)
    - If price > benchmark: Slow down execution (buy less)

    Initial period (first 10 days or min_duration, whichever is smaller):
    Constant daily amount = total_usd / target_duration

    Parameters
    ----------
    prices : np.ndarray
        1D array of stock prices for each trading day.
    total_usd : float
        Total USD amount to execute.
    min_duration : int
        Minimum number of days for execution.
    max_duration : int
        Maximum number of days for execution.
    target_duration : int
        Target number of days (used to set base daily amount).
    discount_bps : float, optional
        Basis points discount to apply to benchmark. Default is 0.

    Returns
    -------
    tuple
        - usd_per_day : np.ndarray - USD executed each day
        - shares_per_day : np.ndarray - Shares bought each day
        - total_shares : float - Total shares acquired
        - end_day : int - Day on which execution completed

    Notes
    -----
    Adaptive logic after initial period:

    1. If price < benchmark (favorable):
       - If past min_duration: Finish ASAP, up to 5x base amount per day
       - If before min_duration: Speed up to finish by min_duration

    2. If price > benchmark (unfavorable):
       - Slow down to finish by max_duration
       - Extra slow mode: If remaining < 5 * base_amount AND days_to_max > 5,
         use only 0.1 * base_amount per day
    """
    n_days = len(prices)
    initial_period = min(INITIAL_PERIOD_DAYS, min_duration)
    base_daily_usd = total_usd / target_duration

    usd_per_day = np.zeros(n_days)
    shares_per_day = np.zeros(n_days)
    remaining_usd = total_usd

    end_day = 0

    for day in range(n_days):
        if remaining_usd <= 0:
            break

        current_price = prices[day]

        # Compute benchmark up to current day (expanding window mean)
        benchmark = np.mean(prices[:day + 1])
        if discount_bps != 0:
            benchmark = benchmark * (1 - discount_bps / 10000.0)

        # Determine today's execution amount
        if day < initial_period:
            # Initial period: constant execution
            today_usd = min(base_daily_usd, remaining_usd)
        else:
            # Adaptive period
            days_remaining_to_min = max(0, min_duration - day - 1)
            days_remaining_to_max = max(0, max_duration - day - 1)

            if current_price < benchmark:
                # Favorable price - speed up
                if day >= min_duration:
                    # Past minimum duration: finish ASAP
                    # Maximum is 5x base amount
                    today_usd = min(
                        MAX_SPEEDUP_MULTIPLIER * base_daily_usd,
                        remaining_usd
                    )
                else:
                    # Before minimum duration: speed up to finish by min_duration
                    if days_remaining_to_min > 0:
                        today_usd = min(
                            remaining_usd / days_remaining_to_min,
                            MAX_SPEEDUP_MULTIPLIER * base_daily_usd,
                            remaining_usd
                        )
                    else:
                        # At min_duration day, can finish now
                        today_usd = min(
                            MAX_SPEEDUP_MULTIPLIER * base_daily_usd,
                            remaining_usd
                        )
            else:
                # Unfavorable price - slow down
                if days_remaining_to_max > 0:
                    # Slow down to finish by max_duration
                    target_daily = remaining_usd / (days_remaining_to_max + 1)

                    # Extra slow mode check
                    if (remaining_usd < MAX_SPEEDUP_MULTIPLIER * base_daily_usd and
                            days_remaining_to_max > EXTRA_SLOW_THRESHOLD_DAYS):
                        today_usd = min(
                            EXTRA_SLOW_MULTIPLIER * base_daily_usd,
                            remaining_usd
                        )
                    else:
                        today_usd = min(target_daily, remaining_usd)
                else:
                    # At max_duration: must finish today
                    today_usd = remaining_usd

        # Execute
        usd_per_day[day] = today_usd
        shares_per_day[day] = today_usd / current_price
        remaining_usd -= today_usd
        end_day = day + 1

        # Check if complete
        if remaining_usd < 0.01:  # Small tolerance for floating point
            break

    total_shares = np.sum(shares_per_day)

    return usd_per_day, shares_per_day, total_shares, end_day


def strategy_3(
    prices: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    discount_bps: float
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Strategy 3: Adaptive execution with discounted benchmark.

    Same as Strategy 2 but always applies the specified discount to the benchmark.
    This makes the strategy more aggressive in buying (price appears more favorable
    relative to the discounted benchmark).

    Parameters
    ----------
    prices : np.ndarray
        1D array of stock prices for each trading day.
    total_usd : float
        Total USD amount to execute.
    min_duration : int
        Minimum number of days for execution.
    max_duration : int
        Maximum number of days for execution.
    target_duration : int
        Target number of days (used to set base daily amount).
    discount_bps : float
        Basis points discount to apply to benchmark.

    Returns
    -------
    tuple
        - usd_per_day : np.ndarray - USD executed each day
        - shares_per_day : np.ndarray - Shares bought each day
        - total_shares : float - Total shares acquired
        - end_day : int - Day on which execution completed

    See Also
    --------
    strategy_2 : The base adaptive strategy without discount.
    """
    return strategy_2(
        prices=prices,
        total_usd=total_usd,
        min_duration=min_duration,
        max_duration=max_duration,
        target_duration=target_duration,
        discount_bps=discount_bps
    )


def compute_execution_vwap_series(
    prices: np.ndarray,
    usd_per_day: np.ndarray,
    shares_per_day: np.ndarray
) -> np.ndarray:
    """
    Compute the cumulative VWAP (execution price) over time.

    Parameters
    ----------
    prices : np.ndarray
        Stock prices (not used directly, kept for interface consistency).
    usd_per_day : np.ndarray
        USD executed each day.
    shares_per_day : np.ndarray
        Shares bought each day.

    Returns
    -------
    np.ndarray
        Cumulative VWAP at each day. Returns NaN for days with no execution.
    """
    cumulative_usd = np.cumsum(usd_per_day)
    cumulative_shares = np.cumsum(shares_per_day)

    # Avoid division by zero
    vwap_series = np.where(
        cumulative_shares > 0,
        cumulative_usd / cumulative_shares,
        np.nan
    )

    return vwap_series


def compute_performance_series(
    vwap_series: np.ndarray,
    benchmark_series: np.ndarray
) -> np.ndarray:
    """
    Compute the performance in basis points over time.

    Performance = -((VWAP - Benchmark) / Benchmark) * 10000

    Positive performance means VWAP < Benchmark (good execution).

    Parameters
    ----------
    vwap_series : np.ndarray
        Cumulative VWAP at each day.
    benchmark_series : np.ndarray
        Benchmark at each day.

    Returns
    -------
    np.ndarray
        Performance in basis points at each day.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        performance = -((vwap_series - benchmark_series) / benchmark_series) * 10000

    return performance
