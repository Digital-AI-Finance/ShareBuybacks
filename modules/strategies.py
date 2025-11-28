"""
Trading strategies for fixed-notional share buyback execution.

Five strategies are implemented:
1. Strategy 1: Simple uniform execution over target duration
2. Strategy 2: Adaptive execution with flexible end time (no discount)
3. Strategy 3: Adaptive execution with discounted benchmark
4. Strategy 4: Multi-factor convex adaptive (D+E+F combined)
5. Strategy 5: Flexible completion adaptive (optional partial completion)
"""

import numpy as np
from typing import Tuple
from .config import (
    INITIAL_PERIOD_DAYS,
    MAX_SPEEDUP_MULTIPLIER,
    EXTRA_SLOW_MULTIPLIER,
    EXTRA_SLOW_THRESHOLD_DAYS,
    S4_BETA,
    S4_GAMMA,
    S4_Z_WINDOW,
    S4_Z_THRESHOLD,
    S4_MAX_MULTIPLIER,
    S4_MIN_MULTIPLIER,
    S4_SIGNAL_BOOST,
    S5_MIN_COMPLETION_PCT,
    S5_UNFAVORABLE_THRESHOLD,
    S5_BETA,
    S5_GAMMA,
    S5_MAX_MULTIPLIER,
    S5_MIN_MULTIPLIER
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


def strategy_4(
    prices: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    beta: float = S4_BETA,
    gamma: float = S4_GAMMA,
    z_window: int = S4_Z_WINDOW,
    z_threshold: float = S4_Z_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Strategy 4: Multi-Factor Convex Adaptive Execution.

    Combines three components:
    - D (Convex): Exponential response to price deviations from benchmark
    - E (Time-Urgency): Quadratic urgency factor as deadline approaches
    - F (Z-Score): Statistical filter to ignore noise

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
    beta : float
        Convex sensitivity parameter (default: 20).
    gamma : float
        Urgency acceleration parameter (default: 3).
    z_window : int
        Rolling window for z-score calculation (default: 20 days).
    z_threshold : float
        Z-score threshold for statistical significance (default: 1.0).

    Returns
    -------
    tuple
        - usd_per_day : np.ndarray - USD executed each day
        - shares_per_day : np.ndarray - Shares bought each day
        - total_shares : float - Total shares acquired
        - end_day : int - Day on which execution completed
    """
    n_days = len(prices)
    base_daily_usd = total_usd / target_duration

    usd_per_day = np.zeros(n_days)
    shares_per_day = np.zeros(n_days)
    remaining_usd = total_usd

    end_day = 0

    for day in range(n_days):
        if remaining_usd <= 0.01:
            break

        current_price = prices[day]

        # Compute expanding window benchmark
        benchmark = np.mean(prices[:day + 1])

        # Component D: Convex scaling (exponential)
        deviation = (benchmark - current_price) / benchmark
        convex_mult = np.exp(beta * deviation)
        convex_mult = np.clip(convex_mult, S4_MIN_MULTIPLIER, S4_MAX_MULTIPLIER)

        # Component E: Time-urgency factor (quadratic)
        time_pct = day / max_duration if max_duration > 0 else 1.0
        urgency = 1.0 + gamma * (time_pct ** 2)

        # Component F: Z-score filter
        z_score = 0.0
        if day >= z_window:
            roll_prices = prices[day - z_window:day]
            roll_mean = np.mean(roll_prices)
            roll_std = np.std(roll_prices)
            if roll_std > 0:
                z_score = (roll_mean - current_price) / roll_std

        # Combine factors
        if abs(z_score) > z_threshold:
            # Statistically significant - full response
            signal_boost = 1.0 + S4_SIGNAL_BOOST * abs(z_score)
            final_mult = convex_mult * urgency * signal_boost
        else:
            # Noise - conservative response (just time pressure)
            final_mult = urgency

        # Apply bounds
        final_mult = np.clip(final_mult, S4_MIN_MULTIPLIER, S4_MAX_MULTIPLIER)

        # Calculate target execution
        target_usd = base_daily_usd * final_mult

        # Apply duration constraints
        days_to_min = max(0, min_duration - day - 1)
        days_to_max = max(1, max_duration - day)

        # Must execute at least this to finish by max_duration
        min_required = remaining_usd / days_to_max

        # Cannot execute more than this to respect min_duration
        if days_to_min > 0:
            max_allowed = remaining_usd / days_to_min
        else:
            max_allowed = remaining_usd

        # Constrain execution within bounds
        today_usd = np.clip(target_usd, min_required, max_allowed)
        today_usd = min(today_usd, remaining_usd)

        # Execute
        usd_per_day[day] = today_usd
        shares_per_day[day] = today_usd / current_price
        remaining_usd -= today_usd
        end_day = day + 1

    total_shares = np.sum(shares_per_day)

    return usd_per_day, shares_per_day, total_shares, end_day


def strategy_5(
    prices: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    min_completion_pct: float = S5_MIN_COMPLETION_PCT,
    unfavorable_threshold: float = S5_UNFAVORABLE_THRESHOLD,
    beta: float = S5_BETA,
    gamma: float = S5_GAMMA
) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
    """
    Strategy 5: Flexible Completion Adaptive Execution.

    Similar to Strategy 4 (convex scaling + time urgency), but allows
    partial completion when prices are unfavorable at deadline.

    Key difference: At deadline, if minimum completion % is met AND
    price is significantly above benchmark, the strategy stops buying
    rather than forcing completion at bad prices.

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
    min_completion_pct : float
        Minimum completion percentage required (85-100). Default: 95.
    unfavorable_threshold : float
        Price deviation threshold below which forced completion is skipped.
        Negative value means price above benchmark. Default: -0.01 (-1%).
    beta : float
        Convex sensitivity parameter (default: 50).
    gamma : float
        Urgency acceleration parameter (default: 1).

    Returns
    -------
    tuple
        - usd_per_day : np.ndarray - USD executed each day
        - shares_per_day : np.ndarray - Shares bought each day
        - total_shares : float - Total shares acquired
        - end_day : int - Day on which execution completed
        - completion_pct : float - Actual completion percentage (0-100)
    """
    n_days = len(prices)
    base_daily_usd = total_usd / target_duration
    min_usd = total_usd * (min_completion_pct / 100.0)

    usd_per_day = np.zeros(n_days)
    shares_per_day = np.zeros(n_days)
    remaining_usd = total_usd
    executed_usd = 0.0

    end_day = 0

    for day in range(min(n_days, max_duration)):
        if remaining_usd <= 0.01:
            break

        current_price = prices[day]

        # Compute expanding window benchmark
        benchmark = np.mean(prices[:day + 1])

        # Component D: Convex scaling (exponential)
        deviation = (benchmark - current_price) / benchmark
        convex_mult = np.exp(beta * deviation)
        convex_mult = np.clip(convex_mult, S5_MIN_MULTIPLIER, S5_MAX_MULTIPLIER)

        # Component E: Time-urgency factor (quadratic)
        time_pct = day / max_duration if max_duration > 0 else 1.0
        urgency = 1.0 + gamma * (time_pct ** 2)

        # Combine factors
        final_mult = convex_mult * urgency

        # Apply bounds
        final_mult = np.clip(final_mult, S5_MIN_MULTIPLIER, S5_MAX_MULTIPLIER)

        # Calculate target execution
        target_usd = base_daily_usd * final_mult

        # Apply duration constraints
        days_to_min = max(0, min_duration - day - 1)
        days_to_max = max(1, max_duration - day)

        # Must execute at least this to finish by max_duration
        # But only if we haven't hit minimum completion yet
        if executed_usd < min_usd:
            min_required = remaining_usd / days_to_max
        else:
            # Minimum met - no forced execution required
            min_required = 0.0

        # Cannot execute more than this to respect min_duration
        if days_to_min > 0:
            max_allowed = remaining_usd / days_to_min
        else:
            max_allowed = remaining_usd

        # Constrain execution within bounds
        today_usd = np.clip(target_usd, min_required, max_allowed)
        today_usd = min(today_usd, remaining_usd)

        # Execute
        usd_per_day[day] = today_usd
        shares_per_day[day] = today_usd / current_price
        remaining_usd -= today_usd
        executed_usd += today_usd
        end_day = day + 1

    # At deadline: decide whether to force-complete remaining
    if remaining_usd > 0.01 and end_day < n_days:
        final_day = end_day - 1 if end_day > 0 else 0
        final_price = prices[final_day]
        final_benchmark = np.mean(prices[:final_day + 1])
        final_deviation = (final_benchmark - final_price) / final_benchmark

        hit_minimum = executed_usd >= min_usd
        price_unfavorable = final_deviation < unfavorable_threshold

        if hit_minimum and price_unfavorable:
            # Accept partial completion - don't buy more
            pass
        else:
            # Force complete at last day
            final_exec_day = min(end_day, n_days - 1)
            usd_per_day[final_exec_day] += remaining_usd
            shares_per_day[final_exec_day] += remaining_usd / prices[final_exec_day]
            executed_usd += remaining_usd
            remaining_usd = 0

    total_shares = np.sum(shares_per_day)
    completion_pct = (executed_usd / total_usd) * 100.0

    return usd_per_day, shares_per_day, total_shares, end_day, completion_pct


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
