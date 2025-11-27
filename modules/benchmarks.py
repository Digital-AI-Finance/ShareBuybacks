"""
Benchmark calculation for share buyback strategies.

The benchmark represents the average price that a naive investor would pay
if they bought shares proportionally over the execution period.
"""

import numpy as np


def compute_benchmark(prices: np.ndarray, discount_bps: float = 0.0) -> np.ndarray:
    """
    Compute the expanding window arithmetic mean benchmark.

    The benchmark at day t is the arithmetic mean of all prices from day 1 to day t.
    Optionally, a discount in basis points can be applied.

    Parameters
    ----------
    prices : np.ndarray
        Array of stock prices. Can be 1D (single path) or 2D (multiple paths).
        For 2D arrays, shape should be (n_simulations, days).
    discount_bps : float, optional
        Discount to apply to the benchmark in basis points.
        A positive value lowers the benchmark.
        Default is 0 (no discount).

    Returns
    -------
    np.ndarray
        Expanding window mean benchmark values, same shape as input.
        The benchmark at index i is the mean of prices[0:i+1].

    Examples
    --------
    >>> prices = np.array([100, 102, 98, 101])
    >>> compute_benchmark(prices)
    array([100. , 101. , 100. ,  100.25])

    >>> compute_benchmark(prices, discount_bps=100)  # 1% discount
    array([99.  , 99.99, 99.  ,  99.2475])
    """
    # Handle 1D case
    if prices.ndim == 1:
        n_days = len(prices)
        benchmark = np.zeros(n_days)
        cumsum = np.cumsum(prices)
        for i in range(n_days):
            benchmark[i] = cumsum[i] / (i + 1)
    else:
        # Handle 2D case (n_simulations, days)
        n_simulations, n_days = prices.shape
        benchmark = np.zeros_like(prices)
        cumsum = np.cumsum(prices, axis=1)
        for i in range(n_days):
            benchmark[:, i] = cumsum[:, i] / (i + 1)

    # Apply discount if specified
    if discount_bps != 0:
        discount_factor = 1 - discount_bps / 10000.0
        benchmark = benchmark * discount_factor

    return benchmark


def compute_benchmark_at_day(prices: np.ndarray, day: int, discount_bps: float = 0.0) -> float:
    """
    Compute the benchmark at a specific day.

    Parameters
    ----------
    prices : np.ndarray
        1D array of stock prices.
    day : int
        Day index (0-based) at which to compute the benchmark.
    discount_bps : float, optional
        Discount to apply in basis points. Default is 0.

    Returns
    -------
    float
        The benchmark value at the specified day.

    Examples
    --------
    >>> prices = np.array([100, 102, 98, 101])
    >>> compute_benchmark_at_day(prices, 3)
    100.25
    """
    if day < 0 or day >= len(prices):
        raise ValueError(f"Day {day} is out of range for prices array of length {len(prices)}")

    mean_price = np.mean(prices[:day + 1])

    if discount_bps != 0:
        discount_factor = 1 - discount_bps / 10000.0
        mean_price = mean_price * discount_factor

    return mean_price
