"""
Performance metrics for share buyback strategy evaluation.

Key metrics:
- VWAP (Volume-Weighted Average Price): Average price paid per share
- Execution Performance: Savings vs. benchmark in basis points
- Standard Error: Statistical precision of performance estimates
"""

import numpy as np
from typing import Union


def vwap(total_usd_spent: float, total_shares: float) -> float:
    """
    Calculate the Volume-Weighted Average Price (VWAP).

    VWAP represents the average price paid per share across all executions.

    Parameters
    ----------
    total_usd_spent : float
        Total USD amount spent on share purchases.
    total_shares : float
        Total number of shares acquired.

    Returns
    -------
    float
        The VWAP (average price per share).

    Raises
    ------
    ValueError
        If total_shares is zero or negative.

    Examples
    --------
    >>> vwap(100000, 1000)
    100.0
    >>> vwap(98000, 1000)  # Got shares cheaper
    98.0
    """
    if total_shares <= 0:
        raise ValueError(f"total_shares must be positive, got {total_shares}")

    return total_usd_spent / total_shares


def execution_performance_bps(
    execution_vwap: float,
    benchmark: float
) -> float:
    """
    Calculate execution performance in basis points.

    Performance = -((VWAP - Benchmark) / Benchmark) * 10000

    A positive value means the execution VWAP was below the benchmark
    (i.e., shares were acquired at a better price).

    Parameters
    ----------
    execution_vwap : float
        The VWAP achieved by the execution strategy.
    benchmark : float
        The benchmark price (e.g., period average).

    Returns
    -------
    float
        Performance in basis points (bps).
        Positive = outperformance (VWAP < Benchmark)
        Negative = underperformance (VWAP > Benchmark)

    Examples
    --------
    >>> execution_performance_bps(99, 100)  # 1% better than benchmark
    100.0
    >>> execution_performance_bps(101, 100)  # 1% worse than benchmark
    -100.0
    >>> execution_performance_bps(100, 100)  # Equal to benchmark
    0.0
    """
    if benchmark <= 0:
        raise ValueError(f"benchmark must be positive, got {benchmark}")

    return -((execution_vwap - benchmark) / benchmark) * 10000


def standard_error(values: Union[np.ndarray, list]) -> float:
    """
    Calculate the standard error of the mean.

    Standard Error = Standard Deviation / sqrt(N)

    This measures the precision of the sample mean as an estimate
    of the population mean.

    Parameters
    ----------
    values : array-like
        Sample values.

    Returns
    -------
    float
        Standard error of the mean.

    Examples
    --------
    >>> se = standard_error([1, 2, 3, 4, 5])
    >>> round(se, 4)
    0.6325
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return np.nan

    if n == 1:
        return 0.0

    std = np.std(values, ddof=1)  # Sample standard deviation
    return std / np.sqrt(n)


def mean_with_se(values: Union[np.ndarray, list]) -> tuple:
    """
    Calculate mean and standard error together.

    Parameters
    ----------
    values : array-like
        Sample values.

    Returns
    -------
    tuple
        (mean, standard_error)

    Examples
    --------
    >>> mean, se = mean_with_se([100, 102, 98, 101])
    >>> mean
    100.25
    """
    values = np.asarray(values)
    mean = np.mean(values)
    se = standard_error(values)
    return mean, se


def format_mean_se(mean: float, se: float, decimals: int = 2) -> str:
    """
    Format mean +/- standard error as a string.

    Parameters
    ----------
    mean : float
        The mean value.
    se : float
        The standard error.
    decimals : int, optional
        Number of decimal places. Default is 2.

    Returns
    -------
    str
        Formatted string like "100.25 +/- 0.63"

    Examples
    --------
    >>> format_mean_se(100.25, 0.6325)
    '100.25 +/- 0.63'
    """
    return f"{mean:.{decimals}f} +/- {se:.{decimals}f}"


def compute_sharpe_ratio(
    performance_bps: np.ndarray,
    risk_free_rate_bps: float = 0.0
) -> float:
    """
    Calculate the Sharpe ratio of performance.

    Sharpe Ratio = (Mean Performance - Risk Free Rate) / Std(Performance)

    Parameters
    ----------
    performance_bps : np.ndarray
        Array of performance values in basis points.
    risk_free_rate_bps : float, optional
        Risk-free rate in basis points. Default is 0.

    Returns
    -------
    float
        Sharpe ratio.
    """
    mean_perf = np.mean(performance_bps)
    std_perf = np.std(performance_bps, ddof=1)

    if std_perf == 0:
        return np.inf if mean_perf > risk_free_rate_bps else 0.0

    return (mean_perf - risk_free_rate_bps) / std_perf


def compute_win_rate(performance_bps: np.ndarray) -> float:
    """
    Calculate the percentage of simulations with positive performance.

    Parameters
    ----------
    performance_bps : np.ndarray
        Array of performance values in basis points.

    Returns
    -------
    float
        Win rate as a percentage (0-100).
    """
    return 100.0 * np.mean(performance_bps > 0)


def compute_percentiles(
    values: np.ndarray,
    percentiles: list = None
) -> dict:
    """
    Calculate specified percentiles of the distribution.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    percentiles : list, optional
        List of percentiles to compute. Default is [5, 25, 50, 75, 95].

    Returns
    -------
    dict
        Dictionary mapping percentile to value.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    result = {}
    for p in percentiles:
        result[p] = np.percentile(values, p)

    return result
