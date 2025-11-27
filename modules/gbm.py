"""
Geometric Brownian Motion (GBM) simulation for stock price paths.

The GBM model assumes stock prices follow:
    dS = mu * S * dt + sigma * S * dW

Where:
    S = stock price
    mu = drift (annualized)
    sigma = volatility (annualized)
    dW = Wiener process increment
"""

import numpy as np
from .config import TRADING_DAYS_PER_YEAR


def generate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    days: int,
    n_simulations: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate stock price paths using Geometric Brownian Motion.

    Uses the exact discretization formula:
        S[t+1] = S[t] * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Where Z ~ N(0,1) and dt = 1/252 (one trading day).

    Parameters
    ----------
    S0 : float
        Initial stock price at day 1.
    mu : float
        Annual drift rate (as decimal, e.g., 0.05 for 5%).
    sigma : float
        Annual volatility (as decimal, e.g., 0.25 for 25%).
    days : int
        Number of trading days to simulate.
    n_simulations : int
        Number of Monte Carlo paths to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_simulations, days) containing price paths.
        Day 1 (index 0) is the starting price S0.

    Examples
    --------
    >>> paths = generate_gbm_paths(100, 0.05, 0.20, 252, 1000, seed=42)
    >>> paths.shape
    (1000, 252)
    >>> paths[:, 0]  # All paths start at S0
    array([100., 100., ..., 100.])
    """
    if seed is not None:
        np.random.seed(seed)

    # Time step in years (one trading day)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # Pre-compute drift and diffusion terms
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Initialize price array
    # Shape: (n_simulations, days)
    prices = np.zeros((n_simulations, days))
    prices[:, 0] = S0  # Day 1 is the starting price

    # Generate all random shocks at once for efficiency
    # We need (days - 1) shocks for each simulation
    Z = np.random.standard_normal((n_simulations, days - 1))

    # Compute log returns and cumulate
    log_returns = drift + diffusion * Z

    # Build price paths using cumulative sum of log returns
    for t in range(1, days):
        prices[:, t] = prices[:, t - 1] * np.exp(log_returns[:, t - 1])

    return prices


def compute_sigma_envelopes(
    S0: float,
    mu: float,
    sigma: float,
    days: int,
    sigmas: list = None
) -> dict:
    """
    Compute theoretical sigma envelopes for GBM paths.

    For a GBM process, the expected value and standard deviation at time t are:
        E[S(t)] = S0 * exp(mu * t)
        Std[S(t)] = S0 * exp(mu * t) * sqrt(exp(sigma^2 * t) - 1)

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Annual drift rate.
    sigma : float
        Annual volatility.
    days : int
        Number of trading days.
    sigmas : list, optional
        List of sigma levels for envelopes. Default is [1, 2, 3, 4].

    Returns
    -------
    dict
        Dictionary with keys 'time', 'mean', and envelope levels like '+1sigma', '-1sigma'.
    """
    if sigmas is None:
        sigmas = [1, 2, 3, 4]

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    t = np.arange(days) * dt  # Time in years

    # Expected value at each time
    mean = S0 * np.exp(mu * t)

    # Standard deviation at each time
    # Using the formula for GBM variance
    variance_factor = np.exp(sigma ** 2 * t) - 1
    std = mean * np.sqrt(np.maximum(variance_factor, 0))

    result = {
        'time': np.arange(days),
        'mean': mean
    }

    for s in sigmas:
        result[f'+{s}sigma'] = mean + s * std
        result[f'-{s}sigma'] = mean - s * std

    return result
