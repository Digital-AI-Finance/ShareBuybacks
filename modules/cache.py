"""
Caching utilities for the Share Buyback Strategy application.

Provides functions to cache expensive computations like GBM path
generation and benchmark calculations.
"""

import streamlit as st
import numpy as np
from typing import Optional, Tuple, Any
import hashlib


def get_params_hash(params: dict) -> str:
    """
    Generate a hash from parameter dictionary for cache keys.

    Parameters
    ----------
    params : dict
        Dictionary of parameters.

    Returns
    -------
    str
        Hash string for cache key.
    """
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()[:16]


def get_cached_paths(
    S0: float,
    mu: float,
    sigma: float,
    n_days: int,
    n_sims: int,
    seed: Optional[int],
    generate_func
) -> np.ndarray:
    """
    Get cached GBM paths or generate new ones.

    Caches up to 3 different path configurations to avoid
    regenerating when only strategy parameters change.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Annual drift.
    sigma : float
        Annual volatility.
    n_days : int
        Number of simulation days.
    n_sims : int
        Number of simulations.
    seed : int or None
        Random seed.
    generate_func : callable
        Function to generate paths if not cached.

    Returns
    -------
    np.ndarray
        Price paths of shape (n_sims, n_days).
    """
    cache_key = (S0, mu, sigma, n_days, n_sims, seed)

    if 'paths_cache' not in st.session_state:
        st.session_state.paths_cache = {}

    if cache_key not in st.session_state.paths_cache:
        # Generate new paths
        paths = generate_func(S0, mu, sigma, n_days, n_sims, seed=seed)
        st.session_state.paths_cache[cache_key] = paths

        # Limit cache size to 3 entries (FIFO)
        if len(st.session_state.paths_cache) > 3:
            oldest_key = next(iter(st.session_state.paths_cache))
            del st.session_state.paths_cache[oldest_key]

    return st.session_state.paths_cache[cache_key]


def get_cached_results(
    stock_params: dict,
    strategy_params: dict
) -> Optional[dict]:
    """
    Get cached simulation results if available.

    Parameters
    ----------
    stock_params : dict
        Stock simulation parameters (S0, mu, sigma, etc.).
    strategy_params : dict
        Strategy parameters (min_duration, max_duration, etc.).

    Returns
    -------
    dict or None
        Cached results or None if not available.
    """
    stock_hash = get_params_hash(stock_params)
    strategy_hash = get_params_hash(strategy_params)
    cache_key = f"{stock_hash}_{strategy_hash}"

    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}

    return st.session_state.results_cache.get(cache_key)


def set_cached_results(
    stock_params: dict,
    strategy_params: dict,
    results: dict
) -> None:
    """
    Cache simulation results.

    Parameters
    ----------
    stock_params : dict
        Stock simulation parameters.
    strategy_params : dict
        Strategy parameters.
    results : dict
        Results to cache.
    """
    stock_hash = get_params_hash(stock_params)
    strategy_hash = get_params_hash(strategy_params)
    cache_key = f"{stock_hash}_{strategy_hash}"

    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}

    st.session_state.results_cache[cache_key] = results

    # Limit cache size
    if len(st.session_state.results_cache) > 5:
        oldest_key = next(iter(st.session_state.results_cache))
        del st.session_state.results_cache[oldest_key]


def clear_all_caches() -> None:
    """Clear all cached data."""
    if 'paths_cache' in st.session_state:
        st.session_state.paths_cache = {}
    if 'results_cache' in st.session_state:
        st.session_state.results_cache = {}


def get_cache_status() -> dict:
    """
    Get current cache status for debugging.

    Returns
    -------
    dict
        Cache statistics.
    """
    paths_count = len(st.session_state.get('paths_cache', {}))
    results_count = len(st.session_state.get('results_cache', {}))

    return {
        'paths_cached': paths_count,
        'results_cached': results_count
    }
