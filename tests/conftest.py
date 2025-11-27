"""
Pytest configuration and fixtures for share buyback strategy tests.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_prices():
    """Simple price series for testing."""
    return np.array([100, 102, 98, 101, 99, 103, 97, 100, 102, 98])


@pytest.fixture
def constant_prices():
    """Constant price series for testing."""
    return np.array([100.0] * 20)


@pytest.fixture
def rising_prices():
    """Rising price series for testing."""
    return np.array([100 + i for i in range(20)], dtype=float)


@pytest.fixture
def falling_prices():
    """Falling price series for testing."""
    return np.array([120 - i for i in range(20)], dtype=float)


@pytest.fixture
def default_params():
    """Default simulation parameters."""
    return {
        'S0': 100.0,
        'mu': 0.0,
        'sigma': 0.25,
        'days': 125,
        'n_simulations': 1000,
        'total_usd': 1_000_000_000,
        'min_duration': 75,
        'max_duration': 125,
        'target_duration': 100,
        'discount_bps': 0
    }


@pytest.fixture
def small_simulation_params():
    """Smaller parameters for faster testing."""
    return {
        'S0': 100.0,
        'mu': 0.0,
        'sigma': 0.25,
        'days': 50,
        'n_simulations': 100,
        'total_usd': 10_000_000,
        'min_duration': 30,
        'max_duration': 50,
        'target_duration': 40,
        'discount_bps': 50
    }


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42
