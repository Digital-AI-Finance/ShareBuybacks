"""
Tests for the GBM (Geometric Brownian Motion) simulation module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gbm import generate_gbm_paths, compute_sigma_envelopes


class TestGenerateGBMPaths:
    """Tests for the generate_gbm_paths function."""

    def test_gbm_shape(self):
        """Test that GBM output has correct shape."""
        n_simulations = 100
        days = 50
        paths = generate_gbm_paths(
            S0=100.0, mu=0.0, sigma=0.25, days=days,
            n_simulations=n_simulations, seed=42
        )
        assert paths.shape == (n_simulations, days)

    def test_gbm_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        params = {'S0': 100.0, 'mu': 0.05, 'sigma': 0.20, 'days': 100, 'n_simulations': 50}

        paths1 = generate_gbm_paths(**params, seed=42)
        paths2 = generate_gbm_paths(**params, seed=42)

        np.testing.assert_array_equal(paths1, paths2)

    def test_gbm_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        params = {'S0': 100.0, 'mu': 0.05, 'sigma': 0.20, 'days': 100, 'n_simulations': 50}

        paths1 = generate_gbm_paths(**params, seed=42)
        paths2 = generate_gbm_paths(**params, seed=123)

        # Should not be equal
        assert not np.allclose(paths1, paths2)

    def test_gbm_starts_at_S0(self):
        """Test that all paths start at S0."""
        S0 = 100.0
        paths = generate_gbm_paths(
            S0=S0, mu=0.0, sigma=0.25, days=50,
            n_simulations=100, seed=42
        )
        # All first values should equal S0
        np.testing.assert_array_equal(paths[:, 0], np.full(100, S0))

    def test_gbm_positive_prices(self):
        """Test that all generated prices are positive."""
        paths = generate_gbm_paths(
            S0=100.0, mu=-0.1, sigma=0.5, days=252,
            n_simulations=1000, seed=42
        )
        assert np.all(paths > 0)

    def test_gbm_drift_direction_positive(self):
        """Test that positive drift results in generally higher prices."""
        n_sims = 10000
        paths = generate_gbm_paths(
            S0=100.0, mu=0.20, sigma=0.10, days=252,
            n_simulations=n_sims, seed=42
        )
        # Mean final price should be higher than S0 with positive drift
        mean_final = np.mean(paths[:, -1])
        assert mean_final > 100.0

    def test_gbm_drift_direction_negative(self):
        """Test that negative drift results in generally lower prices."""
        n_sims = 10000
        paths = generate_gbm_paths(
            S0=100.0, mu=-0.20, sigma=0.10, days=252,
            n_simulations=n_sims, seed=42
        )
        # Mean final price should be lower than S0 with negative drift
        mean_final = np.mean(paths[:, -1])
        assert mean_final < 100.0

    def test_gbm_zero_drift_mean_around_S0(self):
        """Test that zero drift keeps mean around S0."""
        n_sims = 10000
        paths = generate_gbm_paths(
            S0=100.0, mu=0.0, sigma=0.20, days=252,
            n_simulations=n_sims, seed=42
        )
        mean_final = np.mean(paths[:, -1])
        # Should be within 10% of S0 with zero drift
        assert 90.0 < mean_final < 110.0

    def test_gbm_higher_volatility_wider_spread(self):
        """Test that higher volatility produces wider price distribution."""
        n_sims = 5000
        paths_low_vol = generate_gbm_paths(
            S0=100.0, mu=0.0, sigma=0.10, days=252,
            n_simulations=n_sims, seed=42
        )
        paths_high_vol = generate_gbm_paths(
            S0=100.0, mu=0.0, sigma=0.50, days=252,
            n_simulations=n_sims, seed=42
        )

        std_low = np.std(paths_low_vol[:, -1])
        std_high = np.std(paths_high_vol[:, -1])

        assert std_high > std_low


class TestComputeSigmaEnvelopes:
    """Tests for the compute_sigma_envelopes function."""

    def test_envelope_shape(self):
        """Test that envelopes have correct structure."""
        result = compute_sigma_envelopes(S0=100.0, mu=0.0, sigma=0.25, days=50)

        assert 'time' in result
        assert 'mean' in result
        assert '+1sigma' in result
        assert '-1sigma' in result
        assert len(result['time']) == 50

    def test_envelope_mean_at_day_zero(self):
        """Test that mean starts at S0."""
        result = compute_sigma_envelopes(S0=100.0, mu=0.0, sigma=0.25, days=50)
        assert result['mean'][0] == 100.0

    def test_envelope_ordering(self):
        """Test that sigma envelopes are properly ordered."""
        result = compute_sigma_envelopes(S0=100.0, mu=0.0, sigma=0.25, days=50)

        # Upper envelopes should be above mean
        assert np.all(result['+1sigma'] >= result['mean'])
        assert np.all(result['+2sigma'] >= result['+1sigma'])

        # Lower envelopes should be below mean
        assert np.all(result['-1sigma'] <= result['mean'])
        assert np.all(result['-2sigma'] <= result['-1sigma'])
