"""
Tests for the metrics module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.metrics import (
    vwap, execution_performance_bps, standard_error,
    mean_with_se, format_mean_se, compute_sharpe_ratio,
    compute_win_rate, compute_percentiles
)


class TestVWAP:
    """Tests for VWAP calculation."""

    def test_vwap_calculation(self):
        """Test basic VWAP calculation."""
        result = vwap(100000, 1000)
        assert result == 100.0

    def test_vwap_lower_than_initial_price(self):
        """Test VWAP can be lower than a reference price."""
        # If we bought $98,000 worth and got 1000 shares
        result = vwap(98000, 1000)
        assert result == 98.0

    def test_vwap_higher_than_initial_price(self):
        """Test VWAP can be higher than a reference price."""
        # If we bought $102,000 worth and got 1000 shares
        result = vwap(102000, 1000)
        assert result == 102.0

    def test_vwap_zero_shares_raises_error(self):
        """Test that zero shares raises an error."""
        with pytest.raises(ValueError):
            vwap(100000, 0)

    def test_vwap_negative_shares_raises_error(self):
        """Test that negative shares raises an error."""
        with pytest.raises(ValueError):
            vwap(100000, -100)


class TestExecutionPerformanceBps:
    """Tests for execution performance calculation."""

    def test_performance_bps_positive_when_vwap_below_benchmark(self):
        """Test positive performance when VWAP < Benchmark."""
        # VWAP = 99, Benchmark = 100 -> we did better
        result = execution_performance_bps(99, 100)
        assert result == 100.0  # 1% better = 100 bps

    def test_performance_bps_negative_when_vwap_above_benchmark(self):
        """Test negative performance when VWAP > Benchmark."""
        # VWAP = 101, Benchmark = 100 -> we did worse
        result = execution_performance_bps(101, 100)
        assert result == -100.0  # 1% worse = -100 bps

    def test_performance_bps_zero_when_equal(self):
        """Test zero performance when VWAP equals Benchmark."""
        result = execution_performance_bps(100, 100)
        assert result == 0.0

    def test_performance_bps_formula(self):
        """Test the exact formula: -((VWAP - Benchmark) / Benchmark) * 10000."""
        execution_vwap = 98.5
        benchmark = 100.0

        expected = -((execution_vwap - benchmark) / benchmark) * 10000
        result = execution_performance_bps(execution_vwap, benchmark)

        assert np.isclose(result, expected)
        assert np.isclose(result, 150.0)  # 1.5% better

    def test_performance_bps_zero_benchmark_raises_error(self):
        """Test that zero benchmark raises an error."""
        with pytest.raises(ValueError):
            execution_performance_bps(100, 0)


class TestStandardError:
    """Tests for standard error calculation."""

    def test_standard_error_formula(self):
        """Test standard error formula: std / sqrt(n)."""
        values = [1, 2, 3, 4, 5]

        # Manual calculation
        std = np.std(values, ddof=1)
        n = len(values)
        expected = std / np.sqrt(n)

        result = standard_error(values)

        assert np.isclose(result, expected)

    def test_standard_error_known_values(self):
        """Test with known values."""
        # For [1, 2, 3, 4, 5]: mean=3, std=1.5811, se=0.7071
        values = [1, 2, 3, 4, 5]
        result = standard_error(values)

        assert np.isclose(result, 0.7071, rtol=0.01)

    def test_standard_error_empty_array(self):
        """Test standard error of empty array returns NaN."""
        result = standard_error([])
        assert np.isnan(result)

    def test_standard_error_single_value(self):
        """Test standard error of single value returns 0."""
        result = standard_error([42])
        assert result == 0.0

    def test_standard_error_numpy_array(self):
        """Test that function works with numpy arrays."""
        values = np.array([10, 20, 30, 40, 50])
        result = standard_error(values)

        expected = np.std(values, ddof=1) / np.sqrt(len(values))
        assert np.isclose(result, expected)


class TestMeanWithSE:
    """Tests for mean_with_se function."""

    def test_mean_with_se_returns_tuple(self):
        """Test that function returns tuple of (mean, se)."""
        values = [1, 2, 3, 4, 5]
        result = mean_with_se(values)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mean_with_se_values(self):
        """Test mean and SE values are correct."""
        values = [1, 2, 3, 4, 5]
        mean, se = mean_with_se(values)

        assert mean == 3.0
        assert np.isclose(se, standard_error(values))


class TestFormatMeanSE:
    """Tests for format_mean_se function."""

    def test_format_mean_se_default_decimals(self):
        """Test default formatting with 2 decimal places."""
        result = format_mean_se(100.25, 0.6325)
        assert result == "100.25 +/- 0.63"

    def test_format_mean_se_custom_decimals(self):
        """Test formatting with custom decimal places."""
        result = format_mean_se(100.254, 0.6325, decimals=3)
        # 0.6325 rounds to 0.632 or 0.633 depending on rounding mode
        assert result in ["100.254 +/- 0.632", "100.254 +/- 0.633"]


class TestComputeSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio with positive mean performance."""
        performance = np.array([10, 20, 15, 25, 30])
        result = compute_sharpe_ratio(performance)

        mean = np.mean(performance)
        std = np.std(performance, ddof=1)
        expected = mean / std

        assert np.isclose(result, expected)

    def test_sharpe_ratio_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        performance = np.array([10, 20, 15, 25, 30])
        rf_rate = 5

        result = compute_sharpe_ratio(performance, risk_free_rate_bps=rf_rate)

        mean = np.mean(performance)
        std = np.std(performance, ddof=1)
        expected = (mean - rf_rate) / std

        assert np.isclose(result, expected)


class TestComputeWinRate:
    """Tests for win rate calculation."""

    def test_win_rate_all_positive(self):
        """Test win rate when all performances are positive."""
        performance = np.array([10, 20, 15, 25, 30])
        result = compute_win_rate(performance)
        assert result == 100.0

    def test_win_rate_all_negative(self):
        """Test win rate when all performances are negative."""
        performance = np.array([-10, -20, -15, -25, -30])
        result = compute_win_rate(performance)
        assert result == 0.0

    def test_win_rate_mixed(self):
        """Test win rate with mixed performances."""
        performance = np.array([10, -20, 15, -25, 30])  # 3 positive out of 5
        result = compute_win_rate(performance)
        assert result == 60.0


class TestComputePercentiles:
    """Tests for percentile calculation."""

    def test_percentiles_default(self):
        """Test default percentiles [5, 25, 50, 75, 95]."""
        values = np.arange(1, 101)  # 1 to 100
        result = compute_percentiles(values)

        assert 5 in result
        assert 25 in result
        assert 50 in result
        assert 75 in result
        assert 95 in result

    def test_percentiles_custom(self):
        """Test custom percentiles."""
        values = np.arange(1, 101)
        result = compute_percentiles(values, percentiles=[10, 90])

        assert 10 in result
        assert 90 in result
        assert 50 not in result

    def test_percentile_values(self):
        """Test percentile values are approximately correct."""
        values = np.arange(1, 101)
        result = compute_percentiles(values)

        # Median should be around 50
        assert 49 <= result[50] <= 51
