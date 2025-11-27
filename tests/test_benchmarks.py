"""
Tests for the benchmarks module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.benchmarks import compute_benchmark, compute_benchmark_at_day


class TestComputeBenchmark:
    """Tests for the compute_benchmark function."""

    def test_benchmark_is_expanding_mean(self, sample_prices):
        """Test that benchmark is the expanding window mean."""
        benchmark = compute_benchmark(sample_prices)

        # Verify each benchmark value manually
        for i in range(len(sample_prices)):
            expected = np.mean(sample_prices[:i + 1])
            assert np.isclose(benchmark[i], expected)

    def test_benchmark_first_day_equals_price(self, sample_prices):
        """Test that benchmark on day 1 equals the first price."""
        benchmark = compute_benchmark(sample_prices)
        assert benchmark[0] == sample_prices[0]

    def test_benchmark_with_discount(self, sample_prices):
        """Test that discount is applied correctly."""
        discount_bps = 100  # 1% discount
        benchmark_no_discount = compute_benchmark(sample_prices, discount_bps=0)
        benchmark_with_discount = compute_benchmark(sample_prices, discount_bps=discount_bps)

        expected_factor = 1 - discount_bps / 10000.0
        np.testing.assert_array_almost_equal(
            benchmark_with_discount,
            benchmark_no_discount * expected_factor
        )

    def test_benchmark_with_zero_discount(self, sample_prices):
        """Test that zero discount gives same result."""
        benchmark1 = compute_benchmark(sample_prices)
        benchmark2 = compute_benchmark(sample_prices, discount_bps=0)

        np.testing.assert_array_equal(benchmark1, benchmark2)

    def test_benchmark_constant_prices(self, constant_prices):
        """Test benchmark with constant prices equals that price."""
        benchmark = compute_benchmark(constant_prices)
        expected = np.full_like(constant_prices, constant_prices[0])

        np.testing.assert_array_equal(benchmark, expected)

    def test_benchmark_rising_prices(self, rising_prices):
        """Test benchmark with rising prices is always below last price."""
        benchmark = compute_benchmark(rising_prices)

        # Benchmark should be below current price for rising prices (after first day)
        for i in range(1, len(rising_prices)):
            assert benchmark[i] < rising_prices[i]

    def test_benchmark_falling_prices(self, falling_prices):
        """Test benchmark with falling prices is always above last price."""
        benchmark = compute_benchmark(falling_prices)

        # Benchmark should be above current price for falling prices (after first day)
        for i in range(1, len(falling_prices)):
            assert benchmark[i] > falling_prices[i]

    def test_benchmark_2d_array(self):
        """Test benchmark computation for 2D array of paths."""
        prices_2d = np.array([
            [100.0, 102.0, 98.0, 101.0],
            [100.0, 99.0, 101.0, 100.0],
            [100.0, 105.0, 103.0, 107.0]
        ], dtype=float)

        benchmark_2d = compute_benchmark(prices_2d)

        # Check shape
        assert benchmark_2d.shape == prices_2d.shape

        # Manually calculate expected benchmarks for first row
        # Day 0: mean([100]) = 100
        # Day 1: mean([100, 102]) = 101
        # Day 2: mean([100, 102, 98]) = 100
        # Day 3: mean([100, 102, 98, 101]) = 100.25
        expected_row0 = np.array([100.0, 101.0, 100.0, 100.25])
        np.testing.assert_array_almost_equal(benchmark_2d[0], expected_row0)

    def test_benchmark_2d_with_discount(self):
        """Test benchmark with discount for 2D array."""
        prices_2d = np.array([
            [100, 102, 98, 101],
            [100, 99, 101, 100]
        ])
        discount_bps = 50

        benchmark_no_disc = compute_benchmark(prices_2d, discount_bps=0)
        benchmark_with_disc = compute_benchmark(prices_2d, discount_bps=discount_bps)

        expected_factor = 1 - discount_bps / 10000.0
        np.testing.assert_array_almost_equal(
            benchmark_with_disc,
            benchmark_no_disc * expected_factor
        )


class TestComputeBenchmarkAtDay:
    """Tests for the compute_benchmark_at_day function."""

    def test_benchmark_at_day_correct(self, sample_prices):
        """Test benchmark at specific day is correct."""
        for day in range(len(sample_prices)):
            result = compute_benchmark_at_day(sample_prices, day)
            expected = np.mean(sample_prices[:day + 1])
            assert np.isclose(result, expected)

    def test_benchmark_at_day_with_discount(self, sample_prices):
        """Test benchmark at day with discount applied."""
        day = 5
        discount_bps = 100

        result = compute_benchmark_at_day(sample_prices, day, discount_bps=discount_bps)
        expected = np.mean(sample_prices[:day + 1]) * (1 - discount_bps / 10000.0)

        assert np.isclose(result, expected)

    def test_benchmark_at_day_invalid_index(self, sample_prices):
        """Test that invalid day index raises error."""
        with pytest.raises(ValueError):
            compute_benchmark_at_day(sample_prices, -1)

        with pytest.raises(ValueError):
            compute_benchmark_at_day(sample_prices, len(sample_prices))

    def test_benchmark_at_day_zero(self, sample_prices):
        """Test benchmark at day 0 equals first price."""
        result = compute_benchmark_at_day(sample_prices, 0)
        assert result == sample_prices[0]
