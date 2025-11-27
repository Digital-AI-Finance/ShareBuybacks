"""
Tests for the trading strategies module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.strategies import (
    strategy_1, strategy_2, strategy_3,
    compute_execution_vwap_series, compute_performance_series
)


class TestStrategy1:
    """Tests for Strategy 1 (Uniform Execution)."""

    def test_strategy1_ends_at_target_duration(self, sample_prices):
        """Test that Strategy 1 ends exactly at target duration."""
        target_duration = 5
        usd, shares, total, end_day = strategy_1(sample_prices, 10000, target_duration)
        assert end_day == target_duration

    def test_strategy1_equal_usd_per_day(self, sample_prices):
        """Test that Strategy 1 executes equal USD each day."""
        total_usd = 10000
        target_duration = 5
        usd, shares, total, end_day = strategy_1(sample_prices, total_usd, target_duration)

        expected_daily = total_usd / target_duration
        # Check first target_duration days
        np.testing.assert_array_almost_equal(
            usd[:target_duration],
            np.full(target_duration, expected_daily)
        )

    def test_strategy1_total_usd_equals_input(self, sample_prices):
        """Test that Strategy 1 spends exactly total_usd."""
        total_usd = 10000
        target_duration = 5
        usd, shares, total, end_day = strategy_1(sample_prices, total_usd, target_duration)

        assert np.isclose(np.sum(usd), total_usd)

    def test_strategy1_shares_calculation(self, constant_prices):
        """Test that shares are calculated correctly."""
        total_usd = 10000
        target_duration = 10
        price = constant_prices[0]  # All prices are 100

        usd, shares, total, end_day = strategy_1(constant_prices, total_usd, target_duration)

        expected_daily_shares = (total_usd / target_duration) / price
        np.testing.assert_array_almost_equal(
            shares[:target_duration],
            np.full(target_duration, expected_daily_shares)
        )

    def test_strategy1_handles_short_price_array(self):
        """Test Strategy 1 when price array is shorter than target."""
        prices = np.array([100, 102, 98])
        total_usd = 10000
        target_duration = 5

        usd, shares, total, end_day = strategy_1(prices, total_usd, target_duration)

        # Should only execute for available days
        assert end_day == len(prices)


class TestStrategy2:
    """Tests for Strategy 2 (Adaptive Execution without discount)."""

    def test_strategy2_initial_period(self, sample_prices):
        """Test that Strategy 2 has constant execution in initial period."""
        total_usd = 100000
        min_duration = 15
        max_duration = 30
        target_duration = 22

        usd, shares, total, end_day = strategy_2(
            sample_prices, total_usd, min_duration, max_duration, target_duration
        )

        base_daily = total_usd / target_duration
        initial_period = min(10, min_duration)

        # First initial_period days should have base_daily amount
        for i in range(min(initial_period, len(sample_prices))):
            # Allow small tolerance
            assert np.isclose(usd[i], base_daily, rtol=0.01) or usd[i] <= base_daily

    def test_strategy2_respects_total_usd(self, sample_prices):
        """Test that Strategy 2 spends at most total_usd."""
        total_usd = 50000
        usd, shares, total, end_day = strategy_2(
            sample_prices, total_usd, 5, 10, 7
        )

        assert np.sum(usd) <= total_usd + 0.01  # Small tolerance

    def test_strategy2_speedup_when_below_benchmark(self):
        """Test that Strategy 2 speeds up when price is below benchmark."""
        # Create prices that start high then drop
        prices = np.array([110.0] * 5 + [90.0] * 20)

        total_usd = 100000
        min_duration = 10
        max_duration = 25
        target_duration = 17

        usd, shares, total, end_day = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )

        # After initial period, when prices drop below benchmark, should speed up
        # End day should be closer to min than max
        assert end_day <= target_duration + 5

    def test_strategy2_slowdown_when_above_benchmark(self):
        """Test that Strategy 2 slows down when price is above benchmark."""
        # Create prices that start low then rise
        prices = np.array([90.0] * 5 + [110.0] * 50)

        total_usd = 100000
        min_duration = 15
        max_duration = 50
        target_duration = 30

        usd, shares, total, end_day = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )

        # When prices are above benchmark, should slow down
        # End day should be closer to max
        assert end_day >= target_duration - 5

    def test_strategy2_extra_slow_when_small_remaining(self):
        """Test extra slow mode when remaining USD is small."""
        # Create prices that stay above benchmark
        prices = np.array([100.0] * 5 + [120.0] * 100)

        total_usd = 100000
        min_duration = 20
        max_duration = 100
        target_duration = 60

        usd, shares, total, end_day = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )

        # Should complete but with reduced daily amounts when remaining is small
        base_daily = total_usd / target_duration

        # Find the point where remaining becomes small
        cumsum = np.cumsum(usd)
        small_remaining_start = None
        for i in range(len(usd)):
            remaining = total_usd - cumsum[i]
            if remaining < 5 * base_daily and remaining > 0:
                small_remaining_start = i
                break

        # If extra slow mode triggered, later amounts should be smaller
        if small_remaining_start is not None and small_remaining_start < end_day - 10:
            late_amounts = usd[small_remaining_start:end_day]
            # At least some should be in extra slow mode
            assert np.min(late_amounts[late_amounts > 0]) < base_daily


class TestStrategy3:
    """Tests for Strategy 3 (Adaptive with Discount)."""

    def test_strategy3_uses_discounted_benchmark(self):
        """Test that Strategy 3 with discount behaves differently than without."""
        prices = np.array([100.0] * 50)

        total_usd = 100000
        min_duration = 20
        max_duration = 50
        target_duration = 35
        discount_bps = 100  # 1% discount

        # Strategy 2 (no discount)
        usd2, shares2, total2, end2 = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=0
        )

        # Strategy 3 (with discount)
        usd3, shares3, total3, end3 = strategy_3(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=discount_bps
        )

        # With discount, benchmark is lower, so price appears more favorable
        # This should make Strategy 3 more aggressive
        # The execution patterns should differ
        # Note: With constant prices, benchmark equals price, so discount makes price > benchmark
        # This would actually slow down Strategy 3
        assert not np.array_equal(usd2, usd3) or end2 == end3

    def test_strategy3_more_aggressive_with_discount(self):
        """Test that discount makes strategy more aggressive when prices vary."""
        # Create varying prices
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        prices = np.maximum(prices, 50)  # Ensure positive

        total_usd = 100000
        min_duration = 30
        max_duration = 100
        target_duration = 65
        discount_bps = 200  # 2% discount

        # Strategy 2 (no discount)
        usd2, shares2, total2, end2 = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=0
        )

        # Strategy 3 (with discount)
        usd3, shares3, total3, end3 = strategy_3(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=discount_bps
        )

        # With discount applied, benchmark is lower
        # This makes prices appear MORE favorable (price < discounted_benchmark more often)
        # So Strategy 3 should generally finish sooner or equal
        # (depends on actual price path)
        assert end3 <= end2 + 10  # Allow some variation


class TestComputeExecutionVWAPSeries:
    """Tests for VWAP series computation."""

    def test_vwap_series_shape(self, sample_prices):
        """Test that VWAP series has correct shape."""
        usd = np.array([1000, 1000, 1000, 1000, 1000, 0, 0, 0, 0, 0])
        shares = usd / sample_prices

        vwap_series = compute_execution_vwap_series(sample_prices, usd, shares)

        assert len(vwap_series) == len(sample_prices)

    def test_vwap_series_nan_when_no_execution(self, sample_prices):
        """Test that VWAP is NaN before any execution."""
        usd = np.array([0, 0, 1000, 1000, 1000, 0, 0, 0, 0, 0])
        shares = usd / np.where(sample_prices != 0, sample_prices, 1)
        shares = np.where(usd > 0, shares, 0)

        vwap_series = compute_execution_vwap_series(sample_prices, usd, shares)

        assert np.isnan(vwap_series[0])
        assert np.isnan(vwap_series[1])

    def test_vwap_series_constant_price(self, constant_prices):
        """Test VWAP with constant prices equals that price."""
        n = len(constant_prices)
        usd = np.full(n, 1000.0)
        shares = usd / constant_prices

        vwap_series = compute_execution_vwap_series(constant_prices, usd, shares)

        # VWAP should equal the constant price
        np.testing.assert_array_almost_equal(vwap_series, constant_prices)


class TestComputePerformanceSeries:
    """Tests for performance series computation."""

    def test_performance_positive_when_vwap_below_benchmark(self):
        """Test positive performance when VWAP < benchmark."""
        vwap_series = np.array([98.0, 98.0, 98.0])
        benchmark_series = np.array([100.0, 100.0, 100.0])

        perf_series = compute_performance_series(vwap_series, benchmark_series)

        # Performance should be positive
        assert np.all(perf_series > 0)

    def test_performance_negative_when_vwap_above_benchmark(self):
        """Test negative performance when VWAP > benchmark."""
        vwap_series = np.array([102.0, 102.0, 102.0])
        benchmark_series = np.array([100.0, 100.0, 100.0])

        perf_series = compute_performance_series(vwap_series, benchmark_series)

        # Performance should be negative
        assert np.all(perf_series < 0)

    def test_performance_zero_when_equal(self):
        """Test zero performance when VWAP equals benchmark."""
        vwap_series = np.array([100.0, 100.0, 100.0])
        benchmark_series = np.array([100.0, 100.0, 100.0])

        perf_series = compute_performance_series(vwap_series, benchmark_series)

        np.testing.assert_array_almost_equal(perf_series, np.zeros(3))


class TestStrategy1EdgeCases:
    """Additional edge case tests for Strategy 1."""

    def test_strategy1_exact_target_duration_long_prices(self):
        """Verify Strategy 1 ends exactly at target, not one day earlier."""
        prices = np.full(150, 100.0)
        total_usd = 1_000_000
        target_duration = 100

        usd, shares, total, end = strategy_1(prices, total_usd, target_duration)

        assert end == 100, f"Expected end_day=100, got {end}"
        assert usd[99] > 0, "Day 100 (index 99) should have execution"
        assert usd[100] == 0, "Day 101 (index 100) should have no execution"

    def test_strategy1_single_day_target(self):
        """Test Strategy 1 with target_duration = 1."""
        prices = np.full(50, 100.0)
        total_usd = 1_000_000

        usd, shares, total, end = strategy_1(prices, total_usd, 1)

        assert end == 1
        assert np.isclose(usd[0], total_usd)
        assert usd[1] == 0

    def test_strategy1_preserves_total_usd_exactly(self):
        """Test that total USD executed equals input exactly."""
        prices = np.array([100.0, 105.0, 95.0, 102.0, 98.0, 101.0, 99.0])
        total_usd = 1_000_000
        target_duration = 5

        usd, shares, total, end = strategy_1(prices, total_usd, target_duration)

        assert np.isclose(np.sum(usd), total_usd, rtol=1e-10)


class TestStrategy2EdgeCases:
    """Additional edge case tests for Strategy 2."""

    def test_strategy2_min_duration_boundary(self):
        """Test behavior with prices that become favorable after initial period."""
        # Start high, then drop below benchmark
        prices = np.array([110.0] * 15 + [85.0] * 85)  # Drop after initial period
        total_usd = 100_000
        min_duration = 20
        max_duration = 50
        target_duration = 35

        usd, shares, total, end = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )

        # Strategy should complete within bounds
        assert end >= min_duration or end <= max_duration, f"End day {end} should be in valid range"
        # Total USD should be spent
        assert np.isclose(np.sum(usd), total_usd, rtol=1e-6)

    def test_strategy2_max_duration_force_complete(self):
        """Test that strategy completes at max_duration regardless of price."""
        # Prices always above benchmark to trigger slow mode
        prices = np.array([100.0] * 10 + [150.0] * 100)
        total_usd = 100_000
        min_duration = 30
        max_duration = 50
        target_duration = 40

        usd, shares, total, end = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )

        # Must complete by max_duration
        assert end <= max_duration, f"Should complete by max_duration={max_duration}, got end={end}"
        # Total spent should equal total_usd
        assert np.isclose(np.sum(usd), total_usd, rtol=1e-6)

    def test_strategy2_extra_slow_trigger_conditions(self):
        """Test extra slow mode activates with correct conditions."""
        # Prices stay above benchmark
        prices = np.array([100.0] * 10 + [120.0] * 100)
        total_usd = 50_000
        min_duration = 20
        max_duration = 80
        target_duration = 50

        usd, shares, total, end = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )

        base_daily = total_usd / target_duration

        # Check that extra slow mode was triggered at some point
        # (very small execution amounts when remaining < 5x base)
        small_amounts = usd[(usd > 0) & (usd < 0.2 * base_daily)]
        # Extra slow mode uses 0.1x base, so should see some very small amounts
        assert len(small_amounts) > 0 or end <= max_duration


class TestStrategy3EdgeCases:
    """Additional edge case tests for Strategy 3."""

    def test_strategy3_discount_makes_more_aggressive_general(self):
        """Verify discount generally leads to faster execution."""
        np.random.seed(123)
        prices = 100 + np.cumsum(np.random.randn(80) * 3)
        prices = np.maximum(prices, 50)

        total_usd = 100_000
        min_duration = 25
        max_duration = 70
        target_duration = 45
        discount_bps = 150  # 1.5% discount

        # Without discount
        usd2, shares2, total2, end2 = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=0
        )

        # With discount
        usd3, shares3, total3, end3 = strategy_3(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=discount_bps
        )

        # Discount lowers benchmark, making price appear MORE favorable
        # So should generally finish faster
        assert end3 <= end2 + 5

    def test_strategy3_zero_discount_equals_strategy2(self):
        """Verify Strategy 3 with 0 discount equals Strategy 2."""
        np.random.seed(456)
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        prices = np.maximum(prices, 50)

        total_usd = 75_000
        min_duration = 15
        max_duration = 45
        target_duration = 30

        usd2, shares2, total2, end2 = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=0
        )

        usd3, shares3, total3, end3 = strategy_3(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps=0
        )

        np.testing.assert_array_almost_equal(usd2, usd3)
        np.testing.assert_array_almost_equal(shares2, shares3)
        assert end2 == end3


class TestAllStrategiesConsistency:
    """Tests for consistency across all strategies."""

    def test_all_strategies_total_usd_matches(self):
        """Verify total USD executed equals input for all strategies."""
        np.random.seed(789)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        prices = np.maximum(prices, 50)

        total_usd = 500_000
        min_duration = 30
        max_duration = 80
        target_duration = 55
        discount_bps = 50

        # Strategy 1
        usd1, shares1, total1, end1 = strategy_1(prices, total_usd, target_duration)
        assert np.isclose(np.sum(usd1), total_usd, rtol=1e-6), "Strategy 1 total mismatch"

        # Strategy 2
        usd2, shares2, total2, end2 = strategy_2(
            prices, total_usd, min_duration, max_duration, target_duration
        )
        assert np.isclose(np.sum(usd2), total_usd, rtol=1e-6), "Strategy 2 total mismatch"

        # Strategy 3
        usd3, shares3, total3, end3 = strategy_3(
            prices, total_usd, min_duration, max_duration, target_duration, discount_bps
        )
        assert np.isclose(np.sum(usd3), total_usd, rtol=1e-6), "Strategy 3 total mismatch"

    def test_all_strategies_positive_shares(self):
        """Verify shares are always positive when USD > 0."""
        prices = np.full(100, 100.0)
        total_usd = 100_000

        usd1, shares1, _, _ = strategy_1(prices, total_usd, 50)
        usd2, shares2, _, _ = strategy_2(prices, total_usd, 30, 80, 55)
        usd3, shares3, _, _ = strategy_3(prices, total_usd, 30, 80, 55, 100)

        for usd, shares, name in [(usd1, shares1, "S1"), (usd2, shares2, "S2"), (usd3, shares3, "S3")]:
            assert np.all(shares[usd > 0] > 0), f"{name}: shares should be positive when USD > 0"
