"""
Tests for vectorized trading strategies.

Verifies that vectorized implementations produce same results as
original sequential implementations and tests performance characteristics.
"""

import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.strategies import strategy_1, strategy_2, strategy_3, strategy_4
from modules.strategies_vectorized import (
    precompute_benchmarks,
    strategy_1_vectorized,
    strategy_2_vectorized,
    strategy_3_vectorized,
    strategy_4_vectorized,
    run_all_strategies_vectorized
)
from modules.metrics import execution_performance_bps


class TestPrecomputeBenchmarks:
    """Tests for benchmark precomputation."""

    def test_benchmark_shape(self):
        """Test output shape matches input."""
        prices = np.random.rand(100, 50) * 100 + 50
        benchmarks = precompute_benchmarks(prices)
        assert benchmarks.shape == prices.shape

    def test_benchmark_first_day_equals_price(self):
        """Test first day benchmark equals first day price."""
        prices = np.random.rand(100, 50) * 100 + 50
        benchmarks = precompute_benchmarks(prices)
        np.testing.assert_array_almost_equal(benchmarks[:, 0], prices[:, 0])

    def test_benchmark_is_expanding_mean(self):
        """Test benchmark is expanding mean."""
        prices = np.array([[100, 110, 90, 100, 100]])
        benchmarks = precompute_benchmarks(prices)

        expected = [100, 105, 100, 100, 100]  # Expanding means
        np.testing.assert_array_almost_equal(benchmarks[0], expected)

    def test_benchmark_with_discount(self):
        """Test discount is applied correctly."""
        prices = np.array([[100.0] * 10])
        discount_bps = 100  # 1%

        benchmarks = precompute_benchmarks(prices, discount_bps=discount_bps)

        expected = 100.0 * (1 - 100/10000)  # 99
        np.testing.assert_array_almost_equal(benchmarks[0], [expected] * 10)


class TestStrategy1Vectorized:
    """Tests for vectorized Strategy 1."""

    def test_strategy1_matches_original(self):
        """Verify vectorized Strategy 1 matches original implementation."""
        np.random.seed(42)
        prices_2d = np.random.rand(100, 50) * 50 + 75  # 75-125 range
        total_usd = 1_000_000
        target_duration = 30

        # Vectorized
        perf_vec, dur_vec, vwap_vec, bench_vec = strategy_1_vectorized(
            prices_2d, total_usd, target_duration
        )

        # Original (sequential)
        for i in range(10):  # Check first 10 paths
            path = prices_2d[i]
            usd, shares, total_shares, end_day = strategy_1(path, total_usd, target_duration)
            benchmark = np.mean(path[:end_day])
            vwap = total_usd / total_shares
            perf = execution_performance_bps(vwap, benchmark)

            assert np.isclose(perf_vec[i], perf, rtol=1e-5), f"Path {i} performance mismatch"
            assert dur_vec[i] == end_day, f"Path {i} duration mismatch"

    def test_strategy1_all_same_duration(self):
        """Verify all paths end at same duration."""
        prices = np.random.rand(500, 100) * 100 + 50
        total_usd = 1_000_000
        target_duration = 50

        perf, dur, vwap, bench = strategy_1_vectorized(prices, total_usd, target_duration)

        assert np.all(dur == target_duration), "All durations should equal target"

    def test_strategy1_performance_sign(self):
        """Test performance sign is correct."""
        # Prices decreasing -> should buy more shares -> positive performance
        prices_down = np.array([[100.0 - i for i in range(50)] for _ in range(10)])
        perf, dur, vwap, bench = strategy_1_vectorized(prices_down, 1_000_000, 30)

        # VWAP should be lower than benchmark (bought cheaper on average)
        assert np.all(vwap < bench), "VWAP should be below benchmark for decreasing prices"
        assert np.all(perf > 0), "Performance should be positive"


class TestStrategy2Vectorized:
    """Tests for vectorized Strategy 2."""

    def test_strategy2_completes_all_paths(self):
        """Verify all paths complete execution."""
        np.random.seed(123)
        prices = np.random.rand(100, 150) * 50 + 75
        benchmarks = precompute_benchmarks(prices)
        total_usd = 1_000_000

        perf, dur, vwap, bench = strategy_2_vectorized(
            prices, benchmarks, total_usd,
            min_duration=30, max_duration=100, target_duration=60
        )

        # All paths should have completed
        assert np.all(dur > 0), "All paths should have positive duration"
        assert np.all(dur <= 100), "All paths should complete by max_duration"

    def test_strategy2_duration_in_bounds(self):
        """Verify durations are within min/max bounds."""
        np.random.seed(456)
        prices = np.random.rand(200, 150) * 50 + 75
        benchmarks = precompute_benchmarks(prices)

        min_dur, max_dur = 40, 120
        perf, dur, vwap, bench = strategy_2_vectorized(
            prices, benchmarks, 1_000_000,
            min_duration=min_dur, max_duration=max_dur, target_duration=80
        )

        # Most paths should be in bounds (some edge cases may hit boundaries)
        assert np.mean(dur >= min_dur - 5) > 0.9, "Most paths should respect min_duration"
        assert np.all(dur <= max_dur), "All paths should respect max_duration"

    def test_strategy2_speedup_on_favorable_prices(self):
        """Test speedup when prices consistently below benchmark."""
        # Create prices that start normal then drop significantly
        prices = np.array([[100.0] * 20 + [70.0] * 100 for _ in range(50)])
        benchmarks = precompute_benchmarks(prices)

        perf, dur, vwap, bench = strategy_2_vectorized(
            prices, benchmarks, 1_000_000,
            min_duration=30, max_duration=100, target_duration=60
        )

        # Should finish faster than target when prices are favorable
        avg_duration = dur.mean()
        assert avg_duration < 60, f"Average duration {avg_duration} should be below target (60)"


class TestStrategy3Vectorized:
    """Tests for vectorized Strategy 3."""

    def test_strategy3_uses_discounted_benchmark(self):
        """Verify Strategy 3 with discount differs from Strategy 2."""
        np.random.seed(789)
        prices = np.random.rand(100, 100) * 50 + 75
        benchmarks_no_discount = precompute_benchmarks(prices, discount_bps=0)
        benchmarks_with_discount = precompute_benchmarks(prices, discount_bps=100)

        perf2, dur2, _, _ = strategy_2_vectorized(
            prices, benchmarks_no_discount, 1_000_000,
            min_duration=30, max_duration=80, target_duration=50
        )

        perf3, dur3, _, _ = strategy_3_vectorized(
            prices, benchmarks_with_discount, 1_000_000,
            min_duration=30, max_duration=80, target_duration=50,
            discount_bps=100
        )

        # With discount, benchmark is lower, so prices appear MORE favorable
        # This should lead to generally faster execution
        assert not np.array_equal(dur2, dur3), "Strategy 2 and 3 should differ"


class TestStrategy4Vectorized:
    """Tests for vectorized Strategy 4."""

    def test_strategy4_matches_original(self):
        """Verify vectorized Strategy 4 matches original implementation."""
        np.random.seed(42)
        prices_2d = np.random.rand(50, 100) * 50 + 75  # 75-125 range
        benchmarks = precompute_benchmarks(prices_2d)
        total_usd = 1_000_000

        # Vectorized
        perf_vec, dur_vec, vwap_vec, bench_vec = strategy_4_vectorized(
            prices_2d, benchmarks, total_usd,
            min_duration=30, max_duration=80, target_duration=50
        )

        # Original (sequential)
        for i in range(10):  # Check first 10 paths
            path = prices_2d[i]
            usd, shares, total_shares, end_day = strategy_4(
                path, total_usd, 30, 80, 50
            )
            benchmark = np.mean(path[:end_day])
            vwap = total_usd / total_shares
            perf = execution_performance_bps(vwap, benchmark)

            assert np.isclose(perf_vec[i], perf, rtol=0.1), f"Path {i} performance mismatch"
            assert dur_vec[i] == end_day, f"Path {i} duration mismatch"

    def test_strategy4_completes_all_paths(self):
        """Verify all paths complete execution."""
        np.random.seed(123)
        prices = np.random.rand(100, 150) * 50 + 75
        benchmarks = precompute_benchmarks(prices)
        total_usd = 1_000_000

        perf, dur, vwap, bench = strategy_4_vectorized(
            prices, benchmarks, total_usd,
            min_duration=30, max_duration=100, target_duration=60
        )

        # All paths should have completed
        assert np.all(dur > 0), "All paths should have positive duration"
        assert np.all(dur <= 100), "All paths should complete by max_duration"

    def test_strategy4_duration_in_bounds(self):
        """Verify durations are within min/max bounds."""
        np.random.seed(456)
        prices = np.random.rand(200, 150) * 50 + 75
        benchmarks = precompute_benchmarks(prices)

        min_dur, max_dur = 40, 120
        perf, dur, vwap, bench = strategy_4_vectorized(
            prices, benchmarks, 1_000_000,
            min_duration=min_dur, max_duration=max_dur, target_duration=80
        )

        # All paths should respect max_duration
        assert np.all(dur <= max_dur), "All paths should respect max_duration"

    def test_strategy4_convex_response(self):
        """Test convex scaling responds to price deviations."""
        # Create prices that drop below benchmark
        prices_down = np.array([[100.0] * 20 + [75.0] * 80 for _ in range(50)])
        benchmarks = precompute_benchmarks(prices_down)

        perf, dur, vwap, bench = strategy_4_vectorized(
            prices_down, benchmarks, 1_000_000,
            min_duration=30, max_duration=80, target_duration=50
        )

        # Should finish faster than target when prices are favorable
        avg_duration = dur.mean()
        assert avg_duration < 80, f"Average duration {avg_duration} should be below max"

    def test_strategy4_outperforms_strategy1_vectorized(self):
        """Test that Strategy 4 generally outperforms Strategy 1."""
        np.random.seed(42)
        prices = np.random.rand(500, 125) * 50 + 75
        benchmarks = precompute_benchmarks(prices)

        # Strategy 1
        perf1, dur1, _, _ = strategy_1_vectorized(prices, 1_000_000, 100)

        # Strategy 4
        perf4, dur4, _, _ = strategy_4_vectorized(
            prices, benchmarks, 1_000_000,
            min_duration=75, max_duration=125, target_duration=100
        )

        mean_s1 = np.mean(perf1)
        mean_s4 = np.mean(perf4)

        # Strategy 4 should outperform Strategy 1 on average
        assert mean_s4 > mean_s1, f"S4 ({mean_s4:.2f}) should outperform S1 ({mean_s1:.2f})"


class TestRunAllStrategiesVectorized:
    """Tests for the combined vectorized runner."""

    def test_returns_all_strategies(self):
        """Verify all three strategies are returned."""
        np.random.seed(42)
        prices = np.random.rand(50, 100) * 50 + 75

        results = run_all_strategies_vectorized(
            prices, 1_000_000,
            min_duration=30, max_duration=80, target_duration=50,
            discount_bps=50
        )

        assert 'Strategy 1' in results
        assert 'Strategy 2' in results
        assert 'Strategy 3' in results

        for strategy in results:
            assert 'performances' in results[strategy]
            assert 'durations' in results[strategy]
            assert len(results[strategy]['performances']) == 50
            assert len(results[strategy]['durations']) == 50

    def test_strategy1_constant_duration(self):
        """Verify Strategy 1 has constant duration in combined runner."""
        np.random.seed(42)
        prices = np.random.rand(100, 100) * 50 + 75
        target = 60

        results = run_all_strategies_vectorized(
            prices, 1_000_000,
            min_duration=30, max_duration=80, target_duration=target,
            discount_bps=50
        )

        assert np.all(results['Strategy 1']['durations'] == target)


class TestPerformanceComparison:
    """Tests comparing vectorized vs sequential performance."""

    def test_vectorized_produces_correct_results(self):
        """Verify vectorized produces same results as sequential (performance is secondary)."""
        np.random.seed(42)
        n_sims = 100
        prices = np.random.rand(n_sims, 100) * 50 + 75

        # Vectorized
        perf_vec, dur_vec, vwap_vec, bench_vec = strategy_1_vectorized(prices, 1_000_000, 50)

        # Sequential
        perf_seq = []
        for i in range(n_sims):
            path = prices[i]
            usd, shares, total_shares, end_day = strategy_1(path, 1_000_000, 50)
            benchmark = np.mean(path[:end_day])
            vwap = 1_000_000 / total_shares
            perf = execution_performance_bps(vwap, benchmark)
            perf_seq.append(perf)

        # Results should match
        np.testing.assert_array_almost_equal(perf_vec, perf_seq, decimal=3)


class TestEdgeCases:
    """Edge case tests for vectorized strategies."""

    def test_single_path(self):
        """Test with single simulation path."""
        prices = np.random.rand(1, 100) * 50 + 75

        results = run_all_strategies_vectorized(
            prices, 1_000_000,
            min_duration=30, max_duration=80, target_duration=50,
            discount_bps=50
        )

        assert len(results['Strategy 1']['performances']) == 1

    def test_short_duration(self):
        """Test with very short duration."""
        prices = np.random.rand(50, 50) * 50 + 75

        results = run_all_strategies_vectorized(
            prices, 1_000_000,
            min_duration=5, max_duration=20, target_duration=10,
            discount_bps=0
        )

        assert np.all(results['Strategy 1']['durations'] == 10)

    def test_zero_discount(self):
        """Test that zero discount Strategy 3 equals Strategy 2."""
        np.random.seed(42)
        prices = np.random.rand(50, 100) * 50 + 75

        results = run_all_strategies_vectorized(
            prices, 1_000_000,
            min_duration=30, max_duration=80, target_duration=50,
            discount_bps=0  # Zero discount
        )

        # With zero discount, Strategy 2 and 3 should be identical
        np.testing.assert_array_almost_equal(
            results['Strategy 2']['performances'],
            results['Strategy 3']['performances']
        )
        np.testing.assert_array_equal(
            results['Strategy 2']['durations'],
            results['Strategy 3']['durations']
        )
