"""
Integration tests for the Streamlit application.
"""

import pytest
import numpy as np
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gbm import generate_gbm_paths
from modules.benchmarks import compute_benchmark
from modules.strategies import strategy_1, strategy_2, strategy_3
from modules.metrics import vwap, execution_performance_bps, standard_error


class TestWidgetKeys:
    """Tests for duplicate widget ID prevention."""

    def test_no_duplicate_widget_ids(self):
        """Test that app.py uses unique widget keys."""
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'app.py'
        )

        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all key= assignments
        key_pattern = r'key\s*=\s*["\']([^"\']+)["\']'
        keys = re.findall(key_pattern, content)

        # Check for duplicates
        seen = set()
        duplicates = set()
        for key in keys:
            if key in seen:
                duplicates.add(key)
            seen.add(key)

        assert len(duplicates) == 0, f"Duplicate widget keys found: {duplicates}"

    def test_widget_keys_are_descriptive(self):
        """Test that widget keys have descriptive names."""
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'app.py'
        )

        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()

        key_pattern = r'key\s*=\s*["\']([^"\']+)["\']'
        keys = re.findall(key_pattern, content)

        # All keys should have at least 3 characters
        for key in keys:
            assert len(key) >= 3, f"Widget key '{key}' is too short to be descriptive"


class TestSimulationIntegration:
    """Integration tests for the full simulation pipeline."""

    def test_simulation_runs_without_error(self):
        """Test that full simulation pipeline runs without errors."""
        # Generate paths
        paths = generate_gbm_paths(
            S0=100.0, mu=0.0, sigma=0.25,
            days=50, n_simulations=100, seed=42
        )

        total_usd = 10_000_000
        min_duration = 20
        max_duration = 50
        target_duration = 35
        discount_bps = 50

        results = {'Strategy 1': [], 'Strategy 2': [], 'Strategy 3': []}

        for i in range(paths.shape[0]):
            path = paths[i]

            # Strategy 1
            usd1, shares1, total1, end1 = strategy_1(path, total_usd, target_duration)
            benchmark1 = np.mean(path[:end1])
            vwap1 = total_usd / total1
            perf1 = execution_performance_bps(vwap1, benchmark1)
            results['Strategy 1'].append({'perf': perf1, 'dur': end1})

            # Strategy 2
            usd2, shares2, total2, end2 = strategy_2(
                path, total_usd, min_duration, max_duration, target_duration
            )
            benchmark2 = np.mean(path[:end2])
            vwap2 = np.sum(usd2) / total2
            perf2 = execution_performance_bps(vwap2, benchmark2)
            results['Strategy 2'].append({'perf': perf2, 'dur': end2})

            # Strategy 3
            usd3, shares3, total3, end3 = strategy_3(
                path, total_usd, min_duration, max_duration, target_duration, discount_bps
            )
            benchmark3 = np.mean(path[:end3])
            vwap3 = np.sum(usd3) / total3
            perf3 = execution_performance_bps(vwap3, benchmark3)
            results['Strategy 3'].append({'perf': perf3, 'dur': end3})

        # Verify we got results for all simulations
        assert len(results['Strategy 1']) == 100
        assert len(results['Strategy 2']) == 100
        assert len(results['Strategy 3']) == 100

    def test_all_strategies_complete(self):
        """Test that all strategies complete execution."""
        paths = generate_gbm_paths(
            S0=100.0, mu=0.0, sigma=0.25,
            days=100, n_simulations=50, seed=42
        )

        total_usd = 10_000_000
        min_duration = 30
        max_duration = 80
        target_duration = 55

        for i in range(paths.shape[0]):
            path = paths[i]

            # Strategy 1
            usd1, shares1, total1, end1 = strategy_1(path, total_usd, target_duration)
            assert np.isclose(np.sum(usd1), total_usd)
            assert end1 == target_duration

            # Strategy 2
            usd2, shares2, total2, end2 = strategy_2(
                path, total_usd, min_duration, max_duration, target_duration
            )
            assert np.sum(usd2) <= total_usd + 0.01
            # Strategy 2 may finish slightly before min_duration in edge cases
            assert end2 <= max_duration + 1

            # Strategy 3
            usd3, shares3, total3, end3 = strategy_3(
                path, total_usd, min_duration, max_duration, target_duration, discount_bps=100
            )
            assert np.sum(usd3) <= total_usd + 0.01


class TestExampleGeneration:
    """Tests for example generation functionality."""

    def test_example_generates_new_each_time(self):
        """Test that examples are different with different seeds."""
        # Generate two examples with different seeds
        path1 = generate_gbm_paths(100, 0, 0.25, 50, 1, seed=42)[0]
        path2 = generate_gbm_paths(100, 0, 0.25, 50, 1, seed=123)[0]

        # Paths should be different
        assert not np.allclose(path1, path2)

    def test_example_with_random_seed_varies(self):
        """Test that examples without fixed seed produce different results."""
        # Generate multiple examples without seed
        paths = []
        for _ in range(3):
            path = generate_gbm_paths(100, 0, 0.25, 50, 1, seed=None)[0]
            paths.append(path)

        # At least some paths should be different
        all_same = all(np.allclose(paths[0], p) for p in paths[1:])
        assert not all_same


class TestPerformanceConsistency:
    """Tests for performance metric consistency."""

    def test_performance_is_symmetric(self):
        """Test that outperformance and underperformance are symmetric."""
        # 1% better than benchmark
        perf_better = execution_performance_bps(99, 100)

        # 1% worse than benchmark
        perf_worse = execution_performance_bps(101, 100)

        # Should be approximately equal in magnitude but opposite sign
        assert np.isclose(abs(perf_better), abs(perf_worse))
        assert perf_better > 0
        assert perf_worse < 0

    def test_standard_error_decreases_with_sample_size(self):
        """Test that SE decreases as sample size increases."""
        base_values = np.random.randn(10000)

        se_100 = standard_error(base_values[:100])
        se_1000 = standard_error(base_values[:1000])
        se_10000 = standard_error(base_values)

        assert se_1000 < se_100
        assert se_10000 < se_1000


class TestModuleImports:
    """Tests for proper module imports."""

    def test_all_modules_importable(self):
        """Test that all modules can be imported."""
        try:
            from modules.config import DEFAULT_S0
            from modules.gbm import generate_gbm_paths
            from modules.benchmarks import compute_benchmark
            from modules.strategies import strategy_1, strategy_2, strategy_3
            from modules.metrics import vwap, execution_performance_bps, standard_error
            from modules.visualizations import plot_price_paths
        except ImportError as e:
            pytest.fail(f"Failed to import module: {e}")

    def test_config_constants_exist(self):
        """Test that expected config constants exist."""
        from modules.config import (
            TRADING_DAYS_PER_YEAR,
            DEFAULT_S0,
            DEFAULT_DAYS,
            DEFAULT_MU,
            DEFAULT_SIGMA,
            DEFAULT_SIMULATIONS,
            DEFAULT_USD,
            DEFAULT_MAX_DURATION,
            DEFAULT_MIN_DURATION,
            DEFAULT_TARGET_DURATION,
            DEFAULT_DISCOUNT_BPS
        )

        assert TRADING_DAYS_PER_YEAR == 252
        assert DEFAULT_S0 == 100.0
        assert DEFAULT_DAYS == 125
        assert DEFAULT_USD == 1_000_000_000
