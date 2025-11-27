"""
Tests for the visualizations module.
"""

import pytest
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.visualizations import (
    plot_price_paths, plot_performance_histogram, plot_duration_histogram,
    plot_example_timeseries, plot_daily_usd, create_summary_table
)
from modules.gbm import generate_gbm_paths


class TestPlotPricePaths:
    """Tests for price paths visualization."""

    def test_plot_price_paths_returns_figure(self):
        """Test that function returns a Plotly figure."""
        paths = generate_gbm_paths(100, 0, 0.25, 50, 100, seed=42)
        fig = plot_price_paths(paths, S0=100, mu=0, sigma=0.25)

        assert isinstance(fig, go.Figure)

    def test_plot_price_paths_has_data(self):
        """Test that figure has data traces."""
        paths = generate_gbm_paths(100, 0, 0.25, 50, 100, seed=42)
        fig = plot_price_paths(paths, S0=100, mu=0, sigma=0.25, show_n=10)

        # Should have paths + mean + sigma envelopes
        assert len(fig.data) > 10

    def test_plot_price_paths_custom_title(self):
        """Test custom title is applied."""
        paths = generate_gbm_paths(100, 0, 0.25, 50, 100, seed=42)
        title = "My Custom Title"
        fig = plot_price_paths(paths, S0=100, mu=0, sigma=0.25, title=title)

        assert fig.layout.title.text == title


class TestPlotPerformanceHistogram:
    """Tests for performance histogram visualization."""

    def test_plot_performance_histogram_returns_figure(self):
        """Test that function returns a Plotly figure."""
        performance = np.random.randn(1000) * 50  # Random performance in bps
        fig = plot_performance_histogram(performance, "Test Strategy")

        assert isinstance(fig, go.Figure)

    def test_plot_performance_histogram_has_bar_trace(self):
        """Test that histogram has bar trace."""
        performance = np.random.randn(1000) * 50
        fig = plot_performance_histogram(performance, "Test Strategy")

        # Should have at least one bar trace
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    def test_plot_performance_histogram_custom_bins(self):
        """Test custom bin count."""
        performance = np.random.randn(1000) * 50
        fig = plot_performance_histogram(performance, "Test Strategy", bins=100)

        # Should create figure without error
        assert isinstance(fig, go.Figure)


class TestPlotDurationHistogram:
    """Tests for duration histogram visualization."""

    def test_plot_duration_histogram_returns_figure(self):
        """Test that function returns a Plotly figure."""
        durations = np.random.randint(75, 126, size=1000)
        fig = plot_duration_histogram(
            durations, "Test Strategy",
            min_duration=75, target_duration=100, max_duration=125
        )

        assert isinstance(fig, go.Figure)

    def test_plot_duration_histogram_has_vertical_lines(self):
        """Test that histogram has vertical lines for min/target/max."""
        durations = np.random.randint(75, 126, size=1000)
        fig = plot_duration_histogram(
            durations, "Test Strategy",
            min_duration=75, target_duration=100, max_duration=125
        )

        # Should have vertical line shapes
        assert len(fig.layout.shapes) > 0


class TestPlotExampleTimeseries:
    """Tests for example timeseries visualization."""

    def test_plot_example_timeseries_returns_figure(self):
        """Test that function returns a Plotly figure."""
        n = 50
        prices = 100 + np.cumsum(np.random.randn(n))
        benchmark = np.cumsum(prices) / np.arange(1, n + 1)
        exec_price_series = benchmark + np.random.randn(n) * 0.5
        performance_series = -((exec_price_series - benchmark) / benchmark) * 10000

        fig = plot_example_timeseries(
            prices, benchmark, exec_price_series, performance_series,
            end_day=40, min_duration=30, target_duration=40, max_duration=50,
            strategy_name="Test Strategy"
        )

        assert isinstance(fig, go.Figure)

    def test_plot_example_timeseries_has_dual_axis(self):
        """Test that figure has dual y-axis."""
        n = 50
        prices = 100 + np.cumsum(np.random.randn(n))
        benchmark = np.cumsum(prices) / np.arange(1, n + 1)
        exec_price_series = benchmark + np.random.randn(n) * 0.5
        performance_series = -((exec_price_series - benchmark) / benchmark) * 10000

        fig = plot_example_timeseries(
            prices, benchmark, exec_price_series, performance_series,
            end_day=40, min_duration=30, target_duration=40, max_duration=50,
            strategy_name="Test Strategy"
        )

        # Should have secondary y-axis
        assert 'yaxis2' in fig.layout


class TestPlotDailyUSD:
    """Tests for daily USD bar chart visualization."""

    def test_plot_daily_usd_returns_figure(self):
        """Test that function returns a Plotly figure."""
        usd_per_day = np.array([1e6] * 30 + [0] * 20)
        fig = plot_daily_usd(
            usd_per_day, "Test Strategy",
            end_day=30, min_duration=20, target_duration=30, max_duration=50
        )

        assert isinstance(fig, go.Figure)

    def test_plot_daily_usd_has_bar_trace(self):
        """Test that chart has bar trace."""
        usd_per_day = np.array([1e6] * 30 + [0] * 20)
        fig = plot_daily_usd(
            usd_per_day, "Test Strategy",
            end_day=30, min_duration=20, target_duration=30, max_duration=50
        )

        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1


class TestCreateSummaryTable:
    """Tests for summary table creation."""

    def test_create_summary_table_returns_figure(self):
        """Test that function returns a Plotly figure with table."""
        results = {
            'Strategy 1': {'mean_perf': 10.5, 'se_perf': 1.2, 'mean_dur': 100, 'se_dur': 0.5},
            'Strategy 2': {'mean_perf': 15.3, 'se_perf': 2.1, 'mean_dur': 95, 'se_dur': 5.2},
            'Strategy 3': {'mean_perf': 18.7, 'se_perf': 2.5, 'mean_dur': 88, 'se_dur': 7.1}
        }

        fig = create_summary_table(results)

        assert isinstance(fig, go.Figure)

    def test_create_summary_table_has_table_trace(self):
        """Test that figure has table trace."""
        results = {
            'Strategy 1': {'mean_perf': 10.5, 'se_perf': 1.2, 'mean_dur': 100, 'se_dur': 0.5},
            'Strategy 2': {'mean_perf': 15.3, 'se_perf': 2.1, 'mean_dur': 95, 'se_dur': 5.2}
        }

        fig = create_summary_table(results)

        table_traces = [t for t in fig.data if isinstance(t, go.Table)]
        assert len(table_traces) >= 1
