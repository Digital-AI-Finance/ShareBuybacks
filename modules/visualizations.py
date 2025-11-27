"""
Visualization functions for share buyback strategy analysis.

All charts use Plotly for interactive visualizations in Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def plot_price_paths(
    paths: np.ndarray,
    S0: float,
    mu: float,
    sigma: float,
    show_n: int = 100,
    title: str = "Simulated Stock Price Paths"
) -> go.Figure:
    """
    Plot sample GBM paths with theoretical sigma envelopes.

    Parameters
    ----------
    paths : np.ndarray
        Array of shape (n_simulations, days) containing price paths.
    S0 : float
        Initial stock price.
    mu : float
        Annual drift rate.
    sigma : float
        Annual volatility.
    show_n : int, optional
        Number of paths to display. Default is 100.
    title : str, optional
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    n_simulations, n_days = paths.shape
    days = np.arange(1, n_days + 1)

    # Compute sigma envelopes
    dt = 1.0 / 252
    t = np.arange(n_days) * dt
    mean = S0 * np.exp(mu * t)
    variance_factor = np.exp(sigma ** 2 * t) - 1
    std = mean * np.sqrt(np.maximum(variance_factor, 0))

    fig = go.Figure()

    # Select random paths to show
    if n_simulations > show_n:
        indices = np.random.choice(n_simulations, show_n, replace=False)
    else:
        indices = range(n_simulations)

    # Plot individual paths
    for i in indices:
        fig.add_trace(go.Scatter(
            x=days,
            y=paths[i],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 100, 200, 0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Plot mean
    fig.add_trace(go.Scatter(
        x=days,
        y=mean,
        mode='lines',
        line=dict(width=2, color='black'),
        name='Expected Value'
    ))

    # Plot sigma envelopes
    colors = ['rgba(255, 0, 0, 0.3)', 'rgba(255, 100, 0, 0.3)',
              'rgba(255, 150, 0, 0.3)', 'rgba(255, 200, 0, 0.3)']
    for i, n_sigma in enumerate([1, 2, 3, 4]):
        upper = mean + n_sigma * std
        lower = mean - n_sigma * std

        fig.add_trace(go.Scatter(
            x=days,
            y=upper,
            mode='lines',
            line=dict(width=1, color=colors[i], dash='dash'),
            name=f'+{n_sigma} sigma'
        ))
        fig.add_trace(go.Scatter(
            x=days,
            y=lower,
            mode='lines',
            line=dict(width=1, color=colors[i], dash='dash'),
            name=f'-{n_sigma} sigma',
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Trading Day",
        yaxis_title="Stock Price ($)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )

    return fig


def plot_performance_histogram(
    performance_bps: np.ndarray,
    strategy_name: str,
    bins: int = 250,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot histogram of execution performance in basis points.

    Parameters
    ----------
    performance_bps : np.ndarray
        Array of performance values in basis points.
    strategy_name : str
        Name of the strategy for labeling.
    bins : int, optional
        Number of histogram bins. Default is 250.
    title : str, optional
        Custom title. If None, auto-generated.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{strategy_name}: Performance Distribution"

    mean_perf = np.mean(performance_bps)
    std_perf = np.std(performance_bps)

    # Create histogram with percentage y-axis
    counts, bin_edges = np.histogram(performance_bps, bins=bins)
    percentages = counts / len(performance_bps) * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=percentages,
        width=np.diff(bin_edges),
        marker_color='steelblue',
        name='Frequency'
    ))

    # Add vertical line for mean
    fig.add_vline(
        x=mean_perf,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_perf:.1f} bps",
        annotation_position="top right"
    )

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="black",
        line_width=1
    )

    fig.update_layout(
        title=title,
        xaxis_title="Performance (bps)",
        yaxis_title="Frequency (%)",
        showlegend=False,
        height=400
    )

    return fig


def plot_duration_histogram(
    durations: np.ndarray,
    strategy_name: str,
    min_duration: int,
    target_duration: int,
    max_duration: int,
    bins: int = 250,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot histogram of execution durations.

    Parameters
    ----------
    durations : np.ndarray
        Array of execution durations (days).
    strategy_name : str
        Name of the strategy for labeling.
    min_duration : int
        Minimum allowed duration.
    target_duration : int
        Target duration.
    max_duration : int
        Maximum allowed duration.
    bins : int, optional
        Number of histogram bins. Default is 250.
    title : str, optional
        Custom title. If None, auto-generated.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{strategy_name}: Duration Distribution"

    mean_dur = np.mean(durations)

    # Create histogram with percentage y-axis
    counts, bin_edges = np.histogram(durations, bins=bins)
    percentages = counts / len(durations) * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=percentages,
        width=np.diff(bin_edges),
        marker_color='forestgreen',
        name='Frequency'
    ))

    # Add vertical lines for min, target, max
    fig.add_vline(
        x=min_duration,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Min: {min_duration}",
        annotation_position="top left"
    )

    fig.add_vline(
        x=target_duration,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Target: {target_duration}",
        annotation_position="top"
    )

    fig.add_vline(
        x=max_duration,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Max: {max_duration}",
        annotation_position="top right"
    )

    # Add mean line
    fig.add_vline(
        x=mean_dur,
        line_dash="solid",
        line_color="black",
        line_width=2,
        annotation_text=f"Mean: {mean_dur:.1f}",
        annotation_position="bottom"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Duration (days)",
        yaxis_title="Frequency (%)",
        showlegend=False,
        height=400
    )

    return fig


def plot_example_timeseries(
    prices: np.ndarray,
    benchmark: np.ndarray,
    exec_price_series: np.ndarray,
    performance_series: np.ndarray,
    end_day: int,
    min_duration: int,
    target_duration: int,
    max_duration: int,
    strategy_name: str
) -> go.Figure:
    """
    Plot example execution with dual y-axis for prices and performance.

    Parameters
    ----------
    prices : np.ndarray
        Stock prices over time.
    benchmark : np.ndarray
        Benchmark prices over time.
    exec_price_series : np.ndarray
        Cumulative VWAP over time.
    performance_series : np.ndarray
        Performance in bps over time.
    end_day : int
        Day when execution completed.
    min_duration : int
        Minimum allowed duration.
    target_duration : int
        Target duration.
    max_duration : int
        Maximum allowed duration.
    strategy_name : str
        Name of the strategy.

    Returns
    -------
    go.Figure
        Plotly figure object with dual y-axis.
    """
    days = np.arange(1, len(prices) + 1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot stock price
    fig.add_trace(
        go.Scatter(
            x=days,
            y=prices,
            mode='lines',
            name='Stock Price',
            line=dict(color='gray', width=1)
        ),
        secondary_y=False
    )

    # Plot benchmark
    fig.add_trace(
        go.Scatter(
            x=days[:end_day],
            y=benchmark[:end_day],
            mode='lines',
            name='Benchmark (Avg)',
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )

    # Plot execution VWAP
    valid_exec = ~np.isnan(exec_price_series[:end_day])
    fig.add_trace(
        go.Scatter(
            x=days[:end_day][valid_exec],
            y=exec_price_series[:end_day][valid_exec],
            mode='lines',
            name='Execution VWAP',
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )

    # Plot performance on secondary axis
    valid_perf = ~np.isnan(performance_series[:end_day])
    fig.add_trace(
        go.Scatter(
            x=days[:end_day][valid_perf],
            y=performance_series[:end_day][valid_perf],
            mode='lines',
            name='Performance (bps)',
            line=dict(color='green', width=2, dash='dot')
        ),
        secondary_y=True
    )

    # Add vertical lines for duration markers
    fig.add_vline(x=min_duration, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_vline(x=target_duration, line_dash="dash", line_color="blue", opacity=0.5)
    fig.add_vline(x=max_duration, line_dash="dot", line_color="red", opacity=0.5)

    # Mark actual end day
    fig.add_vline(
        x=end_day,
        line_dash="solid",
        line_color="black",
        line_width=2,
        annotation_text=f"End: Day {end_day}",
        annotation_position="top"
    )

    # Add horizontal line at 0 for performance
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5, secondary_y=True)

    fig.update_layout(
        title=f"{strategy_name}: Execution Example",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.update_xaxes(title_text="Trading Day")
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Performance (bps)", secondary_y=True)

    return fig


def plot_daily_usd(
    usd_per_day: np.ndarray,
    strategy_name: str,
    end_day: int,
    min_duration: int,
    target_duration: int,
    max_duration: int,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot daily USD execution as a bar chart.

    Parameters
    ----------
    usd_per_day : np.ndarray
        USD amount executed each day.
    strategy_name : str
        Name of the strategy.
    end_day : int
        Day when execution completed.
    min_duration : int
        Minimum allowed duration.
    target_duration : int
        Target duration.
    max_duration : int
        Maximum allowed duration.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{strategy_name}: Daily USD Execution"

    days = np.arange(1, end_day + 1)
    usd_millions = usd_per_day[:end_day] / 1e6  # Convert to millions

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=days,
        y=usd_millions,
        marker_color='steelblue',
        name='Daily USD'
    ))

    # Add vertical lines for duration markers
    fig.add_vline(x=min_duration, line_dash="dot", line_color="orange",
                  annotation_text=f"Min: {min_duration}", annotation_position="top left")
    fig.add_vline(x=target_duration, line_dash="dash", line_color="blue",
                  annotation_text=f"Target: {target_duration}", annotation_position="top")
    fig.add_vline(x=max_duration, line_dash="dot", line_color="red",
                  annotation_text=f"Max: {max_duration}", annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_title="Trading Day",
        yaxis_title="USD Executed (millions)",
        showlegend=False,
        height=300
    )

    return fig


def create_summary_table(results: dict) -> go.Figure:
    """
    Create a summary table comparing strategy results.

    Parameters
    ----------
    results : dict
        Dictionary with strategy names as keys and dicts containing
        'mean_perf', 'se_perf', 'mean_dur', 'se_dur'.

    Returns
    -------
    go.Figure
        Plotly table figure.
    """
    strategies = list(results.keys())
    mean_perfs = [f"{results[s]['mean_perf']:.2f}" for s in strategies]
    se_perfs = [f"{results[s]['se_perf']:.2f}" for s in strategies]
    mean_durs = [f"{results[s]['mean_dur']:.1f}" for s in strategies]
    se_durs = [f"{results[s]['se_dur']:.1f}" for s in strategies]
    formatted_perfs = [f"{results[s]['mean_perf']:.2f} +/- {results[s]['se_perf']:.2f}" for s in strategies]
    formatted_durs = [f"{results[s]['mean_dur']:.1f} +/- {results[s]['se_dur']:.1f}" for s in strategies]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Strategy', 'Mean Perf (bps)', 'SE Perf', 'Mean Duration', 'SE Duration',
                    'Performance +/- SE', 'Duration +/- SE'],
            fill_color='steelblue',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[strategies, mean_perfs, se_perfs, mean_durs, se_durs, formatted_perfs, formatted_durs],
            fill_color='lavender',
            align='center'
        )
    )])

    fig.update_layout(
        title="Strategy Comparison Summary",
        height=200
    )

    return fig
