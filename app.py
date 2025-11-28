"""
A Fixed-Notional Share Buyback Strategy
Streamlit Application

This application simulates and analyzes share buyback strategies
with flexible end times and benchmark-based adaptive execution.
"""

import streamlit as st
import numpy as np
import pandas as pd

# Import modules
from modules.config import (
    DEFAULT_S0, DEFAULT_DAYS, DEFAULT_MU, DEFAULT_SIGMA, DEFAULT_SIMULATIONS,
    DEFAULT_USD, DEFAULT_MAX_DURATION, DEFAULT_MIN_DURATION, DEFAULT_TARGET_DURATION,
    DEFAULT_DISCOUNT_BPS, TRADING_DAYS_PER_YEAR, S5_MIN_COMPLETION_PCT
)
from modules.gbm import generate_gbm_paths
from modules.benchmarks import compute_benchmark
from modules.strategies import (
    strategy_1, strategy_2, strategy_3, strategy_4, strategy_5,
    compute_execution_vwap_series, compute_performance_series
)
from modules.strategies_vectorized import (
    run_all_strategies_vectorized, precompute_benchmarks
)
from modules.cache import get_cached_paths, get_cached_results, set_cached_results
from modules.metrics import vwap, execution_performance_bps, standard_error, mean_with_se
from modules.visualizations import (
    plot_price_paths, plot_performance_histogram, plot_duration_histogram,
    plot_example_timeseries, plot_daily_usd, create_summary_table
)


# Page configuration
st.set_page_config(
    page_title="Share Buyback Strategy",
    page_icon="",
    layout="wide"
)

# Title and subtitle
st.title("A Fixed-Notional Share Buyback Strategy")
st.markdown("**The value of flexible end times and an easy benchmark**")
st.markdown("[www.candorpartners.net](https://www.candorpartners.net)")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Simulation & Results", "Example", "Explanation"])

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# BUTTONS AT TOP OF SIDEBAR
run_simulation = st.sidebar.button("Run Simulation", key="btn_run_simulation", type="primary")
generate_example = st.sidebar.button("Generate Example", key="btn_generate_example")

st.sidebar.markdown("---")

# Stock simulation parameters
st.sidebar.subheader("Stock Parameters")
S0 = st.sidebar.number_input(
    "Initial Stock Price ($)",
    min_value=1,
    max_value=200,
    value=int(DEFAULT_S0),
    step=1,
    key="input_s0"
)

sigma_pct = st.sidebar.number_input(
    "Annual Volatility (%)",
    min_value=0,
    max_value=100,
    value=int(DEFAULT_SIGMA),
    step=1,
    key="input_sigma"
)
sigma = sigma_pct / 100.0

mu_pct = st.sidebar.number_input(
    "Annual Drift (%)",
    min_value=-100,
    max_value=100,
    value=int(DEFAULT_MU),
    step=1,
    key="input_mu"
)
mu = mu_pct / 100.0

n_days = st.sidebar.slider(
    "Number of Days (X)",
    min_value=5,
    max_value=300,
    value=DEFAULT_DAYS,
    step=5,
    key="slider_days"
)

n_simulations = st.sidebar.number_input(
    "Number of Simulations (Z)",
    min_value=1000,
    max_value=100000,
    value=DEFAULT_SIMULATIONS,
    step=1000,
    key="input_simulations"
)

st.sidebar.markdown("---")

# Execution parameters
st.sidebar.subheader("Execution Parameters")
total_usd = st.sidebar.number_input(
    "Total USD to Execute ($)",
    min_value=1_000_000,
    max_value=100_000_000_000,
    value=DEFAULT_USD,
    step=100_000_000,
    format="%d",
    key="input_total_usd"
)

max_duration = st.sidebar.number_input(
    "Max Duration (days)",
    min_value=1,
    max_value=300,
    value=DEFAULT_MAX_DURATION,
    step=1,
    key="input_max_duration"
)

# Min duration - direct input (default = 60% of max)
default_min_dur = int(max_duration * 0.6)
min_duration = st.sidebar.number_input(
    "Min Duration (days)",
    min_value=1,
    max_value=300,
    value=default_min_dur,
    step=1,
    key="input_min_duration"
)

# Target duration - direct input (default = midpoint)
default_target_dur = (min_duration + max_duration) // 2
target_duration = st.sidebar.number_input(
    "Target Duration (days)",
    min_value=1,
    max_value=300,
    value=default_target_dur,
    step=1,
    key="input_target_duration"
)

st.sidebar.markdown("---")

# Strategy 3 discount
st.sidebar.subheader("Strategy 3 Discount")
discount_bps = st.sidebar.number_input(
    "Benchmark Discount (bps)",
    min_value=0,
    max_value=200,
    value=DEFAULT_DISCOUNT_BPS,
    step=10,
    key="input_discount_bps"
)

st.sidebar.markdown("---")

# Strategy 5 parameters
st.sidebar.subheader("Strategy 5 Parameters")
min_completion_pct = st.sidebar.slider(
    "Minimum Completion (%)",
    min_value=85.0,
    max_value=100.0,
    value=S5_MIN_COMPLETION_PCT,
    step=1.0,
    key="slider_min_completion",
    help="Minimum percentage of USD that must be executed. Lower values allow more patience at unfavorable prices."
)

# Validation warning
if n_days < max_duration:
    st.sidebar.warning(f"Days ({n_days}) should be >= Max Duration ({max_duration})")

st.sidebar.markdown("---")

# Random seed
use_seed = st.sidebar.checkbox("Use Fixed Random Seed", value=False, key="checkbox_use_seed")
if use_seed:
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="input_seed"
    )
else:
    seed = None


# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'example_results' not in st.session_state:
    st.session_state.example_results = None


def run_simulation_progressive(prices, total_usd, min_duration, max_duration,
                               target_duration, discount_bps, min_completion_pct, progress_bar, status_text):
    """
    Run simulation with progressive loading using vectorized strategies.

    Shows intermediate results as batches complete.
    """
    n_sims = prices.shape[0]

    # Batch sizes for progressive loading
    batch_sizes = [1000, 4000, 5000]  # Total = 10000
    if n_sims <= 1000:
        batch_sizes = [n_sims]
    elif n_sims <= 5000:
        batch_sizes = [1000, n_sims - 1000]

    # Accumulate results
    all_results = {
        'Strategy 1': {'performances': [], 'durations': []},
        'Strategy 2': {'performances': [], 'durations': []},
        'Strategy 3': {'performances': [], 'durations': []},
        'Strategy 4': {'performances': [], 'durations': []},
        'Strategy 5': {'performances': [], 'durations': [], 'completion_pcts': []}
    }

    processed = 0
    for batch_size in batch_sizes:
        if processed >= n_sims:
            break

        end_idx = min(processed + batch_size, n_sims)
        batch_paths = prices[processed:end_idx]

        # Run vectorized strategies on batch
        batch_results = run_all_strategies_vectorized(
            batch_paths, total_usd, min_duration, max_duration,
            target_duration, discount_bps, min_completion_pct
        )

        # Accumulate results
        for strategy in all_results:
            all_results[strategy]['performances'].extend(batch_results[strategy]['performances'])
            all_results[strategy]['durations'].extend(batch_results[strategy]['durations'])
            if 'completion_pcts' in all_results[strategy] and 'completion_pcts' in batch_results[strategy]:
                all_results[strategy]['completion_pcts'].extend(batch_results[strategy]['completion_pcts'])

        processed = end_idx
        progress_pct = processed / n_sims
        progress_bar.progress(progress_pct)
        status_text.text(f"Processed {processed:,} / {n_sims:,} simulations ({progress_pct*100:.0f}%)")

    # Convert to numpy arrays
    for strategy in all_results:
        all_results[strategy]['performances'] = np.array(all_results[strategy]['performances'])
        all_results[strategy]['durations'] = np.array(all_results[strategy]['durations'])
        if 'completion_pcts' in all_results[strategy] and len(all_results[strategy]['completion_pcts']) > 0:
            all_results[strategy]['completion_pcts'] = np.array(all_results[strategy]['completion_pcts'])

    return all_results


def run_all_strategies(prices, total_usd, min_duration, max_duration, target_duration, discount_bps):
    """Run all four strategies using vectorized implementation (for backward compatibility)."""
    return run_all_strategies_vectorized(
        prices, total_usd, min_duration, max_duration, target_duration, discount_bps
    )


# TAB 1: Simulation & Results
with tab1:
    if run_simulation:
        # Progressive loading UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting simulation...")

        # Try to use cached paths
        paths = get_cached_paths(
            S0, mu, sigma, n_days, n_simulations, seed,
            generate_gbm_paths
        )

        # Check for cached results
        stock_params = {'S0': S0, 'mu': mu, 'sigma': sigma, 'n_days': n_days, 'n_sims': n_simulations, 'seed': seed}
        strategy_params = {'total_usd': total_usd, 'min_duration': min_duration,
                          'max_duration': max_duration, 'target_duration': target_duration,
                          'discount_bps': discount_bps, 'min_completion_pct': min_completion_pct}

        cached_results = get_cached_results(stock_params, strategy_params)

        if cached_results is not None:
            results = cached_results
            progress_bar.progress(1.0)
            status_text.text("Loaded from cache!")
        else:
            # Run with progressive loading
            results = run_simulation_progressive(
                paths, total_usd, min_duration, max_duration,
                target_duration, discount_bps, min_completion_pct, progress_bar, status_text
            )
            # Cache results
            set_cached_results(stock_params, strategy_params, results)

        # Store in session state
        st.session_state.simulation_results = {
            'paths': paths,
            'results': results,
            'params': {
                'S0': S0, 'mu': mu, 'sigma': sigma, 'n_days': n_days,
                'n_simulations': n_simulations, 'total_usd': total_usd,
                'min_duration': min_duration, 'max_duration': max_duration,
                'target_duration': target_duration, 'discount_bps': discount_bps,
                'min_completion_pct': min_completion_pct
            }
        }

        # Clear progress UI
        progress_bar.empty()
        status_text.empty()
        st.success(f"Simulation complete! Processed {n_simulations:,} paths.")

    if st.session_state.simulation_results is not None:
        sim_data = st.session_state.simulation_results
        paths = sim_data['paths']
        results = sim_data['results']
        params = sim_data['params']

        # Display price paths
        st.subheader("Simulated Stock Price Paths")
        fig_paths = plot_price_paths(
            paths, params['S0'], params['mu'], params['sigma'],
            show_n=100, title=f"Stock Price Simulation ({params['n_simulations']} paths, showing 100)"
        )
        st.plotly_chart(fig_paths, use_container_width=True, key="chart_price_paths")

        # Performance histograms
        st.subheader("Performance Distributions (bps)")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fig_perf1 = plot_performance_histogram(
                results['Strategy 1']['performances'],
                "Strategy 1 (Uniform)"
            )
            st.plotly_chart(fig_perf1, use_container_width=True, key="chart_perf_hist_1")

        with col2:
            fig_perf2 = plot_performance_histogram(
                results['Strategy 2']['performances'],
                "Strategy 2 (Adaptive)"
            )
            st.plotly_chart(fig_perf2, use_container_width=True, key="chart_perf_hist_2")

        with col3:
            fig_perf3 = plot_performance_histogram(
                results['Strategy 3']['performances'],
                f"Strategy 3 (Discount {params['discount_bps']}bps)"
            )
            st.plotly_chart(fig_perf3, use_container_width=True, key="chart_perf_hist_3")

        with col4:
            fig_perf4 = plot_performance_histogram(
                results['Strategy 4']['performances'],
                "Strategy 4 (Convex)"
            )
            st.plotly_chart(fig_perf4, use_container_width=True, key="chart_perf_hist_4")

        with col5:
            fig_perf5 = plot_performance_histogram(
                results['Strategy 5']['performances'],
                f"Strategy 5 (Flex {params['min_completion_pct']:.0f}%)"
            )
            st.plotly_chart(fig_perf5, use_container_width=True, key="chart_perf_hist_5")

        # Duration histograms
        st.subheader("Duration Distributions")
        col6, col7, col8, col9, col10 = st.columns(5)

        with col6:
            fig_dur1 = plot_duration_histogram(
                results['Strategy 1']['durations'],
                "Strategy 1 (Uniform)",
                params['min_duration'], params['target_duration'], params['max_duration']
            )
            st.plotly_chart(fig_dur1, use_container_width=True, key="chart_dur_hist_1")

        with col7:
            fig_dur2 = plot_duration_histogram(
                results['Strategy 2']['durations'],
                "Strategy 2 (Adaptive)",
                params['min_duration'], params['target_duration'], params['max_duration']
            )
            st.plotly_chart(fig_dur2, use_container_width=True, key="chart_dur_hist_2")

        with col8:
            fig_dur3 = plot_duration_histogram(
                results['Strategy 3']['durations'],
                f"Strategy 3 (Discount {params['discount_bps']}bps)",
                params['min_duration'], params['target_duration'], params['max_duration']
            )
            st.plotly_chart(fig_dur3, use_container_width=True, key="chart_dur_hist_3")

        with col9:
            fig_dur4 = plot_duration_histogram(
                results['Strategy 4']['durations'],
                "Strategy 4 (Convex)",
                params['min_duration'], params['target_duration'], params['max_duration']
            )
            st.plotly_chart(fig_dur4, use_container_width=True, key="chart_dur_hist_4")

        with col10:
            fig_dur5 = plot_duration_histogram(
                results['Strategy 5']['durations'],
                f"Strategy 5 (Flex {params['min_completion_pct']:.0f}%)",
                params['min_duration'], params['target_duration'], params['max_duration']
            )
            st.plotly_chart(fig_dur5, use_container_width=True, key="chart_dur_hist_5")

        # Summary statistics table
        st.subheader("Summary Statistics")

        summary_data = []
        for strategy_name in ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4', 'Strategy 5']:
            perfs = results[strategy_name]['performances']
            durs = results[strategy_name]['durations']
            mean_perf, se_perf = mean_with_se(perfs)
            mean_dur, se_dur = mean_with_se(durs)

            row = {
                'Strategy': strategy_name,
                'Mean Performance (bps)': f"{mean_perf:.2f}",
                'SE Performance': f"{se_perf:.2f}",
                'Perf +/- SE': f"{mean_perf:.2f} +/- {se_perf:.2f}",
                'Mean Duration': f"{mean_dur:.1f}",
                'SE Duration': f"{se_dur:.1f}",
                'Duration +/- SE': f"{mean_dur:.1f} +/- {se_dur:.1f}",
                'Win Rate (%)': f"{100 * np.mean(perfs > 0):.1f}"
            }

            # Add completion % for Strategy 5
            if strategy_name == 'Strategy 5' and 'completion_pcts' in results[strategy_name]:
                completions = results[strategy_name]['completion_pcts']
                row['Avg Completion (%)'] = f"{np.mean(completions):.2f}"
                row['Partial (<100%)'] = f"{np.sum(completions < 100)}"
            else:
                row['Avg Completion (%)'] = "100.00"
                row['Partial (<100%)'] = "0"

            summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    else:
        st.info("Click 'Run Simulation' in the sidebar to generate results.")


# TAB 2: Example
with tab2:
    # Uses generate_example button from top of sidebar
    if generate_example:
        with st.spinner("Generating example..."):
            # Generate a single random path
            example_seed = np.random.randint(0, 1000000) if not use_seed else seed
            example_path = generate_gbm_paths(S0, mu, sigma, n_days, 1, seed=example_seed)[0]

            # Run all three strategies
            example_results = {}

            # Strategy 1
            usd1, shares1, total_shares1, end1 = strategy_1(example_path, total_usd, target_duration)
            benchmark1 = compute_benchmark(example_path)
            vwap_series1 = compute_execution_vwap_series(example_path, usd1, shares1)
            perf_series1 = compute_performance_series(vwap_series1, benchmark1)
            example_results['Strategy 1'] = {
                'usd_per_day': usd1, 'shares_per_day': shares1,
                'total_shares': total_shares1, 'end_day': end1,
                'benchmark': benchmark1, 'vwap_series': vwap_series1,
                'perf_series': perf_series1
            }

            # Strategy 2
            usd2, shares2, total_shares2, end2 = strategy_2(
                example_path, total_usd, min_duration, max_duration, target_duration, discount_bps=0
            )
            benchmark2 = compute_benchmark(example_path)
            vwap_series2 = compute_execution_vwap_series(example_path, usd2, shares2)
            perf_series2 = compute_performance_series(vwap_series2, benchmark2)
            example_results['Strategy 2'] = {
                'usd_per_day': usd2, 'shares_per_day': shares2,
                'total_shares': total_shares2, 'end_day': end2,
                'benchmark': benchmark2, 'vwap_series': vwap_series2,
                'perf_series': perf_series2
            }

            # Strategy 3
            usd3, shares3, total_shares3, end3 = strategy_3(
                example_path, total_usd, min_duration, max_duration, target_duration, discount_bps=discount_bps
            )
            benchmark3 = compute_benchmark(example_path, discount_bps=discount_bps)
            vwap_series3 = compute_execution_vwap_series(example_path, usd3, shares3)
            perf_series3 = compute_performance_series(vwap_series3, benchmark3)
            example_results['Strategy 3'] = {
                'usd_per_day': usd3, 'shares_per_day': shares3,
                'total_shares': total_shares3, 'end_day': end3,
                'benchmark': benchmark3, 'vwap_series': vwap_series3,
                'perf_series': perf_series3
            }

            # Strategy 4
            usd4, shares4, total_shares4, end4 = strategy_4(
                example_path, total_usd, min_duration, max_duration, target_duration
            )
            benchmark4 = compute_benchmark(example_path)
            vwap_series4 = compute_execution_vwap_series(example_path, usd4, shares4)
            perf_series4 = compute_performance_series(vwap_series4, benchmark4)
            example_results['Strategy 4'] = {
                'usd_per_day': usd4, 'shares_per_day': shares4,
                'total_shares': total_shares4, 'end_day': end4,
                'benchmark': benchmark4, 'vwap_series': vwap_series4,
                'perf_series': perf_series4
            }

            # Strategy 5
            usd5, shares5, total_shares5, end5, completion_pct5 = strategy_5(
                example_path, total_usd, min_duration, max_duration, target_duration,
                min_completion_pct=min_completion_pct
            )
            benchmark5 = compute_benchmark(example_path)
            vwap_series5 = compute_execution_vwap_series(example_path, usd5, shares5)
            perf_series5 = compute_performance_series(vwap_series5, benchmark5)
            example_results['Strategy 5'] = {
                'usd_per_day': usd5, 'shares_per_day': shares5,
                'total_shares': total_shares5, 'end_day': end5,
                'benchmark': benchmark5, 'vwap_series': vwap_series5,
                'perf_series': perf_series5,
                'completion_pct': completion_pct5
            }

            st.session_state.example_results = {
                'path': example_path,
                'results': example_results,
                'params': {
                    'min_duration': min_duration,
                    'max_duration': max_duration,
                    'target_duration': target_duration,
                    'discount_bps': discount_bps,
                    'min_completion_pct': min_completion_pct
                }
            }

        st.success("Example generated!")

    if st.session_state.example_results is not None:
        ex_data = st.session_state.example_results
        example_path = ex_data['path']
        example_results = ex_data['results']
        ex_params = ex_data['params']

        st.subheader("Single Path Example Execution")

        # Display five strategy execution charts
        for strategy_name in ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4', 'Strategy 5']:
            st.markdown(f"### {strategy_name}")

            res = example_results[strategy_name]

            col_ts, col_usd = st.columns([2, 1])

            with col_ts:
                fig_ts = plot_example_timeseries(
                    example_path,
                    res['benchmark'],
                    res['vwap_series'],
                    res['perf_series'],
                    res['end_day'],
                    ex_params['min_duration'],
                    ex_params['target_duration'],
                    ex_params['max_duration'],
                    strategy_name
                )
                st.plotly_chart(fig_ts, use_container_width=True, key=f"chart_example_ts_{strategy_name}")

            with col_usd:
                fig_usd = plot_daily_usd(
                    res['usd_per_day'],
                    strategy_name,
                    res['end_day'],
                    ex_params['min_duration'],
                    ex_params['target_duration'],
                    ex_params['max_duration']
                )
                st.plotly_chart(fig_usd, use_container_width=True, key=f"chart_example_usd_{strategy_name}")

            # Summary metrics for this strategy
            final_vwap = total_usd / res['total_shares'] if res['total_shares'] > 0 else 0
            final_benchmark = np.mean(example_path[:res['end_day']])
            final_perf = execution_performance_bps(final_vwap, final_benchmark)

            # For Strategy 5, also show completion %
            if strategy_name == 'Strategy 5' and 'completion_pct' in res:
                st.markdown(f"""
                **Results:** End Day: {res['end_day']} | VWAP: ${final_vwap:.2f} | Benchmark: ${final_benchmark:.2f} | Performance: {final_perf:.1f} bps | Completion: {res['completion_pct']:.1f}%
                """)
            else:
                st.markdown(f"""
                **Results:** End Day: {res['end_day']} | VWAP: ${final_vwap:.2f} | Benchmark: ${final_benchmark:.2f} | Performance: {final_perf:.1f} bps
                """)
            st.markdown("---")

    else:
        st.info("Click 'Generate Example' in the sidebar to see a single execution example.")


# TAB 3: Explanation
with tab3:
    st.header("Strategy Explanation")

    st.markdown("""
    ## Geometric Brownian Motion (GBM)

    Stock prices are simulated using Geometric Brownian Motion, the standard model for stock price dynamics:

    $$dS = \\mu S \\, dt + \\sigma S \\, dW$$

    Where:
    - $S$ = Stock price
    - $\\mu$ = Annual drift (expected return)
    - $\\sigma$ = Annual volatility
    - $dW$ = Wiener process (Brownian motion) increment

    ### Discretization

    For simulation, we use the exact discretization formula:

    $$S_{t+1} = S_t \\cdot \\exp\\left((\\mu - \\frac{1}{2}\\sigma^2)\\Delta t + \\sigma\\sqrt{\\Delta t} \\cdot Z\\right)$$

    Where:
    - $\\Delta t = 1/252$ (one trading day)
    - $Z \\sim N(0,1)$ (standard normal random variable)

    ---

    ## Benchmark Calculation

    The benchmark is the **expanding window arithmetic mean** of daily prices:

    $$\\text{Benchmark}_t = \\frac{1}{t} \\sum_{i=1}^{t} P_i$$

    This represents what a naive investor would pay if they bought equal dollar amounts each day.

    For Strategy 3, a discount is applied:

    $$\\text{Discounted Benchmark}_t = \\text{Benchmark}_t \\cdot \\left(1 - \\frac{\\text{discount\\_bps}}{10000}\\right)$$

    ---

    ## Strategy Descriptions

    ### Strategy 1: Uniform Execution

    The simplest approach - execute equal USD amounts each day:

    $$\\text{Daily USD} = \\frac{\\text{Total USD}}{\\text{Target Duration}}$$

    Execute for exactly **target_duration** days, regardless of price movements.

    ---

    ### Strategy 2: Adaptive Execution (No Discount)

    Adapts execution speed based on current price vs. benchmark:

    **Initial Period** (first 10 days or min_duration, whichever is smaller):
    - Execute constant amount: $\\text{base\\_daily} = \\frac{\\text{Total USD}}{\\text{Target Duration}}$

    **Adaptive Period** (after initial period):

    **Case 1: Price < Benchmark (Favorable)**
    - If past min_duration: Finish ASAP, up to $5 \\times \\text{base\\_daily}$ per day
    - If before min_duration: Speed up to finish by min_duration

    **Case 2: Price > Benchmark (Unfavorable)**
    - Slow down to finish by max_duration
    - **Extra slow mode**: If remaining USD < $5 \\times \\text{base\\_daily}$ AND days to max > 5:
      - Use only $0.1 \\times \\text{base\\_daily}$ per day

    ### Strategy 2 Decision Flowchart

    ```
    START: Day d
         |
         v
    [Compute Benchmark = mean(prices[1:d])]
         |
         v
    {d <= initial_period (10)?}
         |
        YES --> Execute base_daily_usd --> [Next Day]
         |
        NO
         |
         v
    {price < benchmark?}
         |
    +----+----+
    |         |
   YES        NO
    |         |
    v         v
    {d > min_duration?}    [SLOW DOWN: target max_duration]
    |         |                    |
   YES       NO                    v
    |         |            {remaining < 5x base AND
    v         v             days_to_max > 5?}
    [ASAP:    [SPEED UP:           |
     max 5x    finish by      +----+----+
     base]     min_dur]      YES       NO
    |         |              |         |
    |         |              v         v
    |         |         [EXTRA      [NORMAL
    |         |          SLOW:       SLOW:
    |         |          0.1x       spread to
    |         |          base]       max]
    +----+----+----+----+----+----+----+
                   |
                   v
         [Execute today_usd]
                   |
                   v
         {remaining_usd < 0.01?}
                   |
              +----+----+
             YES       NO
              |         |
              v         v
            [END]   [Next Day]
    ```

    ---

    ### Strategy 3: Adaptive with Discounted Benchmark

    Same logic as Strategy 2, but applies a discount to the benchmark:

    $$\\text{Adjusted Benchmark} = \\text{Benchmark} \\cdot \\left(1 - \\frac{\\text{discount\\_bps}}{10000}\\right)$$

    This makes prices appear more favorable relative to the benchmark, resulting in:
    - More aggressive buying behavior
    - Typically shorter execution duration
    - Higher conviction in "favorable" price detection

    ---

    ### Strategy 4: Multi-Factor Convex Adaptive

    An advanced strategy that combines three components for potentially superior performance:

    **Component D: Exponential Convex Scaling**

    $$\\text{deviation} = \\frac{\\text{Benchmark} - \\text{Price}}{\\text{Benchmark}}$$
    $$\\text{convex\\_multiplier} = e^{\\beta \\cdot \\text{deviation}}$$

    Where $\\beta = 20$ (convex sensitivity). This creates an exponential response to price deviations,
    concentrating buying power on large discounts rather than small ones.

    **Component E: Time-Urgency Factor**

    $$\\text{urgency} = 1 + \\gamma \\cdot \\left(\\frac{\\text{day}}{\\text{max\\_duration}}\\right)^2$$

    Where $\\gamma = 3$ (urgency acceleration). This quadratically increases buying aggression
    as the deadline approaches, ensuring completion regardless of price.

    **Component F: Z-Score Statistical Filter**

    $$z = \\frac{\\bar{P}_{\\text{rolling}} - P_t}{\\sigma_{\\text{rolling}}}$$

    Where the rolling window is 20 days. Only acts on statistically significant signals
    ($|z| > 1.0$), filtering out noise.

    **Combined Execution:**
    - If $|z| > z_{\\text{threshold}}$: Full response = convex $\\times$ urgency $\\times$ signal boost
    - Otherwise: Conservative response = urgency only

    The final multiplier is bounded: $0.1 \\leq \\text{multiplier} \\leq 8.0$

    ---

    ### Strategy 5: Flexible Completion Adaptive

    Strategy 5 builds on Strategy 4's convex approach but adds **flexible completion** to avoid
    forced buying at unfavorable prices at the deadline.

    **Core Logic (same as Strategy 4):**
    - Exponential convex scaling based on price vs benchmark deviation
    - Time-urgency factor that increases as deadline approaches

    **Key Innovation: Partial Completion**

    At the deadline (max_duration), Strategy 5 evaluates:
    ```
    if executed_usd >= min_usd AND final_deviation < unfavorable_threshold:
        Accept partial completion (don't force-buy remaining)
    else:
        Complete remaining execution normally
    ```

    **Parameters:**
    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | min_completion_pct | 95% | Minimum required execution (85-100%) |
    | unfavorable_threshold | -1% | Skip forced completion if price > benchmark by this |
    | beta | 50 | Convex sensitivity |
    | gamma | 1 | Urgency acceleration |

    **Benefits:**
    - Avoids worst-case forced buying at highly unfavorable prices
    - Maintains most of Strategy 4's adaptive behavior
    - Configurable risk/completion trade-off

    ---

    ## Performance Metrics

    ### VWAP (Volume-Weighted Average Price)

    $$\\text{VWAP} = \\frac{\\text{Total USD Spent}}{\\text{Total Shares Acquired}}$$

    ### Execution Performance (Basis Points)

    $$\\text{Performance (bps)} = -\\left(\\frac{\\text{VWAP} - \\text{Benchmark}}{\\text{Benchmark}}\\right) \\times 10000$$

    - **Positive performance**: VWAP < Benchmark (bought cheaper than average)
    - **Negative performance**: VWAP > Benchmark (bought more expensive than average)

    ### Standard Error

    $$\\text{SE} = \\frac{\\sigma}{\\sqrt{n}}$$

    Where $\\sigma$ is the sample standard deviation and $n$ is the number of simulations.

    ---

    ## Numerical Example

    Consider a simple 5-day example:
    - Prices: [100, 102, 98, 101, 99]
    - Total USD: $10,000
    - Target Duration: 5 days

    **Strategy 1 (Uniform):**
    - Daily USD: $2,000
    - Shares per day: [20.00, 19.61, 20.41, 19.80, 20.20]
    - Total shares: 100.02
    - VWAP: $99.98
    - Benchmark: $100.00
    - Performance: +0.2 bps

    **Strategy 2 (Adaptive) - Detailed Trace:**

    Settings: Total USD = $10,000, min_dur = 3, max_dur = 7, target_dur = 5
    base_daily = $10,000 / 5 = $2,000

    | Day | Price | Benchmark | Price<Bench? | Remaining | Decision | Execute |
    |-----|-------|-----------|--------------|-----------|----------|---------|
    | 1   | 100   | 100.00    | Initial      | $10,000   | Initial  | $2,000  |
    | 2   | 102   | 101.00    | NO (102>101) | $8,000    | Slow     | $1,333  |
    | 3   | 98    | 100.00    | YES (98<100) | $6,667    | Speed up | $3,333  |
    | 4   | 101   | 100.25    | NO (101>100) | $3,334    | Slow     | $833    |
    | 5   | 99    | 100.00    | YES (99<100) | $2,501    | ASAP     | $2,501  |

    Day 2: Price > Benchmark -> Slow down to finish by max (day 7)
           remaining / (7-2+1) = $8,000 / 6 = $1,333

    Day 3: Price < Benchmark, before min_dur -> Speed up for min_dur (day 3)
           remaining / (3-3+1) = $6,667 / 1 = $6,667, capped at 5x = $10,000
           Actually execute $3,333 (min to finish)

    Day 5: Price < Benchmark, past min_dur -> ASAP, max 5x base = $10,000
           Execute remaining $2,501

    Result: Ended day 5, bought more when prices were low (day 3, 5)

    ---

    ## Key Insights

    1. **Strategy 1** provides a simple baseline with predictable execution
    2. **Strategy 2** exploits mean reversion by buying more when prices are favorable
    3. **Strategy 3** with discount creates a more aggressive buying bias
    4. **Strategy 4** uses convex scaling to concentrate buying on large discounts while filtering noise
    5. **Strategy 5** adds flexible completion to avoid forced buying at unfavorable deadline prices
    6. Flexible end times allow the strategy to capitalize on favorable conditions
    7. The benchmark provides a natural target that adjusts to market conditions
    """)

    st.markdown("---")
    st.markdown("*For more information, visit [www.candorpartners.net](https://www.candorpartners.net)*")
