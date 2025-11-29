# App Demo

## Streamlit Application

The Share Buyback Strategy Simulator is built with [Streamlit](https://streamlit.io/), providing an interactive web interface for running simulations.

## Interface Overview

### Sidebar Controls

The left sidebar contains all simulation parameters:

- **Simulation Parameters**: Price dynamics (S0, volatility, drift, days)
- **Execution Parameters**: Buyback settings (USD, duration constraints)
- **Strategy Parameters**: Strategy-specific settings (discount, min completion)
- **Run Simulation**: Button to execute Monte Carlo simulation

### Main Panel

The main area displays:

1. **Performance Distribution**: Histogram of performance across all simulations
2. **Duration Distribution**: How long each strategy takes to complete
3. **Comparison Table**: Side-by-side statistics for all strategies
4. **Example Path**: Single execution trace showing daily behavior

## Screenshots

!!! note "Screenshots coming soon"
    App screenshots will be added in a future update.

### Performance Histogram

Shows the distribution of performance (in basis points) across 10,000+ simulations:

- X-axis: Performance relative to TWAP benchmark
- Y-axis: Frequency
- Each color represents a different strategy

### Strategy Comparison Table

| Column | Description |
|--------|-------------|
| Strategy | Strategy name and number |
| Mean (bps) | Average performance |
| Std (bps) | Standard deviation |
| Min/Max | Range of outcomes |
| Duration | Average completion time |

### Execution Path Visualization

For a single random simulation:

- **Price path**: Stock price over time
- **Benchmark**: Rolling TWAP
- **USD executed**: Cumulative buyback progress
- **Daily amounts**: Bar chart of daily execution

## Try It Yourself

=== "Cloud (Streamlit)"

    The app may be deployed on Streamlit Cloud:

    [Launch App](https://share.streamlit.io/digital-ai-finance/sharebuybacks/main/app.py){ .md-button .md-button--primary }

=== "Local"

    ```bash
    git clone https://github.com/Digital-AI-Finance/ShareBuybacks.git
    cd ShareBuybacks
    pip install -r requirements.txt
    streamlit run app.py
    ```

=== "Executable"

    Download from [Releases](https://github.com/Digital-AI-Finance/ShareBuybacks/releases) and run directly.

## Interactive Features

### Parameter Sliders

All parameters can be adjusted in real-time:

- Drag sliders to change values
- See immediate effect on simulation
- Compare before/after results

### Tabs

Switch between different views:

- **Simulation**: Run Monte Carlo simulations
- **Example Path**: Detailed single execution
- **Explanation**: Strategy documentation

### Export

Results can be copied or the page saved for reporting.
