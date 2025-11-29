# Share Buyback Strategy Simulator

<div style="text-align: center; margin: 2rem 0;">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/github/v/release/Digital-AI-Finance/ShareBuybacks" alt="Release">
</div>

## Overview

A Monte Carlo simulation framework for evaluating **adaptive share buyback execution strategies**. Compare different algorithmic approaches to executing large buyback programs while minimizing market impact and maximizing value.

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **5 Strategies**

    ---

    Compare uniform, adaptive, discounted, convex, and flexible execution approaches

    [:octicons-arrow-right-24: View Strategies](strategies/index.md)

-   :material-flask:{ .lg .middle } **Monte Carlo Simulation**

    ---

    Run 10,000+ price path simulations using Geometric Brownian Motion

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-chart-box:{ .lg .middle } **Interactive Visualizations**

    ---

    Explore performance distributions, execution paths, and strategy comparisons

    [:octicons-arrow-right-24: View Demo](demo/index.md)

-   :material-download:{ .lg .middle } **Standalone Executables**

    ---

    Download pre-built Windows (.exe) and Mac (.app) versions

    [:octicons-arrow-right-24: Releases](https://github.com/Digital-AI-Finance/ShareBuybacks/releases)

</div>

## Key Features

- **Geometric Brownian Motion** price simulation with configurable drift and volatility
- **5 execution strategies** from simple uniform to sophisticated convex adaptive
- **Performance metrics**: VWAP vs benchmark, execution duration, completion rates
- **Interactive Streamlit UI** with real-time parameter adjustment
- **Vectorized NumPy** implementation for fast Monte Carlo simulation

## Quick Start

=== "Streamlit App"

    ```bash
    pip install streamlit plotly pandas numpy
    streamlit run app.py
    ```

=== "Standalone Executable"

    Download from [GitHub Releases](https://github.com/Digital-AI-Finance/ShareBuybacks/releases) and run directly.

## Strategy Performance Summary

| Strategy | Avg Performance | Risk (Std) | Best For |
|----------|-----------------|------------|----------|
| S1 Uniform | 0 bps | Low | Baseline comparison |
| S2 Adaptive | +15-25 bps | Medium | General use |
| S3 Discount | +20-30 bps | Medium | Conservative targets |
| S4 Convex | +25-40 bps | Higher | Aggressive optimization |
| S5 Flexible | +30-45 bps | Variable | Maximum performance |

!!! note "Performance depends on market conditions"
    Results vary based on volatility, execution window, and price dynamics.

## Architecture

```
ShareBuybackApp/
    app.py                    # Streamlit UI
    modules/
        config.py             # Default parameters
        gbm.py                # Price simulation
        strategies.py         # Strategy implementations
        strategies_vectorized.py  # Fast vectorized versions
    tests/
        test_strategies.py    # Unit tests
```

## License

MIT License - See [LICENSE](https://github.com/Digital-AI-Finance/ShareBuybacks/blob/main/LICENSE) for details.
