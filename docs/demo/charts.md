# Interactive Charts

## Strategy Performance Comparison

The following charts illustrate typical simulation results. Run the app to generate your own interactive visualizations.

## Performance Distribution

Strategy performance follows approximately normal distributions, with adaptive strategies shifted positively:

| Strategy | Mean | Std Dev | Skew |
|----------|------|---------|------|
| S1 Uniform | 0 | ~50 | 0 |
| S2 Adaptive | +20 | ~65 | +0.1 |
| S3 Discount | +25 | ~60 | +0.1 |
| S4 Convex | +35 | ~80 | +0.2 |
| S5 Flexible | +40 | ~90 | +0.3 |

## Volatility Sensitivity

Higher volatility amplifies strategy differences:

| Volatility | S1 | S2 | S3 | S4 | S5 |
|------------|----|----|----|----|-----|
| 15% | 0 | +10 | +12 | +15 | +18 |
| 25% | 0 | +20 | +25 | +35 | +40 |
| 35% | 0 | +30 | +38 | +55 | +65 |
| 50% | 0 | +45 | +55 | +85 | +100 |

!!! tip "Key insight"
    Adaptive strategies perform best in volatile markets where discounts are larger and more frequent.

## Duration Distribution

Strategy 4 and 5 tend to complete earlier due to aggressive buying at discounts:

| Strategy | Mean Duration | Std Dev | Range |
|----------|---------------|---------|-------|
| S1 | 100 (fixed) | 0 | 100-100 |
| S2 | 95 | 15 | 75-125 |
| S3 | 93 | 14 | 75-125 |
| S4 | 88 | 18 | 75-125 |
| S5 | 85 | 20 | 75-125 |

## Execution Path Example

A typical execution path shows:

1. **Early phase (days 1-10)**: Constant execution to establish baseline
2. **Adaptive phase (days 11-74)**: Variable execution based on price
3. **Completion phase (days 75+)**: Urgency-driven execution to meet deadline

## Generate Your Own Charts

The Streamlit app generates interactive Plotly charts that you can:

- Zoom and pan
- Hover for data points
- Export as PNG
- Toggle strategy visibility

```python
# Example: Generate performance comparison
import plotly.graph_objects as go

fig = go.Figure()
for strategy, data in results.items():
    fig.add_trace(go.Histogram(
        x=data['performances'],
        name=strategy,
        opacity=0.7
    ))
fig.update_layout(barmode='overlay')
fig.show()
```

## API for Custom Analysis

Use the vectorized strategy functions for batch analysis:

```python
from modules.gbm import generate_gbm_paths
from modules.strategies_vectorized import (
    precompute_benchmarks,
    strategy_4_vectorized
)

# Generate 10,000 price paths
prices = generate_gbm_paths(S0=100, mu=0, sigma=0.25,
                            n_days=125, n_sims=10000)

# Precompute benchmarks
benchmarks = precompute_benchmarks(prices)

# Run Strategy 4 on all paths
perf, dur, vwap, bench = strategy_4_vectorized(
    prices, benchmarks,
    total_usd=1e9,
    min_duration=75,
    max_duration=125,
    target_duration=100
)

print(f"Mean performance: {perf.mean():.2f} bps")
print(f"Mean duration: {dur.mean():.1f} days")
```
