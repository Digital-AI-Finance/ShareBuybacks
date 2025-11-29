# Getting Started

## Installation

### Option 1: Run from Source

```bash
# Clone the repository
git clone https://github.com/Digital-AI-Finance/ShareBuybacks.git
cd ShareBuybacks

# Install dependencies
pip install streamlit plotly pandas numpy

# Run the app
streamlit run app.py
```

### Option 2: Download Standalone Executable

Download pre-built executables from [GitHub Releases](https://github.com/Digital-AI-Finance/ShareBuybacks/releases):

- **Windows**: `ShareBuybackStrategy2.exe`
- **macOS**: `ShareBuybackStrategy-Mac.zip`

Simply download and run - no Python installation required.

## Using the App

### Simulation Parameters

Configure the Monte Carlo simulation in the sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Starting Price (S0) | $100 | Initial stock price |
| Trading Days | 125 | Simulation length (~6 months) |
| Annual Drift | 0% | Expected price trend |
| Annual Volatility | 25% | Price volatility |
| Simulations | 10,000 | Number of Monte Carlo paths |

### Execution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Total USD | $1B | Total buyback amount |
| Min Duration | 75 days | Earliest completion |
| Max Duration | 125 days | Latest completion (deadline) |
| Discount (bps) | 0 | Strategy 3 benchmark discount |

### Strategy 5 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Completion % | 95% | Minimum required completion |

## Running a Simulation

1. **Adjust parameters** in the sidebar
2. **Click "Run Simulation"** button
3. **View results** in the main panel:
   - Performance distribution histogram
   - Duration distribution
   - Strategy comparison table
   - Example execution path

## Understanding Results

### Performance (bps)

Performance is measured in basis points relative to the TWAP benchmark:

$$
\text{Performance} = \frac{\text{Benchmark} - \text{VWAP}}{\text{Benchmark}} \times 10000
$$

- **Positive bps**: Buying below benchmark (good)
- **Negative bps**: Buying above benchmark (bad)
- **Zero bps**: Matching benchmark exactly

### Key Metrics

- **Mean Performance**: Average outperformance across all simulations
- **Std Dev**: Risk/variability of performance
- **Duration**: Average number of days to complete execution
- **Completion %**: Percentage of target USD executed (Strategy 5)

## Tips

!!! tip "Start with defaults"
    The default parameters represent typical market conditions. Experiment from there.

!!! tip "High volatility favors adaptive strategies"
    Strategies 2-5 perform best in volatile markets where price discounts are larger.

!!! warning "Strategy 5 may not complete"
    With low min_completion_pct, Strategy 5 may leave significant USD unexecuted.
