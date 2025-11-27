# Share Buyback Strategy App - Full Specification

This document contains the complete specification to recreate the Share Buyback Strategy Streamlit application from scratch.

---

## 1. Overview

**Purpose:** Interactive simulation tool for comparing share buyback execution strategies using Monte Carlo simulation with Geometric Brownian Motion (GBM) stock price paths.

**Target Users:** Finance professionals, researchers, educators studying optimal execution strategies.

**Tech Stack:**
- Python 3.11+
- Streamlit (UI framework)
- Plotly (interactive charts)
- NumPy (numerical computation)
- Pandas (data manipulation)

**Repository:** https://github.com/Digital-AI-Finance/ShareBuybacks

---

## 2. Mathematical Foundation

### 2.1 Geometric Brownian Motion (GBM)

Stock prices follow the stochastic differential equation:

```
dS = mu * S * dt + sigma * S * dW
```

Where:
- S = stock price
- mu = annual drift (expected return)
- sigma = annual volatility
- dW = Wiener process increment

### 2.2 Exact Discretization Formula

For simulation, use:

```
S[t+1] = S[t] * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
```

Where:
- dt = 1/252 (one trading day, assuming 252 trading days per year)
- Z ~ N(0,1) (standard normal random variable)

### 2.3 Implementation

```python
def generate_gbm_paths(S0, mu, sigma, days, n_simulations, seed=None):
    dt = 1.0 / 252  # One trading day
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    prices = np.zeros((n_simulations, days))
    prices[:, 0] = S0

    Z = np.random.standard_normal((n_simulations, days - 1))
    log_returns = drift + diffusion * Z

    for t in range(1, days):
        prices[:, t] = prices[:, t-1] * np.exp(log_returns[:, t-1])

    return prices
```

---

## 3. Strategy Specifications

### 3.1 Strategy 1: Uniform Execution (Constant Daily Purchase)

**Logic:** Execute equal USD amounts each day for exactly target_duration days.

```
Daily USD = Total USD / Target Duration
Shares[day] = Daily USD / Price[day]
```

**Properties:**
- Duration: Always equals target_duration
- VWAP equals the harmonic mean of prices over the execution period
- No adaptation to market conditions

### 3.2 Strategy 2: Adaptive Execution (Benchmark-Triggered)

**Benchmark Calculation:**
```
Benchmark[t] = (1/t) * sum(Price[1:t])  # Expanding window arithmetic mean
```

**Decision Logic:**

```
base_daily = Total USD / Target Duration
initial_period = min(10, min_duration)

For each day:
    IF day <= initial_period:
        Execute base_daily (constant)

    ELSE IF price < benchmark:  # Favorable - speed up
        IF day > min_duration:
            Execute min(5 * base_daily, remaining)  # ASAP mode
        ELSE:
            Execute min(remaining / days_to_min, 5 * base_daily)  # Speed up

    ELSE:  # Unfavorable - slow down
        IF remaining < 5 * base_daily AND days_to_max > 5:
            Execute 0.1 * base_daily  # Extra slow mode
        ELSE:
            Execute remaining / (days_to_max + 1)  # Normal slow
```

**Key Parameters:**
- initial_period = 10 days (constant buying before adaptive logic)
- max_speedup = 5x base_daily
- extra_slow_multiplier = 0.1x base_daily
- extra_slow_threshold = 5 days

### 3.3 Strategy 3: Adaptive with Discounted Benchmark

Same as Strategy 2, but applies discount to benchmark:

```
Discounted Benchmark = Benchmark * (1 - discount_bps / 10000)
```

This makes prices appear more favorable, resulting in more aggressive buying behavior.

---

## 4. Performance Metrics

### 4.1 VWAP (Volume-Weighted Average Price)

```
VWAP = Total USD Spent / Total Shares Acquired
```

VWAP represents the average price paid per share. Lower is better.

### 4.2 Execution Performance (Basis Points)

```
Performance (bps) = -((VWAP - Benchmark) / Benchmark) * 10000
```

- Positive: VWAP < Benchmark (outperformance - bought cheaper)
- Negative: VWAP > Benchmark (underperformance - bought more expensive)

### 4.3 Standard Error

```
SE = std(values) / sqrt(n)
```

Measures precision of the mean estimate across Monte Carlo simulations.

### 4.4 Win Rate

```
Win Rate = 100 * mean(performance > 0)
```

Percentage of simulations with positive performance.

---

## 5. Default Parameters

### 5.1 Stock Simulation Parameters

```python
TRADING_DAYS_PER_YEAR = 252
DEFAULT_S0 = 100.0              # Starting stock price ($)
DEFAULT_DAYS = 125              # Number of trading days
DEFAULT_MU = 0.0                # Annual drift (0%)
DEFAULT_SIGMA = 25.0            # Annual volatility (25%)
DEFAULT_SIMULATIONS = 10000     # Monte Carlo paths
```

### 5.2 Execution Parameters

```python
DEFAULT_USD = 1_000_000_000     # Total USD ($1 billion)
DEFAULT_MAX_DURATION = 125      # Maximum trading days
DEFAULT_MIN_DURATION_RATIO = 0.6  # Min = 60% of max
DEFAULT_MIN_DURATION = 75       # Calculated: 125 * 0.6
DEFAULT_TARGET_DURATION = 100   # Midpoint: (75 + 125) / 2
DEFAULT_DISCOUNT_BPS = 0        # Basis points discount
```

### 5.3 Strategy Constants

```python
INITIAL_PERIOD_DAYS = 10        # Days before adaptive logic
MAX_SPEEDUP_MULTIPLIER = 5.0    # Max 5x base daily amount
EXTRA_SLOW_MULTIPLIER = 0.1     # 0.1x base when extra slow
EXTRA_SLOW_THRESHOLD_DAYS = 5   # Threshold for extra slow mode
```

---

## 6. User Interface Specification

### 6.1 Page Layout

- **Title:** "A Fixed-Notional Share Buyback Strategy"
- **Subtitle:** "The value of flexible end times and an easy benchmark"
- **Layout:** Wide mode
- **Tabs:** 3 tabs (Simulation & Results, Example, Explanation)

### 6.2 Sidebar Inputs

**Buttons (at top):**
- "Run Simulation" (primary button)
- "Generate Example" (secondary button)

**Stock Parameters:**
- Initial Stock Price ($): 1-200, default 100
- Annual Volatility (%): 0-100, default 25
- Annual Drift (%): -100 to 100, default 0
- Number of Days: 5-300, default 125 (slider)
- Number of Simulations: 1000-100000, default 10000

**Execution Parameters:**
- Total USD to Execute: $1M-$100B, default $1B
- Max Duration (days): 1-300, default 125
- Min Duration (days): 1-300, default 75 (60% of max)
- Target Duration (days): 1-300, default 100 (midpoint)

**Strategy 3 Discount:**
- Benchmark Discount (bps): 0-200, default 0

**Random Seed:**
- Checkbox: "Use Fixed Random Seed"
- Seed input: 0-999999, default 42

### 6.3 Tab 1: Simulation & Results

**Displays:**
1. Simulated Stock Price Paths (Plotly line chart, 100 sample paths)
2. Performance Distributions (3 histograms, one per strategy)
3. Duration Distributions (3 histograms, one per strategy)
4. Summary Statistics Table:
   - Strategy name
   - Mean Performance (bps)
   - SE Performance
   - Mean Duration
   - SE Duration
   - Win Rate (%)

### 6.4 Tab 2: Example

**Displays for each strategy:**
1. Time series chart: Price, Benchmark, VWAP, Performance
2. Daily USD execution bar chart
3. Summary metrics: End Day, VWAP, Benchmark, Performance (bps)

### 6.5 Tab 3: Explanation

Contains mathematical formulas, strategy flowcharts, and numerical examples (embedded markdown/LaTeX).

---

## 7. File Structure

```
ShareBuybackOpus45/
|-- app.py                      # Main Streamlit application
|-- run_app.py                  # PyInstaller launcher script
|-- requirements.txt            # Python dependencies
|-- ShareBuybackApp.spec        # Windows PyInstaller spec
|-- ShareBuybackApp_mac.spec    # macOS PyInstaller spec
|-- .streamlit/
|   |-- config.toml             # Streamlit theme configuration
|-- modules/
|   |-- __init__.py
|   |-- config.py               # Constants and default parameters
|   |-- gbm.py                  # GBM price path generation
|   |-- strategies.py           # Strategy implementations
|   |-- strategies_vectorized.py # Optimized vectorized strategies
|   |-- benchmarks.py           # Benchmark calculations
|   |-- metrics.py              # Performance metrics
|   |-- visualizations.py       # Plotly chart functions
|   |-- cache.py                # Result caching utilities
|-- tests/
|   |-- __init__.py
|   |-- conftest.py             # Pytest fixtures
|   |-- test_gbm.py
|   |-- test_strategies.py
|   |-- test_strategies_vectorized.py
|   |-- test_benchmarks.py
|   |-- test_metrics.py
|   |-- test_visualizations.py
|   |-- test_app_integration.py
|   |-- test_app_regression.py
|-- .github/
|   |-- workflows/
|       |-- build-executables.yml  # GitHub Actions CI/CD
```

---

## 8. Dependencies

```
streamlit>=1.28.0
plotly>=5.17.0
numpy>=1.24.0
pandas>=2.0.0
pytest>=7.4.0
```

For PyInstaller builds, also install:
```
pyarrow
altair
pyinstaller
pydeck
packaging
cachetools
toml
validators
watchdog
pillow
```

---

## 9. Deployment Options

### 9.1 Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 9.2 Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Set main file: app.py
5. Deploy

### 9.3 Standalone Executable (PyInstaller)

**Windows:**
```bash
pyinstaller ShareBuybackApp.spec --clean
# Output: dist/ShareBuybackStrategy2.exe
```

**macOS:**
```bash
pyinstaller ShareBuybackApp_mac.spec --clean
# Output: dist/ShareBuybackStrategy.app
```

---

## 10. Key Implementation Notes

### 10.1 Benchmark Calculation

The benchmark uses an **expanding window arithmetic mean**, NOT a rolling window:
```python
benchmark[t] = np.mean(prices[:t+1])  # All prices from day 1 to day t
```

### 10.2 VWAP as Harmonic Mean

When buying equal dollar amounts each day:
```
VWAP = Total USD / Total Shares
     = n * Daily USD / sum(Daily USD / Price[i])
     = n / sum(1/Price[i])  # Harmonic mean of prices
```

### 10.3 Why Harmonic Mean < Arithmetic Mean

For any set of positive numbers with variance > 0:
```
Harmonic Mean <= Geometric Mean <= Arithmetic Mean
```

This mathematical property means Strategy 1 (uniform execution) tends to achieve VWAP below the arithmetic mean benchmark, giving positive expected performance.

### 10.4 Vectorized Strategy Implementation

For performance with 10,000+ simulations, use vectorized NumPy operations:
```python
def run_all_strategies_vectorized(paths, total_usd, min_dur, max_dur, target_dur, discount_bps):
    # Pre-compute expanding benchmarks for all paths
    benchmarks = np.cumsum(paths, axis=1) / np.arange(1, paths.shape[1]+1)
    # ... vectorized strategy logic
```

### 10.5 Progressive Loading

For better UX, process simulations in batches (1000, 4000, 5000) with progress bar updates.

---

## 11. Testing Requirements

- Unit tests for each module
- Integration tests for app.py
- Regression tests for strategy outputs
- Test fixtures using known random seeds for reproducibility

---

## 12. Critical Formulas Summary

| Formula | Description |
|---------|-------------|
| `S[t+1] = S[t] * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)` | GBM discretization |
| `dt = 1/252` | One trading day in years |
| `Benchmark[t] = mean(P[1:t])` | Expanding window arithmetic mean |
| `VWAP = Total USD / Total Shares` | Volume-weighted average price |
| `Performance = -((VWAP - Benchmark) / Benchmark) * 10000` | Execution performance (bps) |
| `Discounted Benchmark = Benchmark * (1 - discount_bps/10000)` | Strategy 3 benchmark |

---

## 13. Visualization Specifications

### 13.1 Price Paths Chart
- Type: Line chart
- Show: 100 sample paths from n_simulations
- Include: Mean line, sigma envelopes (+/- 1, 2, 3 sigma)
- Colors: Blue paths, black mean, gray envelopes

### 13.2 Performance Histogram
- Type: Histogram
- Bins: Auto
- Show: Mean line (dashed red), zero line
- Title includes strategy name

### 13.3 Duration Histogram
- Type: Histogram
- Show: Min, target, max duration reference lines
- Colors: Green (min), blue (target), red (max)

### 13.4 Example Time Series
- 4-panel subplot: Price, Benchmark, VWAP, Performance
- Show execution end day marker
- Include duration reference lines

---

**END OF SPECIFICATION**

*This document enables full recreation of the Share Buyback Strategy application.*
