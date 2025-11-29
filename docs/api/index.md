# API Reference

## Module Overview

```
modules/
    config.py              # Configuration constants
    gbm.py                 # Price simulation
    strategies.py          # Single-path strategies
    strategies_vectorized.py  # Vectorized strategies
```

## config.py

Default parameters for simulations.

### Trading Calendar

```python
TRADING_DAYS_PER_YEAR = 252
```

### Simulation Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_S0` | 100.0 | Starting stock price |
| `DEFAULT_DAYS` | 125 | Trading days |
| `DEFAULT_MU` | 0.0 | Annual drift |
| `DEFAULT_SIGMA` | 25.0 | Annual volatility (%) |
| `DEFAULT_SIMULATIONS` | 10000 | Monte Carlo paths |

### Execution Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_USD` | 1B | Total USD |
| `DEFAULT_MAX_DURATION` | 125 | Max days |
| `DEFAULT_MIN_DURATION` | 75 | Min days |
| `DEFAULT_TARGET_DURATION` | 100 | Target days |

### Strategy Parameters

| Constant | Default | Description |
|----------|---------|-------------|
| `S4_BETA` | 50.0 | Convex sensitivity |
| `S4_GAMMA` | 1.0 | Urgency acceleration |
| `S5_MIN_COMPLETION_PCT` | 95.0 | Min completion % |

---

## gbm.py

Geometric Brownian Motion price simulation.

### generate_gbm_paths

```python
def generate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    n_days: int,
    n_sims: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate GBM price paths.

    Args:
        S0: Initial stock price
        mu: Annual drift (0.0 = no trend)
        sigma: Annual volatility (0.25 = 25%)
        n_days: Number of trading days
        n_sims: Number of simulation paths
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (n_sims, n_days) price matrix
    """
```

**Example:**

```python
prices = generate_gbm_paths(
    S0=100, mu=0.0, sigma=0.25,
    n_days=125, n_sims=10000, seed=42
)
# prices.shape = (10000, 125)
```

---

## strategies.py

Single-path strategy implementations.

### strategy_1

```python
def strategy_1(
    prices: np.ndarray,
    total_usd: float,
    target_duration: int
) -> Tuple[float, int, float, float]:
    """
    Uniform execution strategy.

    Returns:
        performance_bps, duration, vwap, benchmark
    """
```

### strategy_2

```python
def strategy_2(
    prices: np.ndarray,
    benchmarks: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int
) -> Tuple[float, int, float, float]:
    """
    Adaptive execution strategy.
    """
```

### strategy_3, strategy_4, strategy_5

Similar signatures with strategy-specific parameters.

---

## strategies_vectorized.py

Vectorized implementations for fast Monte Carlo.

### precompute_benchmarks

```python
def precompute_benchmarks(prices: np.ndarray) -> np.ndarray:
    """
    Compute rolling TWAP benchmarks for all paths.

    Args:
        prices: Shape (n_sims, n_days)

    Returns:
        benchmarks: Shape (n_sims, n_days)
    """
```

### strategy_4_vectorized

```python
def strategy_4_vectorized(
    prices: np.ndarray,
    benchmarks: np.ndarray,
    total_usd: float,
    min_duration: int,
    max_duration: int,
    target_duration: int,
    beta: float = 50.0,
    gamma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Strategy 4 across all paths.

    Returns:
        performances: Shape (n_sims,)
        durations: Shape (n_sims,)
        vwaps: Shape (n_sims,)
        final_benchmarks: Shape (n_sims,)
    """
```

---

## Usage Example

```python
import numpy as np
from modules.gbm import generate_gbm_paths
from modules.strategies_vectorized import (
    precompute_benchmarks,
    strategy_1_vectorized,
    strategy_4_vectorized
)

# Generate paths
prices = generate_gbm_paths(100, 0, 0.25, 125, 10000, seed=42)
benchmarks = precompute_benchmarks(prices)

# Compare strategies
p1, d1, v1, b1 = strategy_1_vectorized(prices, 1e9, 100)
p4, d4, v4, b4 = strategy_4_vectorized(prices, benchmarks, 1e9, 75, 125, 100)

print(f"S1: {p1.mean():.2f} +/- {p1.std():.2f} bps")
print(f"S4: {p4.mean():.2f} +/- {p4.std():.2f} bps")
```
