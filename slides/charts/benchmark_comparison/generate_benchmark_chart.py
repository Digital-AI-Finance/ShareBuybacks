"""
Generate Benchmark Comparison Chart showing:
- Price path (GBM)
- Expanding window mean (current benchmark)
- Rolling window mean (alternative)

Output: benchmark_comparison.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def generate_gbm_path(s0=100, mu=0.05, sigma=0.25, n_days=125, seed=42):
    """Generate a single GBM price path."""
    np.random.seed(seed)
    dt = 1/252
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    log_returns = drift + diffusion * np.random.randn(n_days)
    prices = s0 * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, s0)[:n_days]

    return prices

def expanding_mean(prices):
    """Compute expanding window mean."""
    cumsum = np.cumsum(prices)
    counts = np.arange(1, len(prices) + 1)
    return cumsum / counts

def rolling_mean(prices, window=20):
    """Compute rolling window mean."""
    result = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        start = max(0, i - window + 1)
        result[i] = np.mean(prices[start:i+1])
    return result

def create_chart():
    # Generate price path with upward trend
    prices = generate_gbm_path(s0=100, mu=0.10, sigma=0.25, n_days=100, seed=123)

    # Compute benchmarks
    expanding = expanding_mean(prices)
    rolling_20 = rolling_mean(prices, window=20)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    days = np.arange(len(prices))

    # Plot
    ax.plot(days, prices, 'b-', linewidth=2, label='Price', alpha=0.8)
    ax.plot(days, expanding, 'r--', linewidth=2, label='Expanding Mean (benchmark)', alpha=0.8)
    ax.plot(days, rolling_20, 'g-.', linewidth=2, label='Rolling 20-day Mean', alpha=0.8)

    # Highlight lag region
    ax.fill_between(days[50:80], prices[50:80], expanding[50:80],
                    alpha=0.3, color='red', label='Benchmark Lag')

    ax.set_xlabel('Trading Day', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Benchmark Lag in Trending Markets', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Expanding mean lags\nin uptrending market',
                xy=(65, expanding[65]), xytext=(75, 95),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    fig = create_chart()

    output_path = os.path.join(script_dir, 'benchmark_comparison.pdf')
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Generated: {output_path}")
