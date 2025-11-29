# Strategy 3: Discounted Benchmark

## Overview

A conservative variant of Strategy 2 that uses a discounted benchmark for comparison. This makes the strategy more selective about when to accelerate, only buying aggressively at larger discounts.

<div class="strategy-card">
<strong>Type:</strong> Active / Conservative-adaptive<br>
<strong>Completion:</strong> Always 100%<br>
<strong>Performance:</strong> +20-30 bps typical
</div>

## Algorithm

### Discounted Benchmark

Instead of comparing to raw TWAP, Strategy 3 uses a discounted version:

$$
\text{Adjusted Benchmark} = \text{TWAP} \times (1 - \text{discount})
$$

For a 50 bps discount:
$$
\text{Adjusted} = \text{TWAP} \times 0.995
$$

### Effect

- **More conservative**: Only speeds up when price is significantly below TWAP
- **Higher bar**: Requires ~0.5% discount before accelerating
- **Reduced whipsaw**: Less responsive to minor price fluctuations

## Implementation

```python
def strategy_3(prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
               discount_bps=50):
    """Adaptive with discounted benchmark."""
    discount = discount_bps / 10000  # 50 bps = 0.005

    for t in range(max_dur):
        price = prices[t]
        raw_benchmark = benchmarks[t]

        # Apply discount to benchmark
        adjusted_benchmark = raw_benchmark * (1 - discount)

        # Compare price to discounted benchmark
        deviation = (adjusted_benchmark - price) / adjusted_benchmark

        # Same adaptive logic as Strategy 2...
```

## Discount Selection

| Discount (bps) | Effect |
|----------------|--------|
| 0 | Same as Strategy 2 |
| 25 | Slightly more conservative |
| 50 | Moderate selectivity |
| 100 | Very selective acceleration |
| 200 | Rarely accelerates |

!!! tip "Recommended"
    A discount of 25-75 bps typically provides the best risk-adjusted performance.

## Characteristics

### Pros

- More consistent performance
- Lower variance than Strategy 2
- Reduces "false positive" accelerations
- Better for range-bound markets

### Cons

- May miss small opportunities
- Slightly lower mean performance in trending markets
- Still forced to complete by deadline

## Performance Profile

| Metric | Typical Value |
|--------|---------------|
| Mean Performance | +20 to +30 bps |
| Std Dev | ~55-65 bps |
| Min | -80 to -120 bps |
| Max | +180 to +250 bps |

The reduced variance makes Strategy 3 attractive for risk-sensitive mandates.
