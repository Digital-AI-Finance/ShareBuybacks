# Strategy 2: Adaptive Execution

## Overview

An adaptive strategy that adjusts daily execution based on the current price relative to the rolling TWAP benchmark. Buy more when prices are favorable (below benchmark), less when unfavorable.

<div class="strategy-card">
<strong>Type:</strong> Active / Price-adaptive<br>
<strong>Completion:</strong> Always 100%<br>
<strong>Performance:</strong> +15-25 bps typical
</div>

## Algorithm

### Core Logic

1. Calculate rolling TWAP benchmark up to current day
2. Compare current price to benchmark
3. If price < benchmark: speed up (buy more)
4. If price > benchmark: slow down (buy less)
5. Ensure completion by max_duration

### Execution Multiplier

$$
\text{multiplier} = \begin{cases}
\text{speedup} & \text{if } P_t < \text{TWAP}_t \\
\text{slowdown} & \text{if } P_t > \text{TWAP}_t
\end{cases}
$$

Where:
- **Speedup** range: 1.0 to MAX_SPEEDUP (5x default)
- **Slowdown** range: 0.1 to 1.0

### Urgency Factor

As the deadline approaches, urgency increases:

$$
\text{urgency} = \frac{\text{days\_remaining}}{\max(\text{usd\_remaining} / \text{base\_daily}, 1)}
$$

## Implementation

```python
def strategy_2(prices, benchmarks, total_usd, min_dur, max_dur, target_dur):
    """Adaptive execution: buy more when cheap, less when expensive."""
    base_daily = total_usd / target_dur
    usd_remaining = total_usd

    for t in range(max_dur):
        price = prices[t]
        benchmark = benchmarks[t]
        deviation = (benchmark - price) / benchmark

        # Adaptive multiplier
        if deviation > 0:  # Price below benchmark (favorable)
            multiplier = 1.0 + deviation * MAX_SPEEDUP
        else:  # Price above benchmark (unfavorable)
            multiplier = max(0.1, 1.0 + deviation)

        # Urgency adjustment
        days_left = max_dur - t
        if days_left <= 5 and usd_remaining > base_daily * days_left:
            multiplier = max(multiplier, usd_remaining / (base_daily * days_left))

        daily_usd = min(base_daily * multiplier, usd_remaining)
        # ... execute and track
```

## Characteristics

### Pros

- Exploits price discounts systematically
- Self-adjusting to market conditions
- Guaranteed completion
- Moderate risk profile

### Cons

- May delay execution in rising markets
- Performance depends on volatility
- Can be forced to buy at deadline

## Performance Profile

| Metric | Typical Value |
|--------|---------------|
| Mean Performance | +15 to +25 bps |
| Std Dev | ~60-70 bps |
| Min | -100 to -150 bps |
| Max | +200 to +300 bps |

!!! success "Key Insight"
    Strategy 2 generates alpha by systematically buying more at discounts and less at premiums, creating a slight positive skew in the performance distribution.
