# Strategy 1: Uniform Execution

## Overview

The simplest execution strategy: buy a fixed dollar amount each day over the target duration.

<div class="strategy-card">
<strong>Type:</strong> Passive / Non-adaptive<br>
<strong>Completion:</strong> Always 100%<br>
<strong>Performance:</strong> ~0 bps (by design)
</div>

## Algorithm

### Daily Execution Amount

$$
\text{Daily USD} = \frac{\text{Total USD}}{\text{Target Duration}}
$$

For a $1B buyback over 100 days: $10M per day.

### Shares Purchased

$$
\text{Shares}_t = \frac{\text{Daily USD}}{P_t}
$$

Where $P_t$ is the stock price on day $t$.

## Implementation

```python
def strategy_1(prices, total_usd, target_duration):
    """Uniform execution: fixed daily amounts."""
    daily_usd = total_usd / target_duration
    usd_per_day = []
    shares_per_day = []

    for t in range(target_duration):
        price = prices[t]
        shares = daily_usd / price
        usd_per_day.append(daily_usd)
        shares_per_day.append(shares)

    total_shares = sum(shares_per_day)
    vwap = total_usd / total_shares
    benchmark = np.mean(prices[:target_duration])
    performance_bps = (benchmark - vwap) / benchmark * 10000

    return performance_bps, target_duration, vwap, benchmark
```

## Characteristics

### Pros

- Simple and predictable
- Zero market timing risk
- Easy to explain to stakeholders
- Guaranteed completion

### Cons

- No price optimization
- Expected performance = 0 bps
- Ignores market conditions
- May buy at unfavorable prices

## Use Cases

!!! info "When to use Strategy 1"
    - As a baseline for comparing other strategies
    - When market timing is explicitly forbidden
    - For very liquid stocks with minimal price impact
    - When execution predictability is paramount

## Performance Profile

With 10,000 simulations:

| Metric | Typical Value |
|--------|---------------|
| Mean Performance | ~0 bps |
| Std Dev | ~50 bps |
| Min | -150 to -200 bps |
| Max | +150 to +200 bps |

The distribution is symmetric around zero because the strategy has no edge - it simply tracks the market average.
