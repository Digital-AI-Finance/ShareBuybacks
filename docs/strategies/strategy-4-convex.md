# Strategy 4: Convex Adaptive

## Overview

The most sophisticated guaranteed-completion strategy. Uses **exponential (convex) scaling** to make proportionally larger bets on larger discounts, combined with time-based urgency acceleration.

<div class="strategy-card">
<strong>Type:</strong> Active / Convex-adaptive<br>
<strong>Completion:</strong> Always 100%<br>
<strong>Performance:</strong> +25-40 bps typical
</div>

## Algorithm

### Convex Scaling

Instead of linear response to discounts, Strategy 4 uses exponential scaling:

$$
\text{convex\_factor} = e^{\beta \cdot \text{deviation}} - 1
$$

Where:
- $\beta$ = 50 (default sensitivity)
- $\text{deviation} = \frac{\text{TWAP} - P_t}{\text{TWAP}}$

### Urgency Acceleration

Time pressure increases as deadline approaches:

$$
\text{urgency\_factor} = \left(\frac{t}{\text{max\_duration}}\right)^{\gamma}
$$

Where $\gamma$ = 1.0 controls acceleration steepness.

### Combined Multiplier

$$
\text{multiplier} = \text{base} + \text{convex\_factor} \times (1 + \text{urgency})
$$

Clamped between MIN_MULTIPLIER (0.1) and MAX_MULTIPLIER (8.0).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| beta | 50.0 | Convex sensitivity |
| gamma | 1.0 | Urgency acceleration |
| max_multiplier | 8.0 | Maximum speed-up |
| min_multiplier | 0.1 | Minimum (slowdown) |

## Implementation

```python
def strategy_4(prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
               beta=50.0, gamma=1.0):
    """Multi-factor convex adaptive execution."""
    for t in range(max_dur):
        price = prices[t]
        benchmark = benchmarks[t]
        deviation = (benchmark - price) / benchmark

        # Convex scaling
        if deviation > 0:
            convex_factor = math.exp(beta * deviation) - 1
        else:
            convex_factor = -math.exp(-beta * deviation) + 1

        # Time urgency
        progress = t / max_dur
        urgency = progress ** gamma

        # Combined multiplier
        multiplier = 1.0 + convex_factor * (1 + urgency)
        multiplier = max(MIN_MULT, min(MAX_MULT, multiplier))

        daily_usd = base_daily * multiplier
        # ...
```

## Why Convex?

<div class="formula-box">
Linear scaling: 5% discount = 5x response<br>
Convex scaling: 5% discount = ~12x response (exponential)
</div>

This means:
- Small discounts: modest response
- Large discounts: aggressive response
- The strategy "saves ammunition" for the best opportunities

## Performance Profile

| Metric | Typical Value |
|--------|---------------|
| Mean Performance | +25 to +40 bps |
| Std Dev | ~70-85 bps |
| Min | -100 to -180 bps |
| Max | +250 to +400 bps |

!!! warning "Higher variance"
    The aggressive convex scaling produces higher variance. This strategy is best for those seeking maximum expected return over many executions.

## Tuning Beta

| Beta | Behavior |
|------|----------|
| 25 | Conservative, closer to linear |
| 50 | Balanced (default) |
| 75 | Aggressive |
| 100 | Very aggressive |
