# Strategy 5: Flexible Completion

## Overview

The most advanced strategy, combining convex adaptive execution with **optional partial completion**. At the deadline, if prices are unfavorable, Strategy 5 can choose not to complete the full execution.

<div class="strategy-card">
<strong>Type:</strong> Active / Convex + Flexible<br>
<strong>Completion:</strong> Variable (85-100%)<br>
<strong>Performance:</strong> +30-45 bps typical
</div>

## Algorithm

### Core Innovation

Strategy 5 is identical to Strategy 4 during execution, but at the deadline:

1. Check if minimum completion threshold is met
2. If price is unfavorable (above benchmark by threshold), stop
3. Otherwise, force complete as usual

### Decision Logic

At max_duration:

$$
\text{skip\_forced} = \begin{cases}
\text{true} & \text{if completion} \geq \text{min\_pct AND deviation} < \text{threshold} \\
\text{false} & \text{otherwise}
\end{cases}
$$

Where:
- `min_pct` = Minimum completion percentage (e.g., 95%)
- `threshold` = Unfavorable deviation threshold (e.g., -1%)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| min_completion_pct | 95% | Minimum required completion |
| unfavorable_threshold | -1% | Skip if deviation below this |
| beta, gamma | Same as S4 | Convex parameters |

## Implementation

```python
def strategy_5(prices, benchmarks, total_usd, min_dur, max_dur, target_dur,
               min_completion_pct=95.0, unfavorable_threshold=-0.01):
    """Flexible completion: can skip forced buying at bad prices."""

    # Same as Strategy 4 during execution...

    # At deadline (day = max_dur - 1):
    if usd_remaining > 0:
        current_completion = (total_usd - usd_remaining) / total_usd * 100
        deviation = (benchmark - price) / benchmark

        # Decision: force complete or skip?
        if (current_completion >= min_completion_pct and
            deviation < unfavorable_threshold):
            # Skip forced completion - prices unfavorable
            pass
        else:
            # Force complete remaining
            execute(usd_remaining)
```

## Completion Trade-off

| Min Completion | Behavior | Risk |
|----------------|----------|------|
| 100% | Same as Strategy 4 | None |
| 95% | Skip up to 5% at bad prices | Low |
| 90% | Skip up to 10% | Medium |
| 85% | Skip up to 15% | Higher |

!!! warning "Execution risk"
    Lower min_completion means more USD may remain unexecuted. Ensure this aligns with your mandate.

## Performance Profile

| Metric | Typical Value (95% min) |
|--------|-------------------------|
| Mean Performance | +30 to +45 bps |
| Std Dev | ~80-100 bps |
| Avg Completion | 97-99% |
| Min Completion | 95% |

## When to Use

!!! success "Ideal scenarios"
    - Flexible mandates without strict completion requirements
    - When maximizing execution value is paramount
    - Markets with occasional sharp adverse moves

!!! failure "Avoid when"
    - Mandate requires 100% completion
    - Regulatory constraints on execution
    - Low volatility markets (little benefit)

## Comparison with Strategy 4

| Aspect | Strategy 4 | Strategy 5 |
|--------|------------|------------|
| Completion | Always 100% | Variable |
| Mean Performance | Lower | Higher |
| Worst-case | Forced buying | May not complete |
| Complexity | Medium | Higher |
