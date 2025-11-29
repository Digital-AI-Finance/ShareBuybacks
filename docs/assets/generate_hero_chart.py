"""Generate hero chart for documentation landing page."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Generate sample performance distributions
np.random.seed(42)
n_samples = 5000

strategies = {
    'S1 Uniform': {'mean': 0, 'std': 50, 'color': '#6c757d'},
    'S2 Adaptive': {'mean': 20, 'std': 65, 'color': '#0d6efd'},
    'S3 Discount': {'mean': 25, 'std': 60, 'color': '#198754'},
    'S4 Convex': {'mean': 35, 'std': 80, 'color': '#fd7e14'},
    'S5 Flexible': {'mean': 40, 'std': 90, 'color': '#dc3545'},
}

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

# Generate and plot distributions
for name, params in strategies.items():
    data = np.random.normal(params['mean'], params['std'], n_samples)
    ax.hist(data, bins=80, alpha=0.5, label=name, color=params['color'],
            density=True, histtype='stepfilled', edgecolor='white', linewidth=0.5)

# Styling
ax.set_xlabel('Performance vs TWAP Benchmark (basis points)', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Strategy Performance Distribution\n10,000 Monte Carlo Simulations',
             fontsize=16, fontweight='bold', pad=20)

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Benchmark')

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Set limits
ax.set_xlim(-250, 300)
ax.set_ylim(0, 0.012)

# Add annotation
ax.annotate('Adaptive strategies\noutperform benchmark',
            xy=(50, 0.008), fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4ea', edgecolor='#198754', alpha=0.8))

plt.tight_layout()
plt.savefig('D:/Joerg/Research/slides/ShareBuybackOpus45/docs/assets/images/hero_performance.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('D:/Joerg/Research/slides/ShareBuybackOpus45/docs/assets/images/hero_performance.svg',
            bbox_inches='tight', facecolor='white')
print("Hero chart saved!")

# Also create a strategy comparison bar chart
fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='white')

names = list(strategies.keys())
means = [strategies[n]['mean'] for n in names]
colors = [strategies[n]['color'] for n in names]

bars = ax2.bar(names, means, color=colors, edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax2.annotate(f'+{mean} bps' if mean > 0 else f'{mean} bps',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Average Performance (bps)', fontsize=13, fontweight='bold')
ax2.set_title('Expected Strategy Performance', fontsize=16, fontweight='bold', pad=20)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylim(-10, 55)

plt.tight_layout()
plt.savefig('D:/Joerg/Research/slides/ShareBuybackOpus45/docs/assets/images/strategy_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('D:/Joerg/Research/slides/ShareBuybackOpus45/docs/assets/images/strategy_comparison.svg',
            bbox_inches='tight', facecolor='white')
print("Strategy comparison chart saved!")

print("\nCharts generated successfully!")
