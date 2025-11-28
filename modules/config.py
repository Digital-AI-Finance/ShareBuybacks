"""
Configuration constants and default parameters for the Share Buyback Strategy application.
"""

# Trading calendar
TRADING_DAYS_PER_YEAR = 252

# Default simulation parameters
DEFAULT_S0 = 100.0  # Starting stock price
DEFAULT_DAYS = 125  # Number of trading days to simulate
DEFAULT_MU = 0.0  # Annual drift (0 = no trend)
DEFAULT_SIGMA = 25.0  # Annual volatility in percentage (25%)
DEFAULT_SIMULATIONS = 10000  # Number of Monte Carlo simulations

# Default execution parameters
DEFAULT_USD = 1_000_000_000  # Total USD to execute ($1 billion)
DEFAULT_MAX_DURATION = 125  # Maximum trading days
DEFAULT_MIN_DURATION_RATIO = 0.6  # Min duration = 60% of max
DEFAULT_DISCOUNT_BPS = 0  # Basis points discount on benchmark

# Derived defaults
DEFAULT_MIN_DURATION = int(DEFAULT_MAX_DURATION * DEFAULT_MIN_DURATION_RATIO)  # 75 days
DEFAULT_TARGET_DURATION = (DEFAULT_MIN_DURATION + DEFAULT_MAX_DURATION) // 2  # 100 days

# Strategy parameters (Strategy 2/3)
INITIAL_PERIOD_DAYS = 10  # Days of constant buying before adaptive logic
MAX_SPEEDUP_MULTIPLIER = 5.0  # Maximum multiple of base daily amount when speeding up
EXTRA_SLOW_MULTIPLIER = 0.1  # Multiplier when very little USD remaining
EXTRA_SLOW_THRESHOLD_DAYS = 5  # Threshold for extra slow mode

# Strategy 4 parameters (Multi-Factor Convex Adaptive)
S4_BETA = 50.0  # Convex sensitivity: higher = more aggressive on big discounts
S4_GAMMA = 1.0  # Urgency acceleration: higher = faster ramp-up near deadline
S4_Z_WINDOW = 20  # Rolling window for z-score calculation (days)
S4_Z_THRESHOLD = 0.0  # Z-score threshold (0 = disabled, always use convex scaling)
S4_MAX_MULTIPLIER = 8.0  # Maximum execution multiplier
S4_MIN_MULTIPLIER = 0.1  # Minimum execution multiplier
S4_SIGNAL_BOOST = 0.0  # Z-score signal boost factor (0 = disabled)

# Strategy 5 parameters (Flexible Completion Adaptive)
S5_MIN_COMPLETION_PCT = 95.0  # Minimum completion percentage (85-100%)
S5_UNFAVORABLE_THRESHOLD = -0.01  # Skip forced completion if deviation below this (-1%)
S5_BETA = 50.0  # Convex sensitivity (same as S4)
S5_GAMMA = 1.0  # Urgency acceleration (same as S4)
S5_MAX_MULTIPLIER = 8.0  # Maximum execution multiplier
S5_MIN_MULTIPLIER = 0.1  # Minimum execution multiplier
