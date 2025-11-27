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

# Strategy parameters
INITIAL_PERIOD_DAYS = 10  # Days of constant buying before adaptive logic
MAX_SPEEDUP_MULTIPLIER = 5.0  # Maximum multiple of base daily amount when speeding up
EXTRA_SLOW_MULTIPLIER = 0.1  # Multiplier when very little USD remaining
EXTRA_SLOW_THRESHOLD_DAYS = 5  # Threshold for extra slow mode
