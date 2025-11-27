# Share Buyback Strategy Modules
"""
This package contains modules for simulating and analyzing share buyback strategies.
"""

from .config import *
from .gbm import generate_gbm_paths
from .benchmarks import compute_benchmark
from .strategies import strategy_1, strategy_2, strategy_3
from .metrics import vwap, execution_performance_bps, standard_error
