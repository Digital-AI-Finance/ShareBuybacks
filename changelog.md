# Changelog

All notable changes to the Share Buyback Strategy App.

## [1.3.0] - 2025-11-27

### Added
- PyInstaller packaging for standalone Windows executable
  - `run_app.py` - Streamlit launcher script
  - `ShareBuybackApp.spec` - PyInstaller specification file
  - `build_minimal.py` - Automated build script with minimal venv
- Standalone executable: `dist/ShareBuybackStrategy.exe` (~85 MB)

### Technical
- Uses minimal virtual environment for smaller executable size
- Includes package metadata for importlib compatibility
- Configured for offline operation (no usage stats, headless mode)

## [1.2.0] - 2025-11-26

### Added
- NumPy vectorized strategy implementations (25.6x speedup)
  - `modules/strategies_vectorized.py` with vectorized versions of all strategies
  - `precompute_benchmarks()` for efficient expanding mean calculation
  - `strategy_1_vectorized()`, `strategy_2_vectorized()`, `strategy_3_vectorized()`
  - `run_all_strategies_vectorized()` combined runner
- Caching utilities (`modules/cache.py`)
  - GBM paths caching (instant re-runs when only strategy params change)
  - Results caching with hash-based keys
  - FIFO cache eviction (3 path sets, 5 result sets)
- Progressive loading with progress bar
  - Batch processing (1000, 4000, 5000 simulations per batch)
  - Real-time progress updates
- 17 new tests for vectorized strategies (`tests/test_strategies_vectorized.py`)

### Performance
- 10,000 simulations: ~0.5s (was ~12.8s)
- Overall speedup: 25.6x with NumPy vectorization

## [1.1.0] - 2025-11-26

### Changed
- Fixed sidebar input ranges to match specification:
  - Initial price: 1-200 (was 1-10000)
  - Days: slider 5-300 step 5 (was number_input 50-504)
  - Volatility: 0-100% (was 1-200%)
  - Simulations: 1000-100000 step 1000 (was 100-100000 step 100)
  - Discount: 0-200 bps (was 0-500)

### Added
- Direct input for min_duration (was computed from percentage)
- Direct input for target_duration (was computed from midpoint)
- Strategy 2 decision flowchart in Explanation tab
- Detailed numerical trace example showing Strategy 2 day-by-day decisions
- Validation warning when n_days < max_duration

### Fixed
- Moved "Generate Example" button to top of sidebar (alongside "Run Simulation")
- Removed duplicate button definition from Tab 2

## [1.0.0] - 2025-11-26

### Added
- Initial release with all core functionality
- GBM simulation with configurable drift and volatility
- Three trading strategies:
  - Strategy 1: Uniform execution over target duration
  - Strategy 2: Adaptive execution with flexible end time
  - Strategy 3: Adaptive with discounted benchmark
- Tab 1: Simulation & Results with:
  - Price paths chart with sigma envelopes
  - Performance histograms (250 bins)
  - Duration histograms (250 bins)
  - Summary statistics table
- Tab 2: Example with:
  - Single-path execution visualization
  - Daily USD execution bar charts
  - Per-strategy performance metrics
- Tab 3: Explanation with:
  - GBM mathematical formulation
  - Benchmark calculation formulas
  - Strategy logic descriptions
  - VWAP and performance metrics
- 94 unit tests covering:
  - GBM generation and reproducibility
  - Benchmark calculations
  - Strategy execution logic
  - Metrics computation
  - Visualization functions
  - Integration tests
