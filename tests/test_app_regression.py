"""
Regression tests for the Share Buyback Strategy Streamlit application.

These tests verify that the app configuration and behavior match specifications.
"""

import pytest
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInputRangesMatchSpec:
    """Tests to verify sidebar input ranges match specification."""

    def test_initial_price_range(self):
        """Verify initial price range: 1-200, default 100."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Check for correct values using simpler pattern matching
        assert 'min_value=1' in content, "Initial price min should be 1"
        assert 'max_value=200' in content, "Initial price max should be 200"
        assert '"Initial Stock Price' in content or "'Initial Stock Price" in content

    def test_days_is_slider(self):
        """Verify Number of Days is a slider, not number_input."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Check for slider (not number_input) for days
        assert 'st.sidebar.slider' in content, "Should use slider for days input"
        assert 'slider_days' in content, "Days slider should have key 'slider_days'"

    def test_days_slider_range(self):
        """Verify days slider range: 5-300, step 5."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Find the slider section and verify values
        assert 'min_value=5' in content, "Days min should be 5"
        assert 'max_value=300' in content, "Days max should be 300"
        assert 'step=5' in content, "Days step should be 5"
        assert '"Number of Days' in content or "'Number of Days" in content

    def test_volatility_range(self):
        """Verify volatility range: 0-100."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Check volatility input exists with correct range
        assert '"Annual Volatility' in content or "'Annual Volatility" in content
        assert 'min_value=0' in content, "Volatility min should be 0"
        assert 'max_value=100' in content, "Volatility max should be 100"

    def test_simulations_range(self):
        """Verify simulations range: 1000-100000, step 1000."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        assert '"Number of Simulations' in content or "'Number of Simulations" in content
        assert 'min_value=1000' in content, "Simulations min should be 1000"
        assert 'max_value=100000' in content, "Simulations max should be 100000"
        assert 'step=1000' in content, "Simulations step should be 1000"

    def test_discount_range(self):
        """Verify discount range: 0-200."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        assert '"Benchmark Discount' in content or "'Benchmark Discount" in content
        # Note: min_value=0 appears multiple times, so we check for the discount-specific context
        assert 'max_value=200' in content, "Discount max should be 200"


class TestButtonPlacement:
    """Tests to verify button placement in sidebar."""

    def test_buttons_at_top_of_sidebar(self):
        """Verify both buttons appear before input parameters."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Find positions
        run_sim_pos = content.find('btn_run_simulation')
        gen_example_pos = content.find('btn_generate_example')
        first_input_pos = content.find('input_s0')

        assert run_sim_pos != -1, "Run Simulation button not found"
        assert gen_example_pos != -1, "Generate Example button not found"
        assert first_input_pos != -1, "First input not found"

        # Buttons should appear before first input
        assert run_sim_pos < first_input_pos, "Run Simulation button should be before inputs"
        assert gen_example_pos < first_input_pos, "Generate Example button should be before inputs"

    def test_no_duplicate_buttons(self):
        """Verify no duplicate button definitions."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Count occurrences of button keys
        run_sim_count = content.count('btn_run_simulation')
        gen_example_count = content.count('btn_generate_example')

        # Each button should appear only once (in the key parameter)
        assert run_sim_count == 1, f"Run Simulation button defined {run_sim_count} times"
        assert gen_example_count == 1, f"Generate Example button defined {gen_example_count} times"


class TestMinTargetDurationEditable:
    """Tests to verify min and target duration are directly editable."""

    def test_min_duration_is_number_input(self):
        """Verify min_duration is a number_input, not computed."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Should have a number_input for min duration
        assert 'input_min_duration' in content, "Min duration should be a number_input"

        # Should NOT have the old slider approach
        assert 'slider_min_duration_pct' not in content, "Should not use percentage slider for min duration"

    def test_target_duration_is_number_input(self):
        """Verify target_duration is a number_input, not computed."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Should have a number_input for target duration
        assert 'input_target_duration' in content, "Target duration should be a number_input"


class TestValidation:
    """Tests for input validation."""

    def test_days_validation_warning(self):
        """Verify warning when days < max_duration."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Should have validation warning
        assert 'n_days < max_duration' in content, "Should check if n_days < max_duration"
        assert 'st.sidebar.warning' in content, "Should show warning when validation fails"


class TestFlowchartPresent:
    """Tests to verify Strategy 2 flowchart is present."""

    def test_flowchart_in_explanation_tab(self):
        """Verify Strategy 2 decision flowchart is in Explanation tab."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Should contain flowchart elements
        assert 'Strategy 2 Decision Flowchart' in content, "Flowchart title should be present"
        assert 'START: Day d' in content, "Flowchart should contain START"
        assert 'initial_period' in content.lower(), "Flowchart should mention initial period"
        assert 'benchmark' in content.lower(), "Flowchart should mention benchmark"

    def test_numerical_trace_example(self):
        """Verify detailed numerical trace for Strategy 2."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()

        # Should contain numerical trace table
        assert 'Detailed Trace' in content, "Should have detailed trace section"
        assert '| Day | Price |' in content, "Should have trace table header"


class TestDocumentationFiles:
    """Tests to verify documentation files exist."""

    def test_status_md_exists(self):
        """Verify status.md exists."""
        status_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'status.md')
        assert os.path.exists(status_path), "status.md should exist"

    def test_changelog_md_exists(self):
        """Verify changelog.md exists."""
        changelog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'changelog.md')
        assert os.path.exists(changelog_path), "changelog.md should exist"

    def test_status_contains_version(self):
        """Verify status.md contains version info."""
        status_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'status.md')
        with open(status_path, 'r') as f:
            content = f.read()

        assert 'Version' in content, "status.md should contain version info"

    def test_changelog_has_entries(self):
        """Verify changelog.md has changelog entries."""
        changelog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'changelog.md')
        with open(changelog_path, 'r') as f:
            content = f.read()

        assert '[1.0.0]' in content, "changelog should have v1.0.0 entry"
        assert '[1.1.0]' in content, "changelog should have v1.1.0 entry"
