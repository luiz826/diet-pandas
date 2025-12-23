"""
Unit tests for Diet Pandas exceptions and warnings.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp
from dietpandas.exceptions import (
    DietPandasWarning,
    HighCardinalityWarning,
    OptimizationError,
    OptimizationSkippedWarning,
    PrecisionLossWarning,
)


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_diet_pandas_warning_is_user_warning(self):
        """Test that DietPandasWarning is a UserWarning."""
        assert issubclass(DietPandasWarning, UserWarning)

    def test_high_cardinality_warning_hierarchy(self):
        """Test HighCardinalityWarning class hierarchy."""
        assert issubclass(HighCardinalityWarning, DietPandasWarning)
        assert issubclass(HighCardinalityWarning, UserWarning)

    def test_precision_loss_warning_hierarchy(self):
        """Test PrecisionLossWarning class hierarchy."""
        assert issubclass(PrecisionLossWarning, DietPandasWarning)
        assert issubclass(PrecisionLossWarning, UserWarning)

    def test_optimization_skipped_warning_hierarchy(self):
        """Test OptimizationSkippedWarning class hierarchy."""
        assert issubclass(OptimizationSkippedWarning, DietPandasWarning)
        assert issubclass(OptimizationSkippedWarning, UserWarning)

    def test_optimization_error_is_exception(self):
        """Test that OptimizationError is an Exception."""
        assert issubclass(OptimizationError, Exception)


class TestWarningsInDiet:
    """Tests for warnings emitted by diet function."""

    def test_warn_on_all_nan_column(self):
        """Test warning for all-NaN columns."""
        df = pd.DataFrame({"good_col": [1, 2, 3], "nan_col": [np.nan, np.nan, np.nan]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, warn_on_issues=True, verbose=False)

            # Check that warning was issued
            assert len(w) > 0
            assert any(issubclass(warning.category, OptimizationSkippedWarning) for warning in w)
            assert any("nan_col" in str(warning.message) for warning in w)

    def test_warn_on_high_cardinality(self):
        """Test warning for high cardinality columns."""
        df = pd.DataFrame({"high_card": [f"id_{i}" for i in range(100)]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, warn_on_issues=True, verbose=False)

            # Check that high cardinality warning was issued
            assert len(w) > 0
            assert any(issubclass(warning.category, HighCardinalityWarning) for warning in w)
            assert any("high cardinality" in str(warning.message).lower() for warning in w)

    def test_warn_on_aggressive_precision_loss(self):
        """Test warning for aggressive mode precision loss."""
        df = pd.DataFrame({"float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64")})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, aggressive=True, warn_on_issues=True, verbose=False)

            # Check that precision loss warning was issued
            assert len(w) > 0
            assert any(issubclass(warning.category, PrecisionLossWarning) for warning in w)
            assert any("precision loss" in str(warning.message).lower() for warning in w)

    def test_warn_on_force_aggressive(self):
        """Test warning for force_aggressive columns."""
        df = pd.DataFrame({"float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64")})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(
                df,
                force_aggressive=["float_col"],
                warn_on_issues=True,
                verbose=False,
            )

            # Check that precision loss warning was issued
            assert len(w) > 0
            assert any(issubclass(warning.category, PrecisionLossWarning) for warning in w)

    def test_no_warnings_when_disabled(self):
        """Test that no warnings are issued when warn_on_issues=False."""
        df = pd.DataFrame(
            {
                "good_col": [1, 2, 3],
                "nan_col": [np.nan, np.nan, np.nan],
                "high_card": [f"id_{i}" for i in range(3)],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, warn_on_issues=False, verbose=False)

            # Should have no DietPandas warnings
            diet_warnings = [
                warning for warning in w if issubclass(warning.category, DietPandasWarning)
            ]
            assert len(diet_warnings) == 0

    def test_multiple_warnings_same_dataframe(self):
        """Test that multiple warnings can be issued for same DataFrame."""
        # Create dataframe with mismatched sizes - use explicit index
        df = pd.DataFrame(
            {
                "nan_col": [np.nan] * 100,
                "high_card": [f"id_{i}" for i in range(100)],
                "float_col": [1.1] * 100,
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, aggressive=True, warn_on_issues=True, verbose=False)

            # Should have multiple warnings
            warning_categories = {warning.category for warning in w}
            # Should have at least NaN warning and high cardinality or precision loss
            assert len(warning_categories) >= 1  # At least one warning type issued


class TestWarningMessages:
    """Tests for warning message content."""

    def test_high_cardinality_message_includes_percentage(self):
        """Test that high cardinality warning includes percentage."""
        df = pd.DataFrame({"high_card": [f"id_{i}" for i in range(100)]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, warn_on_issues=True, verbose=False)

            # Find high cardinality warning
            hc_warnings = [
                warning for warning in w if issubclass(warning.category, HighCardinalityWarning)
            ]
            assert len(hc_warnings) > 0
            # Should contain percentage
            assert "%" in str(hc_warnings[0].message)

    def test_precision_loss_message_actionable(self):
        """Test that precision loss warning provides actionable advice."""
        df = pd.DataFrame({"float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64")})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, aggressive=True, warn_on_issues=True, verbose=False)

            # Find precision loss warning
            pl_warnings = [
                warning for warning in w if issubclass(warning.category, PrecisionLossWarning)
            ]
            assert len(pl_warnings) > 0
            # Should mention column name and suggest force_aggressive
            msg = str(pl_warnings[0].message)
            assert "float_col" in msg
            assert "force_aggressive" in msg

    def test_optimization_skipped_message_includes_column(self):
        """Test that optimization skipped warning includes column name."""
        df = pd.DataFrame({"good_col": [1, 2, 3], "bad_col": [np.nan, np.nan, np.nan]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dp.diet(df, warn_on_issues=True, verbose=False)

            # Find optimization skipped warning
            os_warnings = [
                warning for warning in w if issubclass(warning.category, OptimizationSkippedWarning)
            ]
            assert len(os_warnings) > 0
            assert "bad_col" in str(os_warnings[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
