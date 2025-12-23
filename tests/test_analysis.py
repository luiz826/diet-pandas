"""
Unit tests for Diet Pandas analysis module.
"""

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp
from dietpandas.analysis import (
    analyze,
    estimate_memory_reduction,
    get_optimization_summary,
)


class TestAnalyze:
    """Tests for the analyze function."""

    def test_analyze_basic_dataframe(self):
        """Test that analyze returns correct structure."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="int64"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
                "cat_col": ["A", "B", "A"],
            }
        )

        result = analyze(df)

        assert "column" in result.columns
        assert "current_dtype" in result.columns
        assert "recommended_dtype" in result.columns
        assert "current_memory_mb" in result.columns
        assert "optimized_memory_mb" in result.columns
        assert "savings_mb" in result.columns
        assert "savings_percent" in result.columns
        assert "reasoning" in result.columns

        assert len(result) == 3  # Three columns

    def test_analyze_shows_int_optimization(self):
        """Test that analyze identifies integer optimization opportunities."""
        df = pd.DataFrame({"small_int": pd.Series([1, 2, 3, 100], dtype="int64")})

        result = analyze(df)

        assert result.iloc[0]["current_dtype"] == "int64"
        assert result.iloc[0]["recommended_dtype"] == "uint8"
        assert result.iloc[0]["savings_mb"] > 0

    def test_analyze_shows_float_optimization(self):
        """Test that analyze identifies float optimization opportunities."""
        df = pd.DataFrame({"float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64")})

        result = analyze(df, aggressive=False)

        assert result.iloc[0]["current_dtype"] == "float64"
        assert result.iloc[0]["recommended_dtype"] == "float32"
        assert result.iloc[0]["savings_mb"] > 0

    def test_analyze_shows_category_optimization(self):
        """Test that analyze identifies category optimization opportunities."""
        df = pd.DataFrame({"cat_col": ["A", "B", "A", "B", "A"] * 10})

        result = analyze(df, categorical_threshold=0.5)

        assert result.iloc[0]["current_dtype"] == "object"
        assert result.iloc[0]["recommended_dtype"] == "category"
        assert result.iloc[0]["savings_mb"] > 0

    def test_analyze_with_boolean_columns(self):
        """Test that analyze identifies boolean optimization opportunities."""
        df = pd.DataFrame({"bool_col": pd.Series([0, 1, 1, 0, 1], dtype="int64")})

        result = analyze(df, optimize_bools=True)

        assert result.iloc[0]["current_dtype"] == "int64"
        assert result.iloc[0]["recommended_dtype"] == "boolean"
        assert result.iloc[0]["savings_mb"] > 0

    def test_analyze_sorted_by_savings(self):
        """Test that analyze results are sorted by savings."""
        df = pd.DataFrame(
            {
                "big_int": pd.Series(range(1000), dtype="int64"),
                "small_int": pd.Series([1, 2, 3], dtype="int64"),
            }
        )

        result = analyze(df)

        # First row should have the most savings
        assert result.iloc[0]["savings_mb"] >= result.iloc[1]["savings_mb"]

    def test_analyze_empty_dataframe(self):
        """Test analyze with empty DataFrame."""
        df = pd.DataFrame()

        result = analyze(df)

        assert len(result) == 0

    def test_analyze_with_nan_columns(self):
        """Test that analyze skips all-NaN columns."""
        df = pd.DataFrame({"good_col": [1, 2, 3], "nan_col": [np.nan, np.nan, np.nan]})

        result = analyze(df)

        # Should only have one row (the good column)
        assert len(result) == 1
        assert result.iloc[0]["column"] == "good_col"

    def test_analyze_aggressive_mode(self):
        """Test analyze with aggressive mode."""
        df = pd.DataFrame({"float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64")})

        result = analyze(df, aggressive=True)

        assert result.iloc[0]["recommended_dtype"] == "float16"
        assert "aggressive" in result.iloc[0]["reasoning"].lower()

    def test_analyze_high_cardinality_warning(self):
        """Test that analyze warns about high cardinality columns."""
        df = pd.DataFrame({"high_card": [f"id_{i}" for i in range(100)]})

        result = analyze(df, categorical_threshold=0.5)

        assert "High cardinality" in result.iloc[0]["reasoning"]
        assert result.iloc[0]["savings_mb"] == 0  # No savings expected


class TestGetOptimizationSummary:
    """Tests for get_optimization_summary function."""

    def test_summary_structure(self):
        """Test that summary has correct structure."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="int64"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
            }
        )

        summary = get_optimization_summary(df)

        assert "total_memory_mb" in summary
        assert "optimized_memory_mb" in summary
        assert "total_savings_mb" in summary
        assert "total_savings_percent" in summary
        assert "optimizable_columns" in summary
        assert "total_columns" in summary

    def test_summary_calculations(self):
        """Test that summary calculates correctly."""
        df = pd.DataFrame({"int_col": pd.Series([1, 2, 3, 100], dtype="int64")})

        summary = get_optimization_summary(df)

        assert summary["total_memory_mb"] > 0
        assert summary["optimized_memory_mb"] > 0
        assert summary["total_savings_mb"] > 0
        assert summary["total_savings_percent"] > 0
        assert summary["optimizable_columns"] == 1
        assert summary["total_columns"] == 1

    def test_summary_with_no_optimization(self):
        """Test summary when no optimization is possible."""
        # Create a DataFrame that's already optimized
        df = pd.DataFrame({"optimized": pd.Series([1, 2, 3], dtype="uint8")})

        summary = get_optimization_summary(df)

        # Should have minimal or no savings
        assert summary["total_savings_percent"] < 10


class TestEstimateMemoryReduction:
    """Tests for estimate_memory_reduction function."""

    def test_estimate_returns_float(self):
        """Test that estimate returns a percentage."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="int64"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
            }
        )

        reduction = estimate_memory_reduction(df)

        assert isinstance(reduction, (int, float))
        assert 0 <= reduction <= 100

    def test_estimate_with_optimizable_data(self):
        """Test estimate with data that can be optimized."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3, 100], dtype="int64"),
                "float_col": pd.Series([1.1, 2.2, 3.3, 4.4], dtype="float64"),
            }
        )

        reduction = estimate_memory_reduction(df)

        assert reduction > 10  # Should see memory reduction

    def test_estimate_with_already_optimized_data(self):
        """Test estimate with already optimized data."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="uint8"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float32"),
            }
        )

        reduction = estimate_memory_reduction(df)

        assert reduction < 10  # Minimal reduction expected


class TestAnalysisIntegration:
    """Integration tests for analysis with diet function."""

    def test_analyze_predicts_diet_behavior(self):
        """Test that analyze predictions match actual diet results."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3, 100], dtype="int64"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
            }
        )

        # Get analysis prediction
        analysis_summary = get_optimization_summary(df)
        predicted_reduction = analysis_summary["total_savings_percent"]

        # Actually optimize
        original_mem = df.memory_usage(deep=True).sum()
        optimized_df = dp.diet(df, verbose=False)
        optimized_mem = optimized_df.memory_usage(deep=True).sum()
        actual_reduction = (original_mem - optimized_mem) / original_mem * 100

        # Predictions should be close to actual (within 10%)
        assert abs(predicted_reduction - actual_reduction) < 10

    def test_analyze_before_optimize_workflow(self):
        """Test the analyze-then-optimize workflow."""
        df = pd.DataFrame(
            {
                "age": pd.Series(range(100), dtype="int64"),
                "score": pd.Series(np.random.randn(100), dtype="float64"),
                "category": ["A", "B", "C"] * 33 + ["A"],
            }
        )

        # Step 1: Analyze
        analysis = analyze(df)
        assert len(analysis) == 3

        # Step 2: Check if optimization is worth it
        summary = get_optimization_summary(df)
        if summary["total_savings_percent"] > 20:
            # Step 3: Optimize
            optimized_df = dp.diet(df, verbose=False)
            assert optimized_df.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
