"""
Unit tests for Diet Pandas core optimization functions.
"""

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp
from dietpandas.core import (
    diet,
    get_memory_report,
    optimize_datetime,
    optimize_float,
    optimize_int,
    optimize_obj,
    optimize_sparse,
)


class TestOptimizeInt:
    """Tests for integer optimization."""

    def test_optimize_small_unsigned_integers(self):
        """Test that small positive integers are converted to uint8."""
        s = pd.Series([0, 1, 2, 100, 255], dtype="int64")
        result = optimize_int(s)
        assert result.dtype == np.uint8
        assert list(result) == [0, 1, 2, 100, 255]

    def test_optimize_medium_unsigned_integers(self):
        """Test that medium positive integers are converted to uint16."""
        s = pd.Series([0, 1000, 30000], dtype="int64")
        result = optimize_int(s)
        assert result.dtype == np.uint16
        assert list(result) == [0, 1000, 30000]

    def test_optimize_small_signed_integers(self):
        """Test that small signed integers are converted to int8."""
        s = pd.Series([-100, 0, 100], dtype="int64")
        result = optimize_int(s)
        assert result.dtype == np.int8
        assert list(result) == [-100, 0, 100]

    def test_optimize_medium_signed_integers(self):
        """Test that medium signed integers are converted to int16."""
        s = pd.Series([-1000, 0, 1000], dtype="int64")
        result = optimize_int(s)
        assert result.dtype == np.int16
        assert list(result) == [-1000, 0, 1000]

    def test_no_downcast_when_not_beneficial(self):
        """Test that very large integers stay as int64."""
        s = pd.Series([0, 2**40], dtype="int64")
        result = optimize_int(s)
        assert result.dtype == np.uint64


class TestOptimizeFloat:
    """Tests for float optimization."""

    def test_optimize_float_to_float32(self):
        """Test that float64 is converted to float32 in safe mode."""
        s = pd.Series([1.1, 2.2, 3.3], dtype="float64")
        result = optimize_float(s, aggressive=False)
        assert result.dtype == np.float32

    def test_optimize_float_to_float16_aggressive(self):
        """Test that float64 is converted to float16 in aggressive mode."""
        s = pd.Series([1.1, 2.2, 3.3], dtype="float64")
        result = optimize_float(s, aggressive=True)
        assert result.dtype == np.float16


class TestOptimizeObj:
    """Tests for object/string optimization."""

    def test_optimize_low_cardinality_to_category(self):
        """Test that low cardinality strings are converted to category."""
        s = pd.Series(["A", "B", "A", "B", "A", "B"] * 10)
        result = optimize_obj(s, categorical_threshold=0.5)
        assert result.dtype.name == "category"
        assert list(result.unique()) == ["A", "B"]

    def test_no_optimization_high_cardinality(self):
        """Test that high cardinality strings remain as object."""
        s = pd.Series([f"ID_{i}" for i in range(100)])
        result = optimize_obj(s, categorical_threshold=0.5)
        assert result.dtype == "object"

    def test_empty_series(self):
        """Test that empty series are handled correctly."""
        s = pd.Series([], dtype="object")
        result = optimize_obj(s)
        assert result.dtype == "object"


class TestDiet:
    """Tests for the main diet function."""

    def test_diet_reduces_memory(self):
        """Test that diet function reduces memory usage."""
        df = pd.DataFrame(
            {
                "small_int": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "category_col": ["A", "B", "A", "B", "A"],
            }
        )

        # Convert to wasteful types
        df["small_int"] = df["small_int"].astype("int64")
        df["float_col"] = df["float_col"].astype("float64")

        start_mem = df.memory_usage(deep=True).sum()
        result = diet(df, verbose=False)
        end_mem = result.memory_usage(deep=True).sum()

        assert end_mem < start_mem

    def test_diet_preserves_data(self):
        """Test that diet function preserves data values."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.1, 2.2, 3.3],
                "c": ["x", "y", "z"],
            }
        )

        result = diet(df, verbose=False)

        # Check that values are preserved
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == pytest.approx([1.1, 2.2, 3.3], rel=1e-5)
        assert list(result["c"]) == ["x", "y", "z"]

    def test_diet_inplace(self):
        """Test that diet function can modify DataFrame in place."""
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="int64"),
            }
        )

        diet(df, verbose=False, inplace=True)

        # When inplace=True, should modify the original DataFrame
        assert df["a"].dtype == np.uint8

    def test_diet_copy(self):
        """Test that diet function returns a copy by default."""
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="int64"),
            }
        )

        original_dtype = df["a"].dtype
        result = diet(df, verbose=False, inplace=False)

        # Original should be unchanged
        assert df["a"].dtype == original_dtype
        # Result should be optimized
        assert result["a"].dtype == np.uint8

    def test_diet_with_nan_values(self):
        """Test that diet handles NaN values correctly."""
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4],
                "b": [1.1, np.nan, 3.3, 4.4],
            }
        )

        result = diet(df, verbose=False)
        assert result.isna().sum().sum() == 2  # Should preserve NaN values

    def test_diet_aggressive_mode(self):
        """Test that aggressive mode uses float16."""
        df = pd.DataFrame(
            {
                "float_col": [1.1, 2.2, 3.3],
            }
        )

        result = diet(df, verbose=False, aggressive=True)
        assert result["float_col"].dtype == np.float16


class TestGetMemoryReport:
    """Tests for memory reporting function."""

    def test_memory_report_structure(self):
        """Test that memory report has correct structure."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )

        report = get_memory_report(df)

        assert "column" in report.columns
        assert "dtype" in report.columns
        assert "memory_bytes" in report.columns
        assert "memory_mb" in report.columns
        assert "percent_of_total" in report.columns

    def test_memory_report_sorted(self):
        """Test that memory report is sorted by memory usage."""
        df = pd.DataFrame(
            {
                "small": [1, 2],
                "large": ["x" * 1000, "y" * 1000],
            }
        )

        report = get_memory_report(df)

        # First row should be the largest memory consumer
        assert report.iloc[0]["memory_bytes"] >= report.iloc[1]["memory_bytes"]


class TestOptimizeDatetime:
    """Tests for datetime optimization."""

    def test_datetime_already_optimized(self):
        """Test that datetime64[ns] columns remain efficient."""
        dates = pd.Series(pd.date_range("2020-01-01", periods=100))
        result = optimize_datetime(dates)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_datetime_from_object(self):
        """Test conversion of object strings to datetime."""
        dates = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        result = optimize_datetime(dates)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_datetime_with_timezone(self):
        """Test that timezone-aware datetimes are preserved."""
        dates = pd.Series(pd.date_range("2020-01-01", periods=10, tz="UTC"))
        result = optimize_datetime(dates)
        assert result.dt.tz is not None


class TestOptimizeSparse:
    """Tests for sparse optimization."""

    def test_highly_sparse_binary(self):
        """Test that highly sparse binary data is converted to sparse format."""
        s = pd.Series([0] * 95 + [1] * 5)
        result = optimize_sparse(s, sparse_threshold=0.9)
        assert isinstance(result.dtype, pd.SparseDtype)

    def test_not_sparse_enough(self):
        """Test that less sparse data remains dense."""
        s = pd.Series([0] * 60 + [1] * 40)
        result = optimize_sparse(s, sparse_threshold=0.9)
        assert not isinstance(result.dtype, pd.SparseDtype)

    def test_empty_series(self):
        """Test that empty series are handled correctly."""
        s = pd.Series([], dtype="int64")
        result = optimize_sparse(s)
        assert len(result) == 0

    def test_already_sparse(self):
        """Test that already sparse series remain sparse."""
        s = pd.Series([0] * 95 + [1] * 5, dtype=pd.SparseDtype(np.int64, 0))
        result = optimize_sparse(s)
        assert isinstance(result.dtype, pd.SparseDtype)

    def test_sparse_with_floats(self):
        """Test sparse optimization with float data."""
        s = pd.Series([0.0] * 92 + [1.5, 2.5, 3.5])
        result = optimize_sparse(s, sparse_threshold=0.9)
        assert isinstance(result.dtype, pd.SparseDtype)


class TestDietWithNewFeatures:
    """Tests for diet function with new features."""

    def test_diet_with_datetime_optimization(self):
        """Test diet with datetime optimization enabled."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "value": range(100)})
        result = dp.diet(df, optimize_datetimes=True, verbose=False)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_diet_with_sparse_optimization(self):
        """Test diet with sparse optimization enabled."""
        df = pd.DataFrame({"sparse_col": [0] * 95 + [1] * 5, "dense_col": range(100)})
        result = dp.diet(df, optimize_sparse_cols=True, verbose=False)
        assert isinstance(result["sparse_col"].dtype, pd.SparseDtype)
        assert not isinstance(result["dense_col"].dtype, pd.SparseDtype)

    def test_diet_with_all_optimizations(self):
        """Test diet with all optimization features enabled."""
        df = pd.DataFrame(
            {
                "int_col": range(100),
                "float_col": np.random.randn(100),
                "sparse_col": [0] * 95 + [1] * 5,
                "cat_col": ["A", "B", "C"] * 33 + ["A"],
                "date_col": pd.date_range("2020-01-01", periods=100),
            }
        )

        original_memory = df.memory_usage(deep=True).sum()
        result = dp.diet(df, optimize_datetimes=True, optimize_sparse_cols=True, verbose=False)
        optimized_memory = result.memory_usage(deep=True).sum()

        # Should reduce memory
        assert optimized_memory < original_memory

    def test_diet_inplace_modification(self):
        """Test that inplace=True modifies the original DataFrame."""
        df = pd.DataFrame({"int_col": range(1000), "float_col": np.random.randn(1000)})

        original_id = id(df)
        result = dp.diet(df, inplace=True, verbose=False)

        # Should return the same object
        assert id(result) == original_id
        assert result["int_col"].dtype != np.int64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
