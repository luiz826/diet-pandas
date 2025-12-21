"""
Additional unit tests for new Diet Pandas features.
"""

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp
from dietpandas.core import optimize_datetime, optimize_sparse


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
