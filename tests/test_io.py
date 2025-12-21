"""
Unit tests for Diet Pandas IO functions.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from dietpandas.io import read_csv, read_excel, read_parquet, to_csv_optimized

# Check if polars is available
try:
    import polars  # noqa: F401

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class TestReadCSV:
    """Tests for CSV reading with optimization."""

    def test_read_csv_basic(self):
        """Test basic CSV reading."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n")
            f.write("1,2.5,foo\n")
            f.write("2,3.5,foo\n")
            f.write("3,4.5,foo\n")
            temp_path = f.name

        try:
            df = read_csv(temp_path, verbose=False)

            # Check that data was loaded correctly
            assert len(df) == 3
            assert list(df.columns) == ["a", "b", "c"]

            # Check that optimization was applied
            assert df["a"].dtype in [np.uint8, np.int8, np.uint16, np.int16]
            assert df["b"].dtype == np.float32
            # With only 1 unique value out of 3, should be category
            assert df["c"].dtype.name == "category"

        finally:
            os.unlink(temp_path)

    def test_read_csv_no_optimization(self):
        """Test CSV reading without optimization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n")
            f.write("1,2.5\n")
            f.write("2,3.5\n")
            temp_path = f.name

        try:
            df = read_csv(temp_path, optimize=False, verbose=False)

            # Should not be optimized (depends on reader defaults, but likely int64/float64)
            # We just check it loads correctly
            assert len(df) == 2

        finally:
            os.unlink(temp_path)

    def test_read_csv_aggressive(self):
        """Test CSV reading with aggressive optimization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n")
            f.write("1,2.5\n")
            f.write("2,3.5\n")
            temp_path = f.name

        try:
            df = read_csv(temp_path, aggressive=True, verbose=False)

            # Check aggressive optimization (float16)
            assert df["b"].dtype == np.float16

        finally:
            os.unlink(temp_path)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_read_csv_with_polars(self):
        """Test CSV reading using Polars engine."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n")
            f.write("1,2.5\n")
            f.write("2,3.5\n")
            temp_path = f.name

        try:
            df = read_csv(temp_path, use_polars=True, verbose=False)

            assert len(df) == 2
            assert isinstance(df, pd.DataFrame)

        finally:
            os.unlink(temp_path)


class TestReadParquet:
    """Tests for Parquet reading with optimization."""

    def test_read_parquet_basic(self):
        """Test basic Parquet reading."""
        # Create a temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # Create and save a DataFrame
            df_original = pd.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": [2.5, 3.5, 4.5],
                }
            )
            df_original.to_parquet(temp_path)

            # Read with optimization
            df = read_parquet(temp_path, verbose=False)

            assert len(df) == 3
            assert df["a"].dtype in [np.uint8, np.int8]
            assert df["b"].dtype == np.float32

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestReadExcel:
    """Tests for Excel reading with optimization."""

    def test_read_excel_basic(self):
        """Test basic Excel reading."""
        # Create a temporary Excel file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            # Create and save a DataFrame
            df_original = pd.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": [2.5, 3.5, 4.5],
                }
            )
            df_original.to_excel(temp_path, index=False)

            # Read with optimization
            df = read_excel(temp_path, verbose=False)

            assert len(df) == 3
            assert df["a"].dtype in [np.uint8, np.int8]
            assert df["b"].dtype == np.float32

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestToCSVOptimized:
    """Tests for optimized CSV writing."""

    def test_to_csv_optimized(self):
        """Test writing CSV with optimization."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            df = pd.DataFrame(
                {
                    "a": pd.Series([1, 2, 3], dtype="int64"),
                    "b": pd.Series([2.5, 3.5, 4.5], dtype="float64"),
                }
            )

            to_csv_optimized(df, temp_path, index=False)

            # Read back and check
            df_read = pd.read_csv(temp_path)
            assert len(df_read) == 3

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
