"""
Unit tests for Diet Pandas IO functions.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp
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


class TestReadJSON:
    """Tests for read_json function."""

    def test_read_json_basic(self):
        """Test basic JSON reading with optimization."""
        # Create test data
        df = pd.DataFrame({"int_col": range(100), "str_col": ["A", "B", "C"] * 33 + ["A"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            df.to_json(f.name)
            temp_file = f.name

        try:
            result = dp.read_json(temp_file, verbose=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            os.unlink(temp_file)

    def test_read_json_no_optimization(self):
        """Test JSON reading without optimization."""
        df = pd.DataFrame({"col": range(10)})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            df.to_json(f.name)
            temp_file = f.name

        try:
            result = dp.read_json(temp_file, optimize=False)
            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(temp_file)


class TestReadFeather:
    """Tests for read_feather function."""

    def test_read_feather_basic(self):
        """Test basic Feather reading with optimization."""
        df = pd.DataFrame({"int_col": range(100), "float_col": np.random.randn(100)})

        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_file = f.name

        try:
            df.to_feather(temp_file)
            result = dp.read_feather(temp_file, verbose=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_feather_no_optimization(self):
        """Test Feather reading without optimization."""
        df = pd.DataFrame({"col": range(10)})

        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_file = f.name

        try:
            df.to_feather(temp_file)
            result = dp.read_feather(temp_file, optimize=False)
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestReadHDF:
    """Tests for read_hdf function."""

    def test_read_hdf_basic(self):
        """Test basic HDF5 reading with optimization."""
        df = pd.DataFrame({"int_col": range(100), "float_col": np.random.randn(100)})

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name

        try:
            df.to_hdf(temp_file, key="data", mode="w")
            result = dp.read_hdf(temp_file, key="data", verbose=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_hdf_no_optimization(self):
        """Test HDF5 reading without optimization."""
        df = pd.DataFrame({"col": range(10)})

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name

        try:
            df.to_hdf(temp_file, key="data", mode="w")
            result = dp.read_hdf(temp_file, key="data", optimize=False)
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestWriteOptimized:
    """Tests for optimized write functions."""

    def test_to_parquet_optimized(self):
        """Test optimized Parquet writing."""
        df = pd.DataFrame({"int_col": range(1000), "float_col": np.random.randn(1000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_file = f.name

        try:
            dp.to_parquet_optimized(df, temp_file, optimize_before_save=True)
            assert os.path.exists(temp_file)

            # Read back and verify
            result = pd.read_parquet(temp_file)
            assert len(result) == 1000
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_to_feather_optimized(self):
        """Test optimized Feather writing."""
        df = pd.DataFrame({"int_col": range(1000), "float_col": np.random.randn(1000)})

        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_file = f.name

        try:
            dp.to_feather_optimized(df, temp_file, optimize_before_save=True)
            assert os.path.exists(temp_file)

            # Read back and verify
            result = pd.read_feather(temp_file)
            assert len(result) == 1000
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_to_csv_optimized_no_optimization(self):
        """Test CSV writing without optimization."""
        df = pd.DataFrame({"col": range(10)})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_file = f.name

        try:
            dp.to_csv_optimized(df, temp_file, optimize_before_save=False)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
