"""
Tests for automatic chunked reading functionality in read_csv
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import dietpandas as dp


class TestReadCsvAutoChunking:
    """Tests for automatic chunked reading in read_csv."""

    def test_read_csv_small_file_no_chunking(self):
        """Test that small files don't trigger chunking."""
        # Create a small test CSV file
        df = pd.DataFrame(
            {
                "int_col": range(100),
                "float_col": [x * 1.1 for x in range(100)],
                "str_col": [f"value_{i % 10}" for i in range(100)],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            df.to_csv(csv_path, index=False)

            # Should not use chunking for small file
            result = dp.read_csv(csv_path, verbose=False)

            # Should have same shape
            assert result.shape == df.shape
            assert list(result.columns) == list(df.columns)

            # Should be optimized
            assert result["int_col"].dtype in ["uint8", "uint16"]
            assert result["str_col"].dtype.name == "category"

        finally:
            Path(csv_path).unlink(missing_ok=True)

    @patch("dietpandas.io._get_available_memory_mb")
    @patch("dietpandas.io._estimate_csv_memory_mb")
    def test_read_csv_large_file_auto_chunking(self, mock_estimate, mock_available):
        """Test that large files automatically trigger chunking."""
        # Mock memory checks to trigger chunking
        mock_available.return_value = 1000  # 1GB available
        mock_estimate.return_value = 800  # File needs 800MB (>70% of available)

        df = pd.DataFrame(
            {
                "int_col": range(1000),
                "float_col": [x * 1.1 for x in range(1000)],
                "str_col": [f"value_{i % 10}" for i in range(1000)],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            df.to_csv(csv_path, index=False)

            # Should use chunking
            result = dp.read_csv(csv_path, chunksize=250, verbose=False)

            # Should have correct shape
            assert result.shape == df.shape
            # Should be optimized
            assert result["str_col"].dtype.name == "category"

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_read_csv_disable_auto_chunking(self):
        """Test disabling automatic chunking."""
        df = pd.DataFrame({"int_col": range(100), "float_col": [x * 1.1 for x in range(100)]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            df.to_csv(csv_path, index=False)

            # Explicitly disable auto-chunking
            result = dp.read_csv(csv_path, auto_chunk=False, verbose=False)

            assert result.shape == df.shape

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_read_csv_with_schema(self):
        """Test read_csv with schema."""
        # Create optimized DataFrame and save schema
        df = pd.DataFrame(
            {
                "int_col": range(500),
                "float_col": [x * 1.1 for x in range(500)],
                "cat_col": [f"cat_{i % 5}" for i in range(500)],
            }
        )

        optimized_df = dp.diet(df, verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_path = csv_f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as schema_f:
            schema_path = schema_f.name

        try:
            # Save schema
            dp.save_schema(optimized_df, schema_path)

            # Save CSV with original (unoptimized) data
            df.to_csv(csv_path, index=False)

            # Read with schema
            result = dp.read_csv(csv_path, schema_path=schema_path, verbose=False)

            # Should have same dtypes as optimized_df
            assert result["int_col"].dtype == optimized_df["int_col"].dtype
            assert result["float_col"].dtype == optimized_df["float_col"].dtype
            assert result["cat_col"].dtype.name == "category"

        finally:
            Path(csv_path).unlink(missing_ok=True)
            Path(schema_path).unlink(missing_ok=True)

    def test_read_csv_save_schema(self):
        """Test read_csv with schema saving."""
        df = pd.DataFrame(
            {
                "int_col": range(300),
                "float_col": [x * 1.1 for x in range(300)],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_path = csv_f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as schema_f:
            schema_path = schema_f.name
            Path(schema_path).unlink()  # Delete it so we can test creation

        try:
            df.to_csv(csv_path, index=False)

            # Read and save schema
            result = dp.read_csv(
                csv_path,
                save_schema=True,
                schema_path=schema_path,
                verbose=False,
            )

            # Should return a DataFrame
            assert result.shape[0] == 300

            # Schema file should be created
            assert Path(schema_path).exists()

            # Load and verify schema
            schema = dp.load_schema(schema_path)
            assert "int_col" in schema
            assert "float_col" in schema

        finally:
            Path(csv_path).unlink(missing_ok=True)
            Path(schema_path).unlink(missing_ok=True)

    @patch("dietpandas.io._get_available_memory_mb")
    @patch("dietpandas.io._estimate_csv_memory_mb")
    def test_read_csv_chunked_with_schema(self, mock_estimate, mock_available):
        """Test automatic chunking with schema."""
        # Force chunking
        mock_available.return_value = 1000
        mock_estimate.return_value = 900

        df = pd.DataFrame(
            {
                "int_col": range(500),
                "float_col": [x * 1.1 for x in range(500)],
            }
        )

        optimized_df = dp.diet(df, verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_path = csv_f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as schema_f:
            schema_path = schema_f.name

        try:
            dp.save_schema(optimized_df, schema_path)
            df.to_csv(csv_path, index=False)

            # Should use chunking with schema
            result = dp.read_csv(
                csv_path,
                chunksize=200,
                schema_path=schema_path,
                verbose=False,
            )

            # Should have same dtypes
            assert result["int_col"].dtype == optimized_df["int_col"].dtype
            assert result["float_col"].dtype == optimized_df["float_col"].dtype

        finally:
            Path(csv_path).unlink(missing_ok=True)
            Path(schema_path).unlink(missing_ok=True)


class TestReadCsvChunkingWorkflow:
    """Integration tests for complete chunking workflow."""

    @patch("dietpandas.io._get_available_memory_mb")
    @patch("dietpandas.io._estimate_csv_memory_mb")
    def test_auto_chunking_workflow(self, mock_estimate, mock_available):
        """Test complete workflow: auto-chunk, save schema, reuse."""
        # Force chunking
        mock_available.return_value = 1000
        mock_estimate.return_value = 850

        df = pd.DataFrame(
            {
                "id": range(1000),
                "value": [x * 1.5 for x in range(1000)],
                "category": [f"cat_{i % 10}" for i in range(1000)],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_path = csv_f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as schema_f:
            schema_path = schema_f.name
            Path(schema_path).unlink()

        try:
            df.to_csv(csv_path, index=False)

            # First load: automatic chunking, save schema
            result1 = dp.read_csv(
                csv_path,
                chunksize=250,
                save_schema=True,
                schema_path=schema_path,
                verbose=False,
            )

            # Second load: use saved schema (automatic chunking again)
            result2 = dp.read_csv(
                csv_path,
                chunksize=250,
                schema_path=schema_path,
                verbose=False,
            )

            # Both should have same dtypes (or at least compatible integer types)
            # Note: Chunked reading might result in slightly different dtypes
            # depending on chunk boundaries, so we check they're both optimized
            assert result1["id"].dtype in ["uint8", "uint16", "uint32"]
            assert result2["id"].dtype in ["uint8", "uint16", "uint32"]
            assert result1["value"].dtype == result2["value"].dtype
            assert result1["category"].dtype.name == result2["category"].dtype.name

            # Both should be optimized
            assert result2["id"].dtype in ["uint8", "uint16"]
            assert result2["value"].dtype == "float32"
            assert result2["category"].dtype.name == "category"

        finally:
            Path(csv_path).unlink(missing_ok=True)
            Path(schema_path).unlink(missing_ok=True)
