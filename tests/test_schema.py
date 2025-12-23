"""
Tests for schema persistence functionality
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp


class TestSaveSchema:
    """Tests for save_schema function."""

    def test_save_schema_basic(self):
        """Test basic schema saving."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="uint8"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float32"),
                "cat_col": pd.Categorical(["A", "B", "C"]),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            dp.save_schema(df, schema_path)

            # Check file exists and contains correct data
            assert Path(schema_path).exists()

            with open(schema_path, "r") as f:
                schema = json.load(f)

            assert "int_col" in schema
            assert "uint8" in schema["int_col"]["dtype"]
            assert "float32" in schema["float_col"]["dtype"]
            assert "category" in schema["cat_col"]["dtype"]

        finally:
            Path(schema_path).unlink(missing_ok=True)

    def test_save_schema_with_nulls(self):
        """Test schema saving with nullable columns."""
        df = pd.DataFrame(
            {
                "with_nulls": [1, 2, np.nan],
                "no_nulls": [1, 2, 3],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            dp.save_schema(df, schema_path)

            with open(schema_path, "r") as f:
                schema = json.load(f)

            assert schema["with_nulls"]["nullable"] is True
            assert schema["no_nulls"]["nullable"] is False

        finally:
            Path(schema_path).unlink(missing_ok=True)

    def test_save_schema_overwrite_protection(self):
        """Test that overwrite protection works."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            # First save should succeed
            dp.save_schema(df, schema_path)

            # Second save without overwrite should fail
            with pytest.raises(FileExistsError):
                dp.save_schema(df, schema_path, overwrite=False)

            # Second save with overwrite should succeed
            dp.save_schema(df, schema_path, overwrite=True)

        finally:
            Path(schema_path).unlink(missing_ok=True)


class TestLoadSchema:
    """Tests for load_schema function."""

    def test_load_schema_basic(self):
        """Test basic schema loading."""
        df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="uint16"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float32"),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            dp.save_schema(df, schema_path)
            schema = dp.load_schema(schema_path)

            assert "int_col" in schema
            assert "uint16" in schema["int_col"]["dtype"]
            assert "float32" in schema["float_col"]["dtype"]

        finally:
            Path(schema_path).unlink(missing_ok=True)

    def test_load_schema_file_not_found(self):
        """Test loading non-existent schema file."""
        with pytest.raises(FileNotFoundError):
            dp.load_schema("nonexistent_schema.json")


class TestApplySchema:
    """Tests for apply_schema function."""

    def test_apply_schema_from_dict(self):
        """Test applying schema from dictionary."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],  # int64
                "float_col": [1.1, 2.2, 3.3],  # float64
            }
        )

        schema = {
            "int_col": {"dtype": "uint8"},
            "float_col": {"dtype": "float32"},
        }

        result = dp.apply_schema(df, schema)

        assert result["int_col"].dtype == np.uint8
        assert result["float_col"].dtype == np.float32

    def test_apply_schema_from_file(self):
        """Test applying schema from file."""
        optimized_df = pd.DataFrame(
            {
                "int_col": pd.Series([1, 2, 3], dtype="uint8"),
                "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float32"),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            # Save optimized schema
            dp.save_schema(optimized_df, schema_path)

            # Create unoptimized DataFrame
            df = pd.DataFrame(
                {
                    "int_col": [1, 2, 3],  # int64
                    "float_col": [1.1, 2.2, 3.3],  # float64
                }
            )

            # Apply schema
            result = dp.apply_schema(df, schema_path)

            assert result["int_col"].dtype == np.uint8
            assert result["float_col"].dtype == np.float32

        finally:
            Path(schema_path).unlink(missing_ok=True)

    def test_apply_schema_categorical(self):
        """Test applying schema with categorical columns."""
        df = pd.DataFrame({"cat_col": ["A", "B", "C", "A", "B"]})

        schema = {"cat_col": {"dtype": "category"}}

        result = dp.apply_schema(df, schema)

        assert result["cat_col"].dtype.name == "category"

    def test_apply_schema_boolean(self):
        """Test applying schema with boolean columns."""
        df = pd.DataFrame({"bool_col": [0, 1, 1, 0, 1]})

        schema = {"bool_col": {"dtype": "boolean"}}

        result = dp.apply_schema(df, schema)

        assert result["bool_col"].dtype.name == "boolean"

    def test_apply_schema_missing_columns_non_strict(self):
        """Test applying schema with missing columns (non-strict mode)."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        schema = {
            "col1": {"dtype": "uint8"},
            "col3": {"dtype": "uint8"},  # Not in DataFrame
        }

        # Should not raise error in non-strict mode
        result = dp.apply_schema(df, schema, strict=False)

        assert result["col1"].dtype == np.uint8
        assert "col2" in result.columns  # Untouched column

    def test_apply_schema_missing_columns_strict(self):
        """Test applying schema with missing columns (strict mode)."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        schema = {
            "col1": {"dtype": "uint8"},
            "col2": {"dtype": "uint8"},  # Not in DataFrame
        }

        with pytest.raises(ValueError, match="Schema contains columns not in"):
            dp.apply_schema(df, schema, strict=True)

    def test_apply_schema_extra_columns_strict(self):
        """Test applying schema when DataFrame has extra columns (strict)."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        schema = {"col1": {"dtype": "uint8"}}  # Missing col2

        with pytest.raises(ValueError, match="DataFrame contains columns not in"):
            dp.apply_schema(df, schema, strict=True)


class TestAutoSchemaPath:
    """Tests for auto_schema_path function."""

    def test_auto_schema_path_csv(self):
        """Test auto schema path generation for CSV."""
        result = dp.auto_schema_path("data.csv")
        assert result == Path("data.diet_schema.json")

    def test_auto_schema_path_with_directory(self):
        """Test auto schema path with directory."""
        result = dp.auto_schema_path("folder/data.csv")
        assert result == Path("folder/data.diet_schema.json")

    def test_auto_schema_path_parquet(self):
        """Test auto schema path for parquet."""
        result = dp.auto_schema_path("data.parquet")
        assert result == Path("data.diet_schema.json")


class TestSchemaWorkflow:
    """Integration tests for complete schema workflow."""

    def test_save_load_apply_workflow(self):
        """Test complete save-load-apply workflow."""
        # Create and optimize DataFrame
        original_df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 100],
                "float_col": [1.1, 2.2, 3.3, 4.4],
                "cat_col": ["A", "B", "A", "B"],
            }
        )

        optimized_df = dp.diet(original_df, verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            # Save schema
            dp.save_schema(optimized_df, schema_path)

            # Later: load new data with same structure
            new_df = pd.DataFrame(
                {
                    "int_col": [5, 6, 7, 150],
                    "float_col": [5.5, 6.6, 7.7, 8.8],
                    "cat_col": ["C", "D", "C", "D"],
                }
            )

            # Apply saved schema
            result = dp.apply_schema(new_df, schema_path)

            # Should have same dtypes as optimized_df
            assert result["int_col"].dtype == optimized_df["int_col"].dtype
            assert result["float_col"].dtype == optimized_df["float_col"].dtype
            # Category column should match (if it was categorical in optimized)
            if optimized_df["cat_col"].dtype.name == "category":
                assert result["cat_col"].dtype.name == "category"

        finally:
            Path(schema_path).unlink(missing_ok=True)
