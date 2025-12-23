"""
Diet Pandas - Schema Persistence Module

This module provides functions for saving and loading DataFrame schemas
to avoid re-analyzing data types on repeated loads.
"""

import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd


def save_schema(df: pd.DataFrame, filepath: Union[str, Path], overwrite: bool = True) -> None:
    """
    Save DataFrame schema to a JSON file for later reuse.

    The schema includes data types for all columns, allowing you to skip
    the optimization analysis phase on subsequent loads of the same dataset.

    Args:
        df: DataFrame whose schema to save
        filepath: Path where schema will be saved (.diet_schema.json)
        overwrite: If True, overwrite existing schema file

    Examples:
        >>> df = dp.diet(df)
        >>> dp.save_schema(df, "data.diet_schema.json")

        >>> # Later, load with the saved schema
        >>> df = pd.read_csv("data.csv")
        >>> df = dp.apply_schema(df, "data.diet_schema.json")
    """
    filepath = Path(filepath)

    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"Schema file {filepath} already exists. " f"Use overwrite=True to replace it."
        )

    # Extract schema information
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        schema[col] = {
            "dtype": str(dtype),
            "nullable": bool(df[col].isna().any()),  # Convert to Python bool
        }

        # Store additional info for sparse columns
        if isinstance(dtype, pd.SparseDtype):
            schema[col]["sparse"] = True
            schema[col]["fill_value"] = str(dtype.fill_value)

    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema(filepath: Union[str, Path]) -> Dict:
    """
    Load DataFrame schema from a JSON file.

    Args:
        filepath: Path to schema file (.diet_schema.json)

    Returns:
        Dictionary mapping column names to dtype specifications

    Examples:
        >>> schema = dp.load_schema("data.diet_schema.json")
        >>> print(schema)
        {'id': {'dtype': 'uint16', 'nullable': False}, ...}
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Schema file not found: {filepath}")

    with open(filepath, "r") as f:
        schema = json.load(f)

    return schema


def apply_schema(
    df: pd.DataFrame, schema: Union[Dict, str, Path], strict: bool = False
) -> pd.DataFrame:
    """
    Apply a saved schema to a DataFrame.

    This function converts DataFrame columns to match the types specified
    in the schema, avoiding the need to re-analyze the data.

    Args:
        df: DataFrame to apply schema to
        schema: Either a schema dict or path to schema JSON file
        strict: If True, raise error if schema columns don't match DataFrame

    Returns:
        DataFrame with applied schema

    Raises:
        ValueError: If strict=True and columns don't match

    Examples:
        >>> # Apply from file
        >>> df = pd.read_csv("data.csv")
        >>> df = dp.apply_schema(df, "data.diet_schema.json")

        >>> # Apply from dict
        >>> schema = {'id': {'dtype': 'uint16'}, 'name': {'dtype': 'category'}}
        >>> df = dp.apply_schema(df, schema)
    """
    # Load schema if filepath provided
    if isinstance(schema, (str, Path)):
        schema = load_schema(schema)

    df = df.copy()

    # Check for missing columns
    schema_cols = set(schema.keys())
    df_cols = set(df.columns)

    missing_in_df = schema_cols - df_cols
    missing_in_schema = df_cols - schema_cols

    if strict:
        if missing_in_df:
            raise ValueError(f"Schema contains columns not in DataFrame: {missing_in_df}")
        if missing_in_schema:
            raise ValueError(f"DataFrame contains columns not in schema: {missing_in_schema}")

    # Apply dtype conversions
    for col, col_schema in schema.items():
        if col not in df.columns:
            continue

        dtype_str = col_schema["dtype"]

        try:
            # Handle sparse dtypes
            if col_schema.get("sparse", False):
                fill_value = col_schema.get("fill_value", "nan")
                if fill_value != "nan":
                    fill_value = float(fill_value)
                df[col] = pd.arrays.SparseArray(df[col], fill_value=fill_value)
            # Handle categorical
            elif dtype_str == "category":
                df[col] = df[col].astype("category")
            # Handle boolean
            elif dtype_str == "boolean" or dtype_str == "bool":
                df[col] = df[col].astype("boolean")
            # Handle other dtypes
            else:
                df[col] = df[col].astype(dtype_str)
        except Exception as e:
            # Skip columns that can't be converted
            if strict:
                raise ValueError(f"Failed to convert column '{col}' to {dtype_str}: {e}")

    return df


def auto_schema_path(data_path: Union[str, Path]) -> Path:
    """
    Generate automatic schema filepath for a data file.

    Args:
        data_path: Path to data file

    Returns:
        Path for corresponding schema file

    Examples:
        >>> dp.auto_schema_path("data.csv")
        PosixPath('data.diet_schema.json')

        >>> dp.auto_schema_path("folder/data.parquet")
        PosixPath('folder/data.diet_schema.json')
    """
    data_path = Path(data_path)
    return data_path.parent / f"{data_path.stem}.diet_schema.json"
