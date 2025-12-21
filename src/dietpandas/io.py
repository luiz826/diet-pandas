"""
Diet Pandas - High-Speed IO Module

This module provides optimized file reading functions using Polars as the parsing engine,
then converting to optimized Pandas DataFrames.
"""

from pathlib import Path
from typing import Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from .core import diet


def read_csv(
    filepath: Union[str, Path],
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    use_polars: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a CSV file using Polars engine (if available), then converts to optimized Pandas.

    This function is often 5-10x faster at parsing CSVs than pandas.read_csv, and the
    resulting DataFrame uses significantly less memory due to automatic optimization.

    Args:
        filepath: Path to CSV file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        use_polars: If True and Polars is available, use it for parsing (default: True)
        **kwargs: Additional arguments passed to the CSV reader

    Returns:
        Optimized pandas DataFrame

    Examples:
        >>> df = read_csv("large_dataset.csv")
        Diet Complete: Memory reduced by 67.3%

        >>> # Disable optimization if needed
        >>> df = read_csv("data.csv", optimize=False)

        >>> # Use aggressive mode for maximum compression
        >>> df = read_csv("data.csv", aggressive=True)
    """
    filepath = str(filepath)

    # Try to use Polars for fast parsing
    if use_polars and POLARS_AVAILABLE:
        try:
            # Step 1: Fast Read with Polars
            # Polars is multi-threaded and much faster at parsing CSVs
            pl_df = pl.read_csv(filepath, **kwargs)

            # Step 2: Convert to Pandas
            pd_df = pl_df.to_pandas()

            if verbose:
                print("Loaded with Polars engine (fast mode)")

        except Exception as e:
            if verbose:
                print(f"Polars parsing failed ({e}), falling back to Pandas")
            # Fallback to standard Pandas
            pd_df = pd.read_csv(filepath, **kwargs)
    else:
        # Use standard Pandas
        if verbose and use_polars and not POLARS_AVAILABLE:
            print("Polars not installed, using standard Pandas reader")
        pd_df = pd.read_csv(filepath, **kwargs)

    # Step 3: Apply the Diet immediately
    if optimize:
        return diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

    return pd_df


def read_parquet(
    filepath: Union[str, Path],
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    use_polars: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a Parquet file using Polars engine (if available), then converts to optimized Pandas.

    Args:
        filepath: Path to Parquet file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        use_polars: If True and Polars is available, use it for parsing (default: True)
        **kwargs: Additional arguments passed to the Parquet reader

    Returns:
        Optimized pandas DataFrame
    """
    filepath = str(filepath)

    # Try to use Polars for fast parsing
    if use_polars and POLARS_AVAILABLE:
        try:
            pl_df = pl.read_parquet(filepath, **kwargs)
            pd_df = pl_df.to_pandas()

            if verbose:
                print("Loaded with Polars engine (fast mode)")

        except Exception as e:
            if verbose:
                print(f"Polars parsing failed ({e}), falling back to Pandas")
            pd_df = pd.read_parquet(filepath, **kwargs)
    else:
        if verbose and use_polars and not POLARS_AVAILABLE:
            print("Polars not installed, using standard Pandas reader")
        pd_df = pd.read_parquet(filepath, **kwargs)

    if optimize:
        return diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

    return pd_df


def read_excel(
    filepath: Union[str, Path],
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads an Excel file and returns an optimized Pandas DataFrame.

    Note: Polars support for Excel is limited, so this uses pandas.read_excel.

    Args:
        filepath: Path to Excel file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        **kwargs: Additional arguments passed to pandas.read_excel

    Returns:
        Optimized pandas DataFrame
    """
    filepath = str(filepath)
    pd_df = pd.read_excel(filepath, **kwargs)

    if optimize:
        return diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

    return pd_df


def to_csv_optimized(
    df: pd.DataFrame, filepath: Union[str, Path], optimize_before_save: bool = True, **kwargs
) -> None:
    """
    Saves a DataFrame to CSV, optionally optimizing it first.

    Args:
        df: DataFrame to save
        filepath: Path where CSV will be saved
        optimize_before_save: If True, optimize the DataFrame before saving
        **kwargs: Additional arguments passed to pandas.to_csv
    """
    if optimize_before_save:
        df = diet(df, verbose=False)

    df.to_csv(filepath, **kwargs)


def read_json(
    filepath: Union[str, Path],
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a JSON file and returns an optimized Pandas DataFrame.

    Args:
        filepath: Path to JSON file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        **kwargs: Additional arguments passed to pandas.read_json

    Returns:
        Optimized pandas DataFrame

    Examples:
        >>> df = read_json("data.json")
        🥗 Diet Complete: Memory reduced by 45.2%
    """
    filepath = str(filepath)
    pd_df = pd.read_json(filepath, **kwargs)

    if optimize:
        return diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

    return pd_df


def read_hdf(
    filepath: Union[str, Path],
    key: str,
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads an HDF5 file and returns an optimized Pandas DataFrame.

    Args:
        filepath: Path to HDF5 file
        key: Group identifier in the HDF5 file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        **kwargs: Additional arguments passed to pandas.read_hdf

    Returns:
        Optimized pandas DataFrame

    Examples:
        >>> df = read_hdf("data.h5", key="dataset1")
        🥗 Diet Complete: Memory reduced by 52.1%
    """
    filepath = str(filepath)
    pd_df = pd.read_hdf(filepath, key=key, **kwargs)

    if optimize:
        return diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

    return pd_df


def read_feather(
    filepath: Union[str, Path],
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a Feather file and returns an optimized Pandas DataFrame.

    Feather is a fast, lightweight columnar data format.

    Args:
        filepath: Path to Feather file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        **kwargs: Additional arguments passed to pandas.read_feather

    Returns:
        Optimized pandas DataFrame

    Examples:
        >>> df = read_feather("data.feather")
        🥗 Diet Complete: Memory reduced by 38.7%
    """
    filepath = str(filepath)
    pd_df = pd.read_feather(filepath, **kwargs)

    if optimize:
        return diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

    return pd_df


def to_parquet_optimized(
    df: pd.DataFrame, filepath: Union[str, Path], optimize_before_save: bool = True, **kwargs
) -> None:
    """
    Saves a DataFrame to Parquet format, optionally optimizing it first.

    Args:
        df: DataFrame to save
        filepath: Path where Parquet file will be saved
        optimize_before_save: If True, optimize the DataFrame before saving
        **kwargs: Additional arguments passed to pandas.to_parquet
    """
    if optimize_before_save:
        df = diet(df, verbose=False)

    df.to_parquet(filepath, **kwargs)


def to_feather_optimized(
    df: pd.DataFrame, filepath: Union[str, Path], optimize_before_save: bool = True, **kwargs
) -> None:
    """
    Saves a DataFrame to Feather format, optionally optimizing it first.

    Args:
        df: DataFrame to save
        filepath: Path where Feather file will be saved
        optimize_before_save: If True, optimize the DataFrame before saving
        **kwargs: Additional arguments passed to pandas.to_feather
    """
    if optimize_before_save:
        df = diet(df, verbose=False)

    df.to_feather(filepath, **kwargs)
