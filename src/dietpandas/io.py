"""
Diet Pandas - High-Speed IO Module

This module provides optimized file reading functions using Polars as the parsing engine,
then converting to optimized Pandas DataFrames.
"""

import os
from pathlib import Path
from typing import Union

import pandas as pd
import psutil

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from .core import diet


def _get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / (1024**2)


def _estimate_csv_memory_mb(filepath: Union[str, Path]) -> float:
    """Estimate memory needed to load CSV (rough estimate: 2x file size)."""
    file_size_bytes = os.path.getsize(filepath)
    return (file_size_bytes * 2) / (1024**2)


def read_csv(
    filepath: Union[str, Path],
    optimize: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    verbose: bool = False,
    use_polars: bool = True,
    schema_path: Union[str, Path, None] = None,
    save_schema: bool = False,
    memory_threshold: float = 0.7,
    auto_chunk: bool = True,
    chunksize: int = 100000,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a CSV file using Polars engine (if available), then converts to optimized Pandas.

    Automatically switches to chunked reading when file is too large to fit in memory.
    This function is often 5-10x faster at parsing CSVs than pandas.read_csv, and the
    resulting DataFrame uses significantly less memory due to automatic optimization.

    Args:
        filepath: Path to CSV file
        optimize: If True, apply diet optimization after reading (default: True)
        aggressive: If True, use aggressive optimization (float16, etc.)
        categorical_threshold: Threshold for converting objects to categories
        verbose: If True, print memory reduction statistics
        use_polars: If True and Polars is available, use it for parsing (default: True)
        schema_path: Optional path to schema file for consistent typing
        save_schema: If True, save schema after optimization (only with chunked reading)
        memory_threshold: Use chunked reading if estimated memory > threshold * available (default: 0.7)
        auto_chunk: If True, automatically use chunked reading for large files (default: True)
        chunksize: Number of rows per chunk when using chunked reading (default: 100,000)
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

        >>> # Use saved schema for consistent typing
        >>> df = read_csv("data.csv", schema_path="data.diet_schema.json")

        >>> # Large files are automatically chunked
        >>> df = read_csv("huge_file.csv")  # Automatically uses chunked reading
    """
    filepath = Path(filepath)

    # Check if we should use chunked reading
    use_chunked = False
    if auto_chunk:
        try:
            estimated_memory = _estimate_csv_memory_mb(filepath)
            available_memory = _get_available_memory_mb()

            if estimated_memory > (available_memory * memory_threshold):
                use_chunked = True
                if verbose:
                    print(
                        f"File size: ~{estimated_memory:.0f}MB, "
                        f"Available: {available_memory:.0f}MB - "
                        f"Using chunked reading"
                    )
        except Exception:
            # If we can't check, proceed with normal reading
            pass

    # Use chunked reading for large files
    if use_chunked:
        return _read_csv_chunked(
            filepath,
            chunksize=chunksize,
            optimize=optimize,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
            verbose=verbose,
            schema_path=schema_path,
            save_schema=save_schema,
            **kwargs,
        )

    # Normal reading path
    filepath_str = str(filepath)

    # Try to use Polars for fast parsing
    if use_polars and POLARS_AVAILABLE:
        try:
            # Step 1: Fast Read with Polars
            # Polars is multi-threaded and much faster at parsing CSVs
            pl_df = pl.read_csv(filepath_str, **kwargs)

            # Step 2: Convert to Pandas
            pd_df = pl_df.to_pandas()

            if verbose:
                print("Loaded with Polars engine (fast mode)")

        except Exception as e:
            if verbose:
                print(f"Polars parsing failed ({e}), falling back to Pandas")
            # Fallback to standard Pandas
            pd_df = pd.read_csv(filepath_str, **kwargs)
    else:
        # Use standard Pandas
        if verbose and use_polars and not POLARS_AVAILABLE:
            print("Polars not installed, using standard Pandas reader")
        pd_df = pd.read_csv(filepath_str, **kwargs)

    # Apply schema if provided
    if schema_path:
        from .schema import apply_schema, load_schema

        if Path(schema_path).exists():
            if verbose:
                print(f"Applying schema from {schema_path}")
            schema = load_schema(schema_path)
            pd_df = apply_schema(pd_df, schema)
            return pd_df

    # Step 3: Apply the Diet immediately
    if optimize:
        result = diet(
            pd_df,
            verbose=verbose,
            aggressive=aggressive,
            categorical_threshold=categorical_threshold,
        )

        # Save schema if requested
        if save_schema and schema_path:
            from .schema import save_schema as save_schema_func

            save_schema_func(result, schema_path)
            if verbose:
                print(f"Saved schema to {schema_path}")

        return result

    return pd_df


def _read_csv_chunked(
    filepath: Union[str, Path],
    chunksize: int,
    optimize: bool,
    aggressive: bool,
    categorical_threshold: float,
    verbose: bool,
    schema_path: Union[str, Path, None],
    save_schema: bool,
    **kwargs,
) -> pd.DataFrame:
    """Internal function for chunked CSV reading."""
    from .schema import apply_schema, load_schema
    from .schema import save_schema as save_schema_func

    filepath = str(filepath)
    chunks = []
    schema = None

    # Load schema if provided
    if schema_path and Path(schema_path).exists():
        schema = load_schema(schema_path)
        if verbose:
            print(f"Loaded schema from {schema_path}")

    # Read in chunks
    chunk_reader = pd.read_csv(filepath, chunksize=chunksize, **kwargs)

    for i, chunk in enumerate(chunk_reader):
        if verbose:
            print(f"Processing chunk {i + 1}...", end=" ")

        # Apply schema if available
        if schema:
            chunk = apply_schema(chunk, schema)
        elif optimize:
            # Optimize the chunk
            chunk = diet(
                chunk,
                aggressive=aggressive,
                categorical_threshold=categorical_threshold,
                verbose=False,
            )

        chunks.append(chunk)

        # Save schema from first chunk if requested
        if i == 0 and save_schema and schema_path:
            save_schema_func(chunk, schema_path)
            if verbose:
                print(f"Saved schema to {schema_path}")

        if verbose:
            chunk_mb = chunk.memory_usage(deep=True).sum() / 1024**2
            print(f"{chunk_mb:.2f} MB")

    # Concatenate all chunks
    if verbose:
        print("Concatenating chunks...")

    result = pd.concat(chunks, ignore_index=True)

    if verbose:
        total_mb = result.memory_usage(deep=True).sum() / 1024**2
        print(f"Total memory: {total_mb:.2f} MB")

    return result


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
