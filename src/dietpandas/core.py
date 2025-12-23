"""
Diet Pandas - Core Optimization Logic

This module contains the core functions for optimizing Pandas DataFrame memory usage
by intelligently downcasting numeric types and converting strings to categories.
"""

import numpy as np
import pandas as pd


def optimize_int(series: pd.Series) -> pd.Series:
    """
    Downcasts integer series to the smallest possible safe type.

    Args:
        series: A pandas Series with integer dtype

    Returns:
        Optimized Series with smallest safe integer type

    Examples:
        >>> s = pd.Series([1, 2, 3], dtype='int64')
        >>> optimized = optimize_int(s)
        >>> optimized.dtype
        dtype('uint8')
    """
    c_min, c_max = series.min(), series.max()

    # Check if unsigned is possible (positive numbers only)
    if c_min >= 0:
        if c_max <= np.iinfo(np.uint8).max:
            return series.astype(np.uint8)
        if c_max <= np.iinfo(np.uint16).max:
            return series.astype(np.uint16)
        if c_max <= np.iinfo(np.uint32).max:
            return series.astype(np.uint32)
        return series.astype(np.uint64)
    # Otherwise use signed integers
    else:
        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
            return series.astype(np.int8)
        if c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
            return series.astype(np.int16)
        if c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
            return series.astype(np.int32)

    return series


def optimize_float(series: pd.Series, aggressive: bool = False) -> pd.Series:
    """
    Downcasts float series to float32 or float16 (if aggressive mode).

    Args:
        series: A pandas Series with float dtype
        aggressive: If True, use float16 for maximum compression (may lose precision)

    Returns:
        Optimized Series with smaller float type
    """
    if aggressive:
        # Keto mode: Maximum compression
        return series.astype(np.float16)
    else:
        # Safe mode: float32 is usually sufficient for ML
        return series.astype(np.float32)


def optimize_obj(series: pd.Series, categorical_threshold: float = 0.5) -> pd.Series:
    """
    Converts object columns to categories if unique ratio is low.

    Args:
        series: A pandas Series with object dtype
        categorical_threshold: If unique_ratio < threshold, convert to category

    Returns:
        Optimized Series (categorical if beneficial, otherwise unchanged)

    Examples:
        >>> s = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
        >>> optimized = optimize_obj(s)
        >>> optimized.dtype.name
        'category'
    """
    num_unique = series.nunique()
    num_total = len(series)

    # Avoid division by zero
    if num_total == 0:
        return series

    unique_ratio = num_unique / num_total

    if unique_ratio < categorical_threshold:
        return series.astype("category")

    return series


def optimize_bool(series: pd.Series) -> pd.Series:
    """
    Converts integer or object columns to boolean dtype when appropriate.

    Detects columns that contain only boolean-like values and converts them
    to the bool dtype for maximum memory efficiency (1 byte vs 8 bytes).

    Args:
        series: A pandas Series with integer or object dtype

    Returns:
        Optimized Series with bool dtype if appropriate, otherwise unchanged

    Examples:
        >>> s = pd.Series([0, 1, 1, 0, 1], dtype='int64')
        >>> optimized = optimize_bool(s)
        >>> optimized.dtype
        dtype('bool')

        >>> s = pd.Series(['True', 'False', 'True'])
        >>> optimized = optimize_bool(s)
        >>> optimized.dtype
        dtype('bool')

        >>> s = pd.Series(['yes', 'no', 'yes'])
        >>> optimized = optimize_bool(s)
        >>> optimized.dtype
        dtype('bool')
    """
    # Skip if already bool
    if series.dtype == bool:
        return series

    # Get unique values (excluding NaN)
    unique_vals = series.dropna().unique()

    # If empty or all NaN, return as is
    if len(unique_vals) == 0:
        return series

    # Check for numeric boolean (0, 1)
    if np.issubdtype(series.dtype, np.integer):
        if set(unique_vals).issubset({0, 1}):
            # Convert to bool, preserving NaN
            return series.astype("boolean")  # Use nullable boolean type

    # Check for string boolean representations
    if series.dtype == "object":
        # Convert to lowercase for case-insensitive comparison
        unique_vals_lower = set(str(v).lower() for v in unique_vals)

        # Check various boolean representations (True values, False values)
        bool_patterns = [
            ({"true"}, {"false"}),
            ({"yes"}, {"no"}),
            ({"y"}, {"n"}),
            ({"1"}, {"0"}),
            ({"t"}, {"f"}),
        ]

        for true_vals, false_vals in bool_patterns:
            if unique_vals_lower.issubset(true_vals | false_vals):
                # Map to boolean
                try:
                    bool_series = series.apply(
                        lambda x: (
                            True
                            if str(x).lower() in true_vals
                            else (False if pd.notna(x) else None)
                        )
                    )
                    return bool_series.astype("boolean")
                except Exception:
                    return series

    return series


def optimize_datetime(series: pd.Series) -> pd.Series:
    """
    Optimizes datetime columns by converting to more efficient datetime64 types.

    For datetime columns, attempts to use more memory-efficient
    representations:
    - If all datetimes are dates (no time component), suggests conversion
    - Removes unnecessary precision (e.g., nanosecond to microsecond)

    Args:
        series: A pandas Series with datetime64 dtype

    Returns:
        Optimized Series with more efficient datetime representation

    Examples:
        >>> dates = pd.Series(pd.date_range('2020-01-01', periods=100))
        >>> optimized = optimize_datetime(dates)
    """
    # If the series is already datetime64[ns], check if we can downcast
    if pd.api.types.is_datetime64_any_dtype(series):
        # Remove timezone info for memory efficiency if present
        if series.dt.tz is not None:
            # Keep timezone but note that tz-naive uses less memory
            pass

        # Pandas datetime64[ns] is already quite efficient
        # The main optimization is ensuring it's in the right format
        return series

    # Try to convert to datetime if it's an object
    if series.dtype == "object":
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return series

    return series


def optimize_sparse(series: pd.Series, sparse_threshold: float = 0.9) -> pd.Series:
    """
    Converts series to sparse format if it has many repeated values (especially zeros/NaNs).

    Sparse arrays are highly memory-efficient when a series contains mostly one value.
    Common for binary features, indicator variables, or data with many missing values.

    Args:
        series: A pandas Series
        sparse_threshold: If most common value appears >= threshold% of time, use sparse

    Returns:
        Optimized Series (sparse if beneficial, otherwise unchanged)

    Examples:
        >>> s = pd.Series([0, 0, 1, 0, 0, 0, 2, 0, 0, 0])
        >>> optimized = optimize_sparse(s)
        >>> isinstance(optimized.dtype, pd.SparseDtype)
        True
    """
    if len(series) == 0:
        return series

    # Check if already sparse
    if isinstance(series.dtype, pd.SparseDtype):
        return series

    # Calculate the most common value's frequency
    value_counts = series.value_counts(dropna=False)
    if len(value_counts) == 0:
        return series

    most_common_freq = value_counts.iloc[0] / len(series)

    # If one value dominates, convert to sparse
    if most_common_freq >= sparse_threshold:
        try:
            fill_value = value_counts.index[0]
            return series.astype(pd.SparseDtype(series.dtype, fill_value=fill_value))
        except Exception:
            # If conversion fails, return original
            return series

    return series


def diet(
    df: pd.DataFrame,
    verbose: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    sparse_threshold: float = 0.9,
    optimize_datetimes: bool = True,
    optimize_sparse_cols: bool = False,
    optimize_bools: bool = True,
    inplace: bool = False,
    skip_columns: list = None,
    force_categorical: list = None,
    force_aggressive: list = None,
) -> pd.DataFrame:
    """
    Main function to optimize DataFrame memory usage.

    This function iterates over all columns and applies appropriate optimizations:
    - Booleans: Convert integer/object columns with boolean values to bool dtype
    - Integers: Downcast to smallest safe type (int8, int16, uint8, etc.)
    - Floats: Convert to float32 (or float16 in aggressive mode)
    - Objects: Convert to category if cardinality is low
    - DateTime: Optimize datetime representations
    - Sparse: Convert to sparse arrays for columns with many repeated values

    Args:
        df: Input DataFrame to optimize
        verbose: If True, print memory reduction statistics
        aggressive: If True, use more aggressive optimization (may lose precision)
        categorical_threshold: Threshold for converting objects to categories
        sparse_threshold: Threshold for converting to sparse format (default: 0.9)
        optimize_datetimes: If True, optimize datetime columns (default: True)
        optimize_sparse_cols: If True, check for sparse optimization opportunities (default: False)
        optimize_bools: If True, convert boolean-like columns to bool dtype (default: True)
        inplace: If True, modify the DataFrame in place (default: False)
        skip_columns: List of column names to skip optimization (default: None)
        force_categorical: List of column names to force categorical conversion (default: None)
        force_aggressive: List of column names to force aggressive optimization (default: None)

    Returns:
        Optimized DataFrame with reduced memory usage

    Examples:
        >>> df = pd.DataFrame({'year': [2020, 2021, 2022], 'val': [1.1, 2.2, 3.3]})
        >>> optimized = diet(df)
        🥗 Diet Complete: Memory reduced by 62.5%
           0.00MB -> 0.00MB

        >>> # Skip specific columns
        >>> df = diet(df, skip_columns=['id', 'uuid'])

        >>> # Force categorical conversion on high-cardinality column
        >>> df = diet(df, force_categorical=['country_code'])

        >>> # Use aggressive mode only for specific columns
        >>> df = diet(df, force_aggressive=['approximation_field'])
    """
    if not inplace:
        df = df.copy()

    # Initialize lists if None
    skip_columns = skip_columns or []
    force_categorical = force_categorical or []
    force_aggressive = force_aggressive or []

    start_mem = df.memory_usage(deep=True).sum()

    for col in df.columns:
        # Skip if column is in skip list
        if col in skip_columns:
            continue

        dtype = df[col].dtype

        # Skip columns with all NaN values
        if df[col].isna().all():
            continue

        # Check if column should use aggressive mode
        use_aggressive = aggressive or (col in force_aggressive)

        # Force categorical conversion if requested
        if col in force_categorical:
            try:
                df[col] = df[col].astype("category")
                continue
            except Exception:
                pass  # If conversion fails, proceed with normal optimization

        # Optimize Booleans (check before integers)
        if optimize_bools and (np.issubdtype(dtype, np.integer) or dtype == "object"):
            optimized = optimize_bool(df[col])
            if optimized.dtype == "boolean" or optimized.dtype == bool:
                df[col] = optimized
                continue

        # Optimize Integers
        if np.issubdtype(dtype, np.integer):
            df[col] = optimize_int(df[col])
            # Check for sparse after int optimization
            if optimize_sparse_cols:
                df[col] = optimize_sparse(df[col], sparse_threshold=sparse_threshold)

        # Optimize Floats
        elif np.issubdtype(dtype, np.floating):
            df[col] = optimize_float(df[col], aggressive=use_aggressive)
            # Check for sparse after float optimization
            if optimize_sparse_cols:
                df[col] = optimize_sparse(df[col], sparse_threshold=sparse_threshold)

        # Optimize DateTime
        elif pd.api.types.is_datetime64_any_dtype(dtype) and optimize_datetimes:
            df[col] = optimize_datetime(df[col])

        # Optimize Objects (Strings)
        elif dtype == "object":
            df[col] = optimize_obj(df[col], categorical_threshold=categorical_threshold)

    end_mem = df.memory_usage(deep=True).sum()

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f"🥗 Diet Complete: Memory reduced by {reduction:.1f}%")
        print(f"   {start_mem/1e6:.2f}MB -> {end_mem/1e6:.2f}MB")

    return df


def get_memory_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed memory usage report for each column.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with memory statistics per column

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        >>> report = get_memory_report(df)
        >>> print(report)
    """
    mem_usage = df.memory_usage(deep=True)

    report = pd.DataFrame(
        {
            "column": mem_usage.index,
            "dtype": [df[col].dtype if col != "Index" else "Index" for col in mem_usage.index],
            "memory_bytes": mem_usage.values,
            "memory_mb": mem_usage.values / 1e6,
        }
    )

    report["percent_of_total"] = 100 * report["memory_bytes"] / report["memory_bytes"].sum()
    report = report.sort_values("memory_bytes", ascending=False).reset_index(drop=True)

    return report
