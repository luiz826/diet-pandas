"""
Diet Pandas - Core Optimization Logic

This module contains the core functions for optimizing Pandas DataFrame memory usage
by intelligently downcasting numeric types and converting strings to categories.
"""

import pandas as pd
import numpy as np
from typing import Optional


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


def diet(
    df: pd.DataFrame,
    verbose: bool = True,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Main function to optimize DataFrame memory usage.
    
    This function iterates over all columns and applies appropriate optimizations:
    - Integers: Downcast to smallest safe type (int8, int16, uint8, etc.)
    - Floats: Convert to float32 (or float16 in aggressive mode)
    - Objects: Convert to category if cardinality is low
    
    Args:
        df: Input DataFrame to optimize
        verbose: If True, print memory reduction statistics
        aggressive: If True, use more aggressive optimization (may lose precision)
        categorical_threshold: Threshold for converting objects to categories
        inplace: If True, modify the DataFrame in place (default: False)
        
    Returns:
        Optimized DataFrame with reduced memory usage
        
    Examples:
        >>> df = pd.DataFrame({'year': [2020, 2021, 2022], 'val': [1.1, 2.2, 3.3]})
        >>> optimized = diet(df)
        Diet Complete: Memory reduced by 62.5%
           0.00MB -> 0.00MB
    """
    if not inplace:
        df = df.copy()
    
    start_mem = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        dtype = df[col].dtype
        
        # Skip columns with all NaN values
        if df[col].isna().all():
            continue
        
        # Optimize Integers
        if np.issubdtype(dtype, np.integer):
            df[col] = optimize_int(df[col])
            
        # Optimize Floats
        elif np.issubdtype(dtype, np.floating):
            df[col] = optimize_float(df[col], aggressive=aggressive)
            
        # Optimize Objects (Strings)
        elif dtype == 'object':
            df[col] = optimize_obj(df[col], categorical_threshold=categorical_threshold)
            
    end_mem = df.memory_usage(deep=True).sum()
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f"Diet Complete: Memory reduced by {reduction:.1f}%")
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
    
    report = pd.DataFrame({
        'column': mem_usage.index,
        'dtype': [df[col].dtype if col != 'Index' else 'Index' for col in mem_usage.index],
        'memory_bytes': mem_usage.values,
        'memory_mb': mem_usage.values / 1e6
    })
    
    report['percent_of_total'] = 100 * report['memory_bytes'] / report['memory_bytes'].sum()
    report = report.sort_values('memory_bytes', ascending=False).reset_index(drop=True)
    
    return report
