"""
Diet Pandas - Analysis Module

This module provides functions for analyzing DataFrames and estimating
optimization opportunities before applying changes.
"""

import numpy as np
import pandas as pd

from .core import optimize_bool, optimize_float, optimize_int, optimize_obj


def analyze(
    df: pd.DataFrame,
    aggressive: bool = False,
    categorical_threshold: float = 0.5,
    sparse_threshold: float = 0.9,
    optimize_datetimes: bool = True,
    optimize_sparse_cols: bool = False,
    optimize_bools: bool = True,
) -> pd.DataFrame:
    """
    Analyze DataFrame and return optimization recommendations without modifying it.

    This function performs a "dry run" of the optimization process, providing
    insights into potential memory savings and recommended data type changes
    without actually modifying the DataFrame.

    Args:
        df: Input DataFrame to analyze
        aggressive: If True, simulate aggressive optimization (float16)
        categorical_threshold: Threshold for converting objects to categories
        sparse_threshold: Threshold for converting to sparse format
        optimize_datetimes: If True, include datetime optimization analysis
        optimize_sparse_cols: If True, check for sparse optimization opportunities
        optimize_bools: If True, check for boolean optimization opportunities

    Returns:
        DataFrame with columns:
        - column: Column name
        - current_dtype: Current data type
        - recommended_dtype: Recommended data type after optimization
        - current_memory_mb: Current memory usage in MB
        - optimized_memory_mb: Estimated memory after optimization in MB
        - savings_mb: Memory savings in MB
        - savings_percent: Percent reduction in memory
        - reasoning: Explanation of the recommendation

    Examples:
        >>> df = pd.DataFrame({'age': [25, 30, 35], 'name': ['A', 'B', 'A']})
        >>> analysis = analyze(df)
        >>> print(analysis)
    """
    recommendations = []

    for col in df.columns:
        dtype = df[col].dtype
        series = df[col]

        # Skip all-NaN columns
        if series.isna().all():
            continue

        current_memory = series.memory_usage(deep=True)
        current_dtype_str = str(dtype)
        recommended_dtype_str = current_dtype_str
        optimized_memory = current_memory
        reasoning = "No optimization needed"

        # Try boolean optimization first
        if optimize_bools and (np.issubdtype(dtype, np.integer) or dtype == "object"):
            try:
                optimized = optimize_bool(series)
                if optimized.dtype in ["boolean", bool]:
                    recommended_dtype_str = "boolean"
                    optimized_memory = optimized.memory_usage(deep=True)
                    reasoning = (
                        "Boolean-like values detected (0/1 or yes/no). "
                        "Convert to boolean for 87.5% memory reduction."
                    )
                    recommendations.append(
                        _create_recommendation(
                            col,
                            current_dtype_str,
                            recommended_dtype_str,
                            current_memory,
                            optimized_memory,
                            reasoning,
                        )
                    )
                    continue
            except Exception:
                pass

        # Try integer optimization
        if np.issubdtype(dtype, np.integer):
            try:
                optimized = optimize_int(series)
                if optimized.dtype != dtype:
                    recommended_dtype_str = str(optimized.dtype)
                    optimized_memory = optimized.memory_usage(deep=True)
                    reasoning = (
                        f"Integer values fit in smaller type. "
                        f"Range: [{series.min()}, {series.max()}]"
                    )
                    if optimize_sparse_cols:
                        from .core import optimize_sparse

                        sparse_optimized = optimize_sparse(optimized, sparse_threshold)
                        if isinstance(sparse_optimized.dtype, pd.SparseDtype):
                            recommended_dtype_str = str(sparse_optimized.dtype)
                            optimized_memory = sparse_optimized.memory_usage(deep=True)
                        sparse_pct = sparse_threshold * 100
                        reasoning += (
                            f" Sparse format recommended "
                            f"({sparse_pct}% values are identical)."
                        )
            except Exception:
                pass

        # Try float optimization
        elif np.issubdtype(dtype, np.floating):
            try:
                optimized = optimize_float(series, aggressive=aggressive)
                if optimized.dtype != dtype:
                    recommended_dtype_str = str(optimized.dtype)
                    optimized_memory = optimized.memory_usage(deep=True)
                    if aggressive:
                        reasoning = (
                            "Float64 → float16 (aggressive mode). "
                            "⚠️  May lose precision for large/small values."
                        )
                    else:
                        reasoning = (
                            "Float64 → float32 conversion. "
                            "Safe for most ML/scientific applications."
                        )
                    if optimize_sparse_cols:
                        from .core import optimize_sparse

                        sparse_optimized = optimize_sparse(optimized, sparse_threshold)
                        if isinstance(sparse_optimized.dtype, pd.SparseDtype):
                            recommended_dtype_str = str(sparse_optimized.dtype)
                            optimized_memory = sparse_optimized.memory_usage(deep=True)
                        reasoning += " Sparse format also recommended."
            except Exception:
                pass

        # Try datetime optimization
        elif pd.api.types.is_datetime64_any_dtype(dtype) and optimize_datetimes:
            # Datetime optimization typically doesn't change dtype significantly
            reasoning = "Datetime column (already efficient)"

        # Try object/string optimization
        elif dtype == "object":
            try:
                unique_ratio = series.nunique() / len(series) if len(series) > 0 else 1
                if unique_ratio < categorical_threshold:
                    optimized = optimize_obj(series, categorical_threshold)
                    if optimized.dtype.name == "category":
                        recommended_dtype_str = "category"
                        optimized_memory = optimized.memory_usage(deep=True)
                        reasoning = (
                            f"Low cardinality ({unique_ratio:.1%} unique values). "
                            f"Convert to category for memory savings."
                        )
                else:
                    reasoning = (
                        f"High cardinality ({unique_ratio:.1%} unique). "
                        f"⚠️  Not suitable for category conversion."
                    )
            except Exception:
                pass

        recommendations.append(
            _create_recommendation(
                col,
                current_dtype_str,
                recommended_dtype_str,
                current_memory,
                optimized_memory,
                reasoning,
            )
        )

    # Create DataFrame from recommendations
    if not recommendations:
        return pd.DataFrame(
            columns=[
                "column",
                "current_dtype",
                "recommended_dtype",
                "current_memory_mb",
                "optimized_memory_mb",
                "savings_mb",
                "savings_percent",
                "reasoning",
            ]
        )

    result = pd.DataFrame(recommendations)
    # Sort by savings (descending)
    result = result.sort_values("savings_mb", ascending=False).reset_index(drop=True)

    return result


def _create_recommendation(
    column: str,
    current_dtype: str,
    recommended_dtype: str,
    current_memory: int,
    optimized_memory: int,
    reasoning: str,
) -> dict:
    """Helper function to create a recommendation dictionary."""
    savings = current_memory - optimized_memory
    savings_percent = (savings / current_memory * 100) if current_memory > 0 else 0

    return {
        "column": column,
        "current_dtype": current_dtype,
        "recommended_dtype": recommended_dtype,
        "current_memory_mb": current_memory / 1e6,
        "optimized_memory_mb": optimized_memory / 1e6,
        "savings_mb": savings / 1e6,
        "savings_percent": savings_percent,
        "reasoning": reasoning,
    }


def get_optimization_summary(df: pd.DataFrame, **kwargs) -> dict:
    """
    Get a summary of optimization opportunities.

    Args:
        df: Input DataFrame to analyze
        **kwargs: Additional arguments passed to analyze()

    Returns:
        Dictionary with summary statistics:
        - total_memory_mb: Current total memory usage
        - optimized_memory_mb: Estimated memory after optimization
        - total_savings_mb: Total memory savings
        - total_savings_percent: Overall percent reduction
        - optimizable_columns: Number of columns that can be optimized

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        >>> summary = get_optimization_summary(df)
        >>> print(f"Potential savings: {summary['total_savings_percent']:.1f}%")
    """
    analysis = analyze(df, **kwargs)

    total_current = analysis["current_memory_mb"].sum()
    total_optimized = analysis["optimized_memory_mb"].sum()
    total_savings = analysis["savings_mb"].sum()
    total_savings_percent = (
        (total_savings / total_current * 100) if total_current > 0 else 0
    )
    optimizable_columns = (analysis["savings_mb"] > 0).sum()

    return {
        "total_memory_mb": total_current,
        "optimized_memory_mb": total_optimized,
        "total_savings_mb": total_savings,
        "total_savings_percent": total_savings_percent,
        "optimizable_columns": optimizable_columns,
        "total_columns": len(analysis),
    }


def estimate_memory_reduction(df: pd.DataFrame, **kwargs) -> float:
    """
    Quick estimate of potential memory reduction percentage.

    Args:
        df: Input DataFrame to analyze
        **kwargs: Additional arguments passed to analyze()

    Returns:
        Estimated memory reduction as a percentage (0-100)

    Examples:
        >>> df = pd.DataFrame({'year': [2020, 2021], 'val': [1.1, 2.2]})
        >>> reduction = estimate_memory_reduction(df)
        >>> print(f"Estimated reduction: {reduction:.1f}%")
    """
    summary = get_optimization_summary(df, **kwargs)
    return summary["total_savings_percent"]
