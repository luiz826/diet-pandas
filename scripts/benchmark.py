"""
Performance Benchmarking Script for Diet Pandas

This script benchmarks diet-pandas against standard pandas across various
data types and sizes to demonstrate memory and performance improvements.
"""

import time
from typing import Dict, List

import numpy as np
import pandas as pd

import dietpandas as dp


def create_test_dataframe(size: int = 100000) -> pd.DataFrame:
    """
    Creates a test DataFrame with various data types.

    Args:
        size: Number of rows in the DataFrame

    Returns:
        Test DataFrame with mixed types
    """
    np.random.seed(42)

    return pd.DataFrame(
        {
            # Integer columns (will benefit from downcasting)
            "small_int": np.random.randint(0, 100, size),
            "medium_int": np.random.randint(0, 10000, size),
            "large_int": np.random.randint(0, 1000000, size),
            # Float columns
            "float_col": np.random.randn(size),
            "precise_float": np.random.randn(size) * 0.001,
            # Categorical string columns
            "category_low": np.random.choice(["A", "B", "C", "D"], size),
            "category_high": np.random.choice(["cat_" + str(i) for i in range(100)], size),
            # Sparse columns (mostly zeros)
            "sparse_binary": np.random.choice([0, 1], size, p=[0.95, 0.05]),
            "sparse_numeric": np.random.choice([0, 1, 2, 3], size, p=[0.90, 0.05, 0.03, 0.02]),
            # DateTime column
            "date": pd.date_range("2020-01-01", periods=size, freq="1min"),
            # Text column with high cardinality
            "text_id": ["text_" + str(i % 1000) for i in range(size)],
        }
    )


def measure_memory(df: pd.DataFrame) -> float:
    """
    Measure DataFrame memory usage in MB.

    Args:
        df: DataFrame to measure

    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1e6


def run_benchmark(sizes: List[int]) -> Dict:
    """
    Run benchmarks across different dataset sizes.

    Args:
        sizes: List of dataset sizes to test

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "size": [],
        "pandas_memory_mb": [],
        "diet_memory_mb": [],
        "memory_reduction_pct": [],
        "diet_time_sec": [],
        "speedup": [],
    }

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking with {size:,} rows")
        print(f"{'='*60}")

        # Create test data
        df_original = create_test_dataframe(size)
        original_memory = measure_memory(df_original)

        # Measure diet optimization time
        start_time = time.time()
        df_optimized = dp.diet(df_original, verbose=False)
        diet_time = time.time() - start_time

        optimized_memory = measure_memory(df_optimized)
        reduction = 100 * (original_memory - optimized_memory) / original_memory

        # Store results
        results["size"].append(size)
        results["pandas_memory_mb"].append(original_memory)
        results["diet_memory_mb"].append(optimized_memory)
        results["memory_reduction_pct"].append(reduction)
        results["diet_time_sec"].append(diet_time)
        results["speedup"].append(1)  # Placeholder

        # Print results
        print(f"\nOriginal Memory:  {original_memory:>10.2f} MB")
        print(f"Optimized Memory: {optimized_memory:>10.2f} MB")
        print(f"Reduction:        {reduction:>10.1f}%")
        print(f"Optimization Time: {diet_time:>9.3f} sec")

        # Show detailed breakdown by column
        print("\nPer-Column Memory Usage:")
        print("-" * 60)
        for col in df_original.columns:
            orig_mem = df_original[col].memory_usage(deep=True) / 1e6
            opt_mem = df_optimized[col].memory_usage(deep=True) / 1e6
            col_reduction = 100 * (orig_mem - opt_mem) / orig_mem if orig_mem > 0 else 0

            print(
                f"{col:20s} {df_original[col].dtype!s:12s} -> {df_optimized[col].dtype!s:12s} "
                f"{orig_mem:6.2f}MB -> {opt_mem:6.2f}MB ({col_reduction:5.1f}%)"
            )

    return pd.DataFrame(results)


def benchmark_file_io(filepath: str = "/tmp/benchmark_data.csv") -> None:
    """
    Benchmark file I/O operations with and without optimization.

    Args:
        filepath: Path to temporary file for testing
    """
    print(f"\n{'='*60}")
    print("File I/O Benchmark")
    print(f"{'='*60}")

    # Create test data
    df = create_test_dataframe(100000)

    # Save test file
    df.to_csv(filepath, index=False)

    # Benchmark standard pandas read
    start = time.time()
    df_pandas = pd.read_csv(filepath)
    pandas_time = time.time() - start
    pandas_memory = measure_memory(df_pandas)

    # Benchmark diet-pandas read
    start = time.time()
    df_diet = dp.read_csv(filepath, verbose=False)
    diet_time = time.time() - start
    diet_memory = measure_memory(df_diet)

    print("\nStandard pandas.read_csv:")
    print(f"  Time:   {pandas_time:.3f} sec")
    print(f"  Memory: {pandas_memory:.2f} MB")

    print("\nDiet-pandas read_csv:")
    print(f"  Time:   {diet_time:.3f} sec")
    print(f"  Memory: {diet_memory:.2f} MB")

    speedup = pandas_time / diet_time
    memory_reduction = 100 * (pandas_memory - diet_memory) / pandas_memory

    print("\nImprovement:")
    print(f"  Speed:  {speedup:.2f}x")
    print(f"  Memory: {memory_reduction:.1f}% reduction")

    # Cleanup
    import os

    if os.path.exists(filepath):
        os.remove(filepath)


def benchmark_sparse_optimization() -> None:
    """
    Benchmark sparse optimization on sparse data.
    """
    print(f"\n{'='*60}")
    print("Sparse Data Optimization Benchmark")
    print(f"{'='*60}")

    size = 100000

    # Create highly sparse data
    df = pd.DataFrame(
        {
            "sparse_95": np.random.choice([0, 1], size, p=[0.95, 0.05]),
            "sparse_99": np.random.choice([0, 1], size, p=[0.99, 0.01]),
            "sparse_999": np.random.choice([0, 1], size, p=[0.999, 0.001]),
            "dense": np.random.randint(0, 100, size),
        }
    )

    original_memory = measure_memory(df)

    # Optimize with sparse enabled
    df_sparse = dp.diet(df, optimize_sparse_cols=True, verbose=False)
    sparse_memory = measure_memory(df_sparse)

    print(f"\nOriginal Memory (dense):  {original_memory:.2f} MB")
    print(f"Optimized Memory (sparse): {sparse_memory:.2f} MB")
    print(f"Reduction: {100 * (original_memory - sparse_memory) / original_memory:.1f}%")

    print("\nData types:")
    for col in df_sparse.columns:
        print(f"  {col}: {df_sparse[col].dtype}")


def main():
    """
    Main benchmarking function.
    """
    print("\n" + "=" * 60)
    print("DIET-PANDAS PERFORMANCE BENCHMARKS")
    print("=" * 60)

    # Run benchmarks on different sizes
    sizes = [10000, 50000, 100000, 500000]
    results_df = run_benchmark(sizes)

    # Show summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # File I/O benchmark
    benchmark_file_io()

    # Sparse optimization benchmark
    benchmark_sparse_optimization()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
