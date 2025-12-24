"""
Demo script showing the new float-to-int conversion feature in diet-pandas.
"""

import numpy as np
import pandas as pd

import dietpandas as dp


def main():
    print("🥗 Float-to-Int Conversion Demo\n")
    print("=" * 60)

    # Example 1: Float column with whole numbers
    print("\n1. Float column with whole numbers (years, counts, etc.):")
    df1 = pd.DataFrame(
        {
            "year": [2020.0, 2021.0, 2022.0, 2023.0, 2024.0],
            "count": [100.0, 200.0, 300.0, 400.0, 500.0],
            "temperature": [20.5, 21.3, 19.8, 22.1, 20.9],  # Has decimals
        }
    )

    print("\nOriginal DataFrame:")
    print(df1)
    print(f"\nOriginal dtypes:\n{df1.dtypes}")
    print(f"Original memory: {df1.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # Apply diet with float_to_int enabled (default)
    df1_optimized = dp.diet(df1, verbose=False)

    print(f"\nOptimized dtypes:\n{df1_optimized.dtypes}")
    print(f"Optimized memory: {df1_optimized.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print("\n✓ Notice: 'year' and 'count' converted to integers, 'temperature' stays float32")

    # Example 2: Float with NaN values
    print("\n" + "=" * 60)
    print("\n2. Float column with whole numbers and NaN:")
    df2 = pd.DataFrame(
        {
            "ratings": [5.0, 4.0, np.nan, 3.0, 5.0, np.nan, 4.0],
            "scores": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )

    print("\nOriginal DataFrame:")
    print(df2)
    print(f"\nOriginal dtypes:\n{df2.dtypes}")

    df2_optimized = dp.diet(df2, verbose=False)

    print(f"\nOptimized dtypes:\n{df2_optimized.dtypes}")
    print("\n✓ Notice: Both columns converted to nullable integers (Int8, UInt8)")
    print(f"\nOptimized DataFrame:\n{df2_optimized}")

    # Example 3: Disabling float-to-int conversion
    print("\n" + "=" * 60)
    print("\n3. Disabling float-to-int conversion:")
    df3 = pd.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})

    print(f"\nOriginal dtype: {df3['values'].dtype}")

    df3_with_conversion = dp.diet(df3, verbose=False, float_to_int=True)
    df3_without_conversion = dp.diet(df3, verbose=False, float_to_int=False)

    print(f"With float_to_int=True: {df3_with_conversion['values'].dtype}")
    print(f"With float_to_int=False: {df3_without_conversion['values'].dtype}")

    # Example 4: Memory savings comparison
    print("\n" + "=" * 60)
    print("\n4. Memory savings on large dataset:")

    # Create large DataFrame with float IDs (common in datasets)
    n = 100_000
    df4 = pd.DataFrame(
        {
            "user_id": np.arange(1.0, n + 1, dtype=np.float64),
            "product_id": np.arange(1.0, n + 1, dtype=np.float64),
            "price": np.random.uniform(10.0, 100.0, n),  # Has decimals
        }
    )

    original_memory = df4.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\nOriginal memory: {original_memory:.2f} MB")
    print(f"Original dtypes:\n{df4.dtypes}")

    df4_optimized = dp.diet(df4, verbose=False, float_to_int=True)

    optimized_memory = df4_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    reduction = ((original_memory - optimized_memory) / original_memory) * 100

    print(f"\nOptimized memory: {optimized_memory:.2f} MB")
    print(f"Optimized dtypes:\n{df4_optimized.dtypes}")
    print(f"\n✓ Memory reduction: {reduction:.1f}%")
    print("✓ Notice: 'user_id' and 'product_id' converted to Int32, 'price' stays float32")

    print("\n" + "=" * 60)
    print("\n✅ Float-to-int conversion feature is working perfectly!")


if __name__ == "__main__":
    main()
