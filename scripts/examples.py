"""
Example usage of Diet Pandas.

This script demonstrates the key features of the diet-pandas library.
"""

import numpy as np
import pandas as pd

import dietpandas as dp


def create_sample_data():
    """Create a sample DataFrame with wasteful types."""
    np.random.seed(42)

    data = {
        # Small integers stored as int64 (wasteful)
        "user_age": np.random.randint(18, 80, size=10000),
        "product_rating": np.random.randint(1, 6, size=10000),
        # Floats stored as float64 (wasteful for ML)
        "price": np.random.uniform(10, 1000, size=10000),
        "discount": np.random.uniform(0, 0.5, size=10000),
        # Repetitive strings stored as object (wasteful)
        "category": np.random.choice(["Electronics", "Books", "Clothing", "Food"], size=10000),
        "country": np.random.choice(["USA", "UK", "Germany", "France", "Japan"], size=10000),
        # High-cardinality string (should stay as object)
        "transaction_id": [f"TXN_{i:08d}" for i in range(10000)],
    }

    df = pd.DataFrame(data)

    # Force wasteful types
    df["user_age"] = df["user_age"].astype("int64")
    df["product_rating"] = df["product_rating"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["discount"] = df["discount"].astype("float64")

    return df


def example_1_basic_optimization():
    """Example 1: Basic DataFrame optimization."""
    print("=" * 80)
    print("Example 1: Basic DataFrame Optimization")
    print("=" * 80)

    df = create_sample_data()

    print("\nBEFORE Optimization:")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("\nData types:")
    print(df.dtypes)

    print("\nApplying Diet...")
    df_optimized = dp.diet(df, verbose=True)

    print("\nAFTER Optimization:")
    print("\nData types:")
    print(df_optimized.dtypes)

    print("\nData integrity check:")
    print(f"Age values preserved: {(df['user_age'] == df_optimized['user_age']).all()}")
    print(f"Price values preserved (approx): {np.allclose(df['price'], df_optimized['price'])}")


def example_2_fast_csv_reading():
    """Example 2: Fast CSV reading with Polars."""
    print("\n" + "=" * 80)
    print("Example 2: Fast CSV Reading")
    print("=" * 80)

    # Create a sample CSV
    df = create_sample_data()
    csv_path = "sample_data.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nCreated sample CSV: {csv_path}")

    # Read with standard Pandas
    print("\nStandard pandas.read_csv():")
    import time

    start = time.time()
    df_pandas = pd.read_csv(csv_path)
    pandas_time = time.time() - start
    pandas_mem = df_pandas.memory_usage(deep=True).sum() / 1e6
    print(f"   Time: {pandas_time:.3f}s, Memory: {pandas_mem:.2f} MB")

    # Read with Diet Pandas
    print("\nDiet Pandas read_csv():")
    start = time.time()
    df_diet = dp.read_csv(csv_path, verbose=True)
    diet_time = time.time() - start
    diet_mem = df_diet.memory_usage(deep=True).sum() / 1e6
    print(f"   Time: {diet_time:.3f}s, Memory: {diet_mem:.2f} MB")

    print("\nPerformance improvement:")
    print(f"   Speed: {pandas_time/diet_time:.2f}x faster")
    print(f"   Memory: {(1 - diet_mem/pandas_mem)*100:.1f}% reduction")

    # Cleanup
    import os

    os.remove(csv_path)


def example_3_aggressive_mode():
    """Example 3: Aggressive optimization mode."""
    print("\n" + "=" * 80)
    print("Example 3: Aggressive Mode (Keto Diet)")
    print("=" * 80)

    df = create_sample_data()

    print("\nSafe Mode (default):")
    df_safe = dp.diet(df, aggressive=False, verbose=True)

    print("\nAggressive Mode:")
    df_keto = dp.diet(df, aggressive=True, verbose=True)

    print("\nFloat precision comparison:")
    print(f"Safe mode float type: {df_safe['price'].dtype}")
    print(f"Aggressive mode float type: {df_keto['price'].dtype}")

    print("\nPrecision loss check:")
    original_sum = df["price"].sum()
    safe_sum = df_safe["price"].sum()
    keto_sum = df_keto["price"].sum()

    print(f"Original sum: {original_sum:.2f}")
    print(
        f"Safe mode sum: {safe_sum:.2f} (error: {abs(original_sum-safe_sum)/original_sum*100:.4f}%)"
    )
    print(
        f"Keto mode sum: {keto_sum:.2f} (error: {abs(original_sum-keto_sum)/original_sum*100:.4f}%)"
    )


def example_4_memory_report():
    """Example 4: Detailed memory report."""
    print("\n" + "=" * 80)
    print("Example 4: Memory Usage Report")
    print("=" * 80)

    df = create_sample_data()
    df_optimized = dp.diet(df, verbose=False)

    print("\nMemory Report:")
    report = dp.get_memory_report(df_optimized)
    print(report.to_string(index=False))


def example_5_integration():
    """Example 5: Integration with ML libraries."""
    print("\n" + "=" * 80)
    print("Example 5: Integration with Scikit-Learn")
    print("=" * 80)

    df = create_sample_data()
    df_optimized = dp.diet(df, verbose=False)

    # Prepare for ML
    df_ml = df_optimized.drop(columns=["transaction_id"])

    # Convert categories to numeric for ML
    from sklearn.preprocessing import LabelEncoder

    for col in df_ml.select_dtypes(include=["category"]).columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    print("\nDataFrame ready for Machine Learning:")
    print(f"   Shape: {df_ml.shape}")
    print(f"   Memory: {df_ml.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"   Dtypes:\n{df_ml.dtypes}")

    # Could now use with sklearn
    # from sklearn.ensemble import RandomForestClassifier
    # X = df_ml.drop('target', axis=1)
    # y = df_ml['target']
    # model = RandomForestClassifier()
    # model.fit(X, y)


if __name__ == "__main__":
    print("\nDiet Pandas - Example Usage\n")

    example_1_basic_optimization()
    example_2_fast_csv_reading()
    example_3_aggressive_mode()
    example_4_memory_report()
    example_5_integration()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
