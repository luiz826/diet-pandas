# Quick Start

This guide will get you up and running with Diet Pandas in minutes.

## Basic Usage

### Optimize an Existing DataFrame

The simplest way to use Diet Pandas is to optimize an existing DataFrame:

```python
import pandas as pd
import dietpandas as dp

# Create a DataFrame with inefficient types
df = pd.DataFrame({
    'age': [25, 30, 35, 40],          # int64 (wasteful)
    'score': [95.5, 87.3, 92.1, 88.9], # float64 (wasteful)
    'country': ['USA', 'USA', 'UK', 'USA']  # object (wasteful)
})

print("Before optimization:")
print(df.memory_usage(deep=True))
# age         32 bytes
# score       32 bytes
# country    180 bytes

# Optimize the DataFrame
df_optimized = dp.diet(df)

print("\nAfter optimization:")
print(df_optimized.memory_usage(deep=True))
# age          4 bytes (uint8)
# score       16 bytes (float32)
# country     24 bytes (category)
```

### Load and Optimize CSV Files

Replace `pandas.read_csv()` with `dietpandas.read_csv()`:

```python
import dietpandas as dp

# Instead of: df = pd.read_csv("data.csv")
df = dp.read_csv("data.csv")
# Automatically optimized and 5-10x faster!
```

## Common Use Cases

### Working with Large CSVs

```python
import dietpandas as dp

# Load a large CSV file
df = dp.read_csv("large_sales_data.csv")
# 🥗 Diet Complete: Memory reduced by 67.3%
#    2.3 GB -> 0.75 GB

# Use the DataFrame normally
print(df.head())
print(df.describe())
```

### Aggressive Optimization (Keto Mode)

For maximum compression when you can tolerate some precision loss:

```python
import dietpandas as dp

# Safe mode (default) - preserves precision
df = dp.diet(df, aggressive=False)

# Aggressive mode - maximum compression
df_keto = dp.diet(df, aggressive=True)
# Converts float64 -> float16 for extreme memory savings
```

### In-Place Optimization

Modify the DataFrame directly without creating a copy:

```python
import dietpandas as dp

# Optimize in-place (saves memory during optimization)
dp.diet(df, inplace=True)
# Original df is now optimized
```

### Get Memory Report

See exactly where memory is being used:

```python
import dietpandas as dp

report = dp.get_memory_report(df)
print(report)
#          column      dtype  memory_bytes  memory_mb  percent_of_total
# 0  description     object     450000000     450.00              67.3
# 1     user_id      int64      32000000      32.00              4.8
# 2   timestamp  datetime64      32000000      32.00              4.8
```

## Advanced Features

### DateTime Optimization

```python
import dietpandas as dp

# Enable datetime optimization
df = dp.diet(df, optimize_datetimes=True)
```

### Sparse Data Optimization

For data with many repeated values:

```python
import dietpandas as dp

# Enable sparse optimization (perfect for binary features)
df = dp.diet(df, optimize_sparse_cols=True)
# Converts columns with >90% repeated values to sparse format
```

### Multiple File Formats

```python
import dietpandas as dp

# All these return optimized DataFrames
df_csv = dp.read_csv("data.csv")
df_parquet = dp.read_parquet("data.parquet")
df_excel = dp.read_excel("data.xlsx")
df_json = dp.read_json("data.json")
df_hdf = dp.read_hdf("data.h5", key="dataset")
df_feather = dp.read_feather("data.feather")
```

## Real-World Example

Here's a complete example showing the impact on a typical dataset:

```python
import pandas as pd
import dietpandas as dp

# Original approach
df_heavy = pd.read_csv("sales_2024.csv")
print(f"Memory: {df_heavy.memory_usage(deep=True).sum() / 1e6:.1f} MB")
# Memory: 2300.0 MB

# Diet Pandas approach
df_light = dp.read_csv("sales_2024.csv")
print(f"Memory: {df_light.memory_usage(deep=True).sum() / 1e6:.1f} MB")
# Memory: 750.0 MB
# 🥗 Diet Complete: Memory reduced by 67.4%

# Both are standard pandas DataFrames!
assert type(df_light) == pd.DataFrame
```

## Next Steps

- Learn more about [Basic Usage](../guide/basic-usage.md)
- Explore [File I/O Options](../guide/file-io.md)
- Check out [Advanced Optimization Techniques](../guide/advanced.md)
- See [API Reference](../api/core.md) for all available functions
