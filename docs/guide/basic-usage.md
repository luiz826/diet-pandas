# Basic Usage

Learn how to use Diet Pandas effectively for everyday data science tasks.

## The `diet()` Function

The core function of Diet Pandas is `diet()`, which optimizes a DataFrame's memory usage.

### Default Behavior (Safe Mode)

By default, `diet()` performs lossless optimization:

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'id': range(1000),              # int64
    'age': [25, 30, 35] * 333 + [25],  # int64
    'score': [95.5, 87.3, 92.1] * 333 + [95.5],  # float64
    'city': ['NYC', 'LA', 'SF'] * 333 + ['NYC']  # object
})

print("Before:")
print(df.memory_usage(deep=True))
# Index        132 bytes
# id          8000 bytes
# age         8000 bytes
# score       8000 bytes
# city       60000 bytes

df_optimized = dp.diet(df)

print("\nAfter:")
print(df_optimized.memory_usage(deep=True))
# Index        132 bytes
# id          2000 bytes  (uint16)
# age         1000 bytes  (uint8)
# score       4000 bytes  (float32)
# city         1200 bytes (category)
```

### Understanding the Output

When optimization completes, you'll see:

```
🥗 Diet Complete: Memory reduced by 67.3%
   84.13 KB -> 27.49 KB
```

### In-Place Optimization

To modify the DataFrame directly without creating a copy:

```python
import dietpandas as dp

dp.diet(df, inplace=True)
# df is now optimized
```

This is useful when working with large DataFrames where you don't want to duplicate memory.

### Silent Mode

To suppress output messages:

```python
df = dp.diet(df, verbose=False)
```

## Optimization Modes

### Safe Mode (Default)

Preserves precision for most use cases:

```python
df = dp.diet(df, aggressive=False)  # Default
# float64 -> float32 (7 decimal digits precision)
```

### Aggressive Mode (Keto Diet)

Maximum compression with some precision loss:

```python
df = dp.diet(df, aggressive=True)
# float64 -> float16 (3 decimal digits precision)
# Use for: visualization, approximate calculations
# Avoid for: financial calculations, scientific computing
```

## Customizing Optimization

### Categorical Threshold

Control when strings are converted to categories:

```python
# Convert to category if <30% unique values (stricter)
df = dp.diet(df, categorical_threshold=0.3)

# Convert to category if <70% unique values (more aggressive)
df = dp.diet(df, categorical_threshold=0.7)
```

**Rule of thumb:**
- Low threshold (0.3): Only very repetitive data becomes categorical
- High threshold (0.7): More columns become categorical
- Default (0.5): Balanced approach

### DateTime Optimization

Enable datetime string detection and conversion:

```python
df = dp.diet(df, optimize_datetimes=True)
```

This automatically detects object columns with datetime strings and converts them to `datetime64`.

### Sparse Data Optimization

For data with many repeated values:

```python
df = dp.diet(df, optimize_sparse_cols=True)
```

This converts columns where >90% of values are the same to sparse format.

**Best for:**
- Binary features (0/1)
- Indicator variables
- Data with many zeros or NaNs
- One-hot encoded features

## Common Patterns

### Pattern 1: Load and Optimize

```python
import pandas as pd
import dietpandas as dp

# Load with standard pandas
df = pd.read_csv("data.csv")

# Clean and transform
df = df.dropna()
df['new_col'] = df['col1'] + df['col2']

# Optimize before analysis
df = dp.diet(df)

# Now analyze with less memory
print(df.describe())
```

### Pattern 2: Optimize in Pipeline

```python
import dietpandas as dp

def load_and_clean(filepath):
    df = dp.read_csv(filepath)  # Already optimized
    df = df.dropna()
    df = df[df['age'] > 18]
    return df

df = load_and_clean("users.csv")
```

### Pattern 3: Selective Optimization

```python
import dietpandas as dp

# Don't optimize high-cardinality ID columns
df_ids = df[['user_id', 'transaction_id']]
df_data = df.drop(['user_id', 'transaction_id'], axis=1)

# Optimize only the data columns
df_data = dp.diet(df_data)

# Recombine
df = pd.concat([df_ids, df_data], axis=1)
```

### Pattern 4: Iterative Optimization

```python
import dietpandas as dp

# Load data
df = pd.read_csv("large_file.csv")

# Process in chunks
for i in range(0, len(df), 10000):
    chunk = df.iloc[i:i+10000]
    chunk = dp.diet(chunk)
    # Process chunk
    process(chunk)
```

## Memory Reports

### Basic Report

```python
import dietpandas as dp

report = dp.get_memory_report(df)
print(report)
```

Output:
```
        column      dtype  memory_bytes  memory_mb  percent_of_total
0  description     object     450000000     450.00              67.3
1     user_id      int64      32000000      32.00              4.8
2   timestamp  datetime64      32000000      32.00              4.8
```

### Comparing Before/After

```python
import dietpandas as dp

# Before optimization
report_before = dp.get_memory_report(df)
print("Before:")
print(report_before)

# Optimize
df = dp.diet(df)

# After optimization
report_after = dp.get_memory_report(df)
print("\nAfter:")
print(report_after)
```

## Data Preservation

Diet Pandas preserves your data:

```python
import pandas as pd
import dietpandas as dp

# Original data
df = pd.DataFrame({'values': [1.1, 2.2, 3.3]})

# Optimize
df_opt = dp.diet(df)

# Data is preserved (within float32 precision)
assert df['values'].sum() == df_opt['values'].sum()
assert df['values'].mean() == df_opt['values'].mean()
```

## When NOT to Use Optimization

**Avoid optimization for:**

1. **ID Columns**: High-cardinality strings or large integers
2. **Precise Calculations**: Financial data requiring exact decimal precision
3. **Small DataFrames**: Optimization overhead not worth it (<1MB)
4. **Streaming Data**: Optimize in batches instead

## Next Steps

- Learn about [File I/O](file-io.md)
- Explore [Advanced Optimization](advanced.md)
- Check [Memory Reports](memory-reports.md)
