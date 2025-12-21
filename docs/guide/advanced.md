# Advanced Optimization

Advanced techniques for maximizing memory efficiency with Diet Pandas.

## Understanding the Optimization Process

Diet Pandas performs optimization in this order:

1. **Integer Optimization**: Find smallest integer type for each column
2. **Float Optimization**: Downcast to float32 (or float16 in aggressive mode)
3. **Object Optimization**: Convert low-cardinality strings to categories
4. **DateTime Optimization**: Convert datetime strings to datetime64
5. **Sparse Optimization**: Convert highly repetitive columns to sparse

## Sparse Data Optimization

### When to Use Sparse

Sparse optimization is perfect for:

- **Binary features** (0/1, True/False)
- **One-hot encoded columns**
- **Indicator variables**
- **Data with many NaNs or zeros**

```python
import dietpandas as dp
import pandas as pd
import numpy as np

# Create sparse data (95% zeros)
df = pd.DataFrame({
    'feature_1': [0] * 950 + [1] * 50,
    'feature_2': [0] * 900 + [1] * 100,
    'value': np.random.randn(1000)
})

print(f"Before: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
# Before: 23.5 KB

# Enable sparse optimization
df = dp.diet(df, optimize_sparse_cols=True)

print(f"After: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
# After: 1.2 KB
# 🥗 Diet Complete: Memory reduced by 94.9%
```

### Sparse Threshold

Control when sparse optimization triggers:

```python
# Default: optimize if >90% of values are the same
df = dp.diet(df, optimize_sparse_cols=True, sparse_threshold=0.9)

# More aggressive: optimize if >80% of values are the same
df = dp.diet(df, optimize_sparse_cols=True, sparse_threshold=0.8)

# Very strict: only if >95% of values are the same
df = dp.diet(df, optimize_sparse_cols=True, sparse_threshold=0.95)
```

### Working with Sparse DataFrames

Sparse DataFrames work just like regular DataFrames:

```python
# All pandas operations work
result = df['sparse_col'].sum()
mean = df['sparse_col'].mean()
filtered = df[df['sparse_col'] > 0]

# Convert back to dense if needed
df['sparse_col'] = df['sparse_col'].sparse.to_dense()
```

## DateTime Optimization

### Automatic Detection

Diet Pandas can detect and convert datetime strings:

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'date_str': ['2020-01-01', '2020-01-02', '2020-01-03'] * 1000,
    'value': range(3000)
})

print(df.dtypes)
# date_str    object
# value        int64

df = dp.diet(df, optimize_datetimes=True)

print(df.dtypes)
# date_str    datetime64[ns]
# value            uint16
```

### Benefits of DateTime Optimization

1. **Memory savings**: 50-70% reduction vs object
2. **Type safety**: Proper datetime operations
3. **Performance**: Faster filtering and grouping
4. **Functionality**: Access to `.dt` accessor

```python
# Now you can use datetime operations
df['year'] = df['date_str'].dt.year
df['month'] = df['date_str'].dt.month

# Fast filtering
recent = df[df['date_str'] > '2020-06-01']
```

## Aggressive Mode Deep Dive

### Precision Trade-offs

Understanding float16 limitations:

```python
import numpy as np

# float64: ~15 decimal digits precision
# float32: ~7 decimal digits precision  
# float16: ~3 decimal digits precision

# Example
value = 1.23456789

# Safe mode (float32)
float32_value = np.float32(value)
print(float32_value)  # 1.234568 ✓ Good enough for ML

# Aggressive mode (float16)
float16_value = np.float16(value)
print(float16_value)  # 1.235 ⚠️ Some loss
```

### When to Use Aggressive Mode

**✅ Good for:**
- Data visualization
- Exploratory analysis
- Approximate calculations
- Image processing
- Audio/video processing

**❌ Avoid for:**
- Financial calculations
- Scientific computing requiring precision
- Cumulative operations (errors compound)
- Legal/regulatory requirements

### Selective Aggressive Mode

Apply aggressive mode only to specific columns:

```python
import dietpandas as dp

# Optimize most columns safely
df_safe = dp.diet(df[['important_col1', 'important_col2']], 
                  aggressive=False)

# Aggressive for visualization columns
df_viz = dp.diet(df[['display_col1', 'display_col2']], 
                 aggressive=True)

# Combine
df = pd.concat([df_safe, df_viz], axis=1)
```

## Custom Categorical Thresholds

### Understanding Cardinality

The categorical threshold determines when to convert strings:

```python
# Default: convert if unique ratio < 50%
df = dp.diet(df, categorical_threshold=0.5)

# Example:
# - 1000 rows, 400 unique values -> 40% unique -> category ✓
# - 1000 rows, 600 unique values -> 60% unique -> keep object ✗
```

### Optimal Thresholds by Use Case

```python
# Conservative (only very repetitive)
df = dp.diet(df, categorical_threshold=0.3)
# Good for: Mixed data, unsure about cardinality

# Balanced (default)
df = dp.diet(df, categorical_threshold=0.5)
# Good for: Most use cases

# Aggressive (more conversions)
df = dp.diet(df, categorical_threshold=0.7)
# Good for: Data with lots of repeated strings
```

### Memory Impact Example

```python
import pandas as pd
import dietpandas as dp

# 10K rows with 1000 unique cities
df = pd.DataFrame({
    'city': np.random.choice(['NYC', 'LA', 'SF'] * 333, 10000)
})

# Cardinality: 3/10000 = 0.03% unique
print(f"Object: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
# Object: 78.2 KB

df_cat = dp.diet(df)
print(f"Category: {df_cat.memory_usage(deep=True).sum() / 1024:.1f} KB")
# Category: 10.5 KB
# 🥗 Diet Complete: Memory reduced by 86.6%
```

## Combining Optimization Strategies

### Full Optimization

Enable all optimization features:

```python
df = dp.diet(
    df,
    aggressive=True,              # Maximum float compression
    categorical_threshold=0.7,     # Aggressive categorization
    optimize_datetimes=True,       # Convert datetime strings
    optimize_sparse_cols=True,     # Sparse arrays
    sparse_threshold=0.85,         # Lower sparse threshold
    verbose=True                   # Show results
)
```

### Selective Optimization

Optimize different parts differently:

```python
import dietpandas as dp

# Identify column types
numeric_cols = df.select_dtypes(include=['number']).columns
string_cols = df.select_dtypes(include=['object']).columns
date_cols = ['date_column', 'timestamp']

# Optimize numeric (aggressive)
df[numeric_cols] = dp.diet(
    df[numeric_cols], 
    aggressive=True, 
    verbose=False
)

# Optimize strings (conservative)
df[string_cols] = dp.diet(
    df[string_cols], 
    categorical_threshold=0.3, 
    verbose=False
)

# Optimize dates
df[date_cols] = dp.diet(
    df[date_cols], 
    optimize_datetimes=True, 
    verbose=False
)
```

## Pre-Processing for Better Optimization

### Clean Before Optimizing

```python
import dietpandas as dp

# Remove outliers first
df = df[df['value'] < 1000]

# Now optimize with better range
df = dp.diet(df)
# Values now fit in int16 instead of int32!
```

### Handle Missing Data

```python
# Option 1: Drop missing
df = df.dropna()
df = dp.diet(df)

# Option 2: Fill missing
df['col'].fillna(0, inplace=True)
df = dp.diet(df, optimize_sparse_cols=True)
# Zeros might trigger sparse optimization!
```

## Monitoring Optimization

### Detailed Memory Reports

```python
import dietpandas as dp

# Before
report_before = dp.get_memory_report(df)
print("BEFORE:")
print(report_before.to_string())

# Optimize
df = dp.diet(df)

# After
report_after = dp.get_memory_report(df)
print("\nAFTER:")
print(report_after.to_string())

# Calculate savings per column
report_before['savings_mb'] = report_before['memory_mb'] - report_after['memory_mb']
print("\nSAVINGS PER COLUMN:")
print(report_before[['column', 'savings_mb']].sort_values('savings_mb', ascending=False))
```

## Performance Optimization

### Optimize Once, Reuse

```python
# Bad: Optimize repeatedly
for i in range(100):
    df = pd.read_csv("data.csv")
    df = dp.diet(df)  # Slow!
    process(df)

# Good: Optimize once, cache
df = dp.read_csv("data.csv")
dp.to_feather_optimized(df, "cache.feather")

for i in range(100):
    df = dp.read_feather("cache.feather")  # Fast!
    process(df)
```

### Parallel Processing

```python
from multiprocessing import Pool
import dietpandas as dp

def process_file(filepath):
    df = dp.read_csv(filepath)
    return df.groupby('category')['value'].sum()

with Pool(4) as pool:
    results = pool.map(process_file, file_list)
```

## Next Steps

- Review [Memory Reports Guide](memory-reports.md)
- Check [Performance Benchmarks](../performance/benchmarks.md)
- See [API Reference](../api/core.md)
