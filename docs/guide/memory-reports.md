# Memory Reports

Learn how to analyze and understand memory usage in your DataFrames.

## Basic Memory Report

Get a detailed breakdown of memory usage per column:

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({
    'id': range(10000),
    'name': ['User' + str(i) for i in range(10000)],
    'score': [95.5] * 10000
})

report = dp.get_memory_report(df)
print(report)
```

Output:
```
  column    dtype  memory_bytes  memory_mb  percent_of_total
0   name   object        590000      0.590              88.1
1     id    int64         80000      0.080              11.9
2  score  float64         80000      0.080              11.9
```

## Understanding the Report

### Columns

- **column**: Column name
- **dtype**: Data type
- **memory_bytes**: Raw bytes used
- **memory_mb**: Memory in megabytes
- **percent_of_total**: Percentage of total DataFrame memory

### Sorting

The report is automatically sorted by memory usage (highest first).

## Before/After Comparison

Compare memory usage before and after optimization:

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35] * 1000,
    'country': ['USA', 'UK', 'FR'] * 1000
})

# Before optimization
print("=" * 50)
print("BEFORE OPTIMIZATION")
print("=" * 50)
report_before = dp.get_memory_report(df)
print(report_before)
print(f"\nTotal: {report_before['memory_mb'].sum():.2f} MB")

# Optimize
df_optimized = dp.diet(df)

# After optimization
print("\n" + "=" * 50)
print("AFTER OPTIMIZATION")
print("=" * 50)
report_after = dp.get_memory_report(df_optimized)
print(report_after)
print(f"\nTotal: {report_after['memory_mb'].sum():.2f} MB")

# Calculate savings
total_before = report_before['memory_mb'].sum()
total_after = report_after['memory_mb'].sum()
savings_pct = (1 - total_after / total_before) * 100
print(f"\n💰 Savings: {savings_pct:.1f}% ({total_before - total_after:.2f} MB)")
```

## Identifying Memory Hogs

Find columns using the most memory:

```python
import dietpandas as dp

report = dp.get_memory_report(df)

# Top 5 memory users
print("Top 5 columns by memory usage:")
print(report.head(5)[['column', 'memory_mb', 'percent_of_total']])

# Columns using >10% of total memory
memory_hogs = report[report['percent_of_total'] > 10]
print(f"\nColumns using >10% of memory: {len(memory_hogs)}")
print(memory_hogs[['column', 'percent_of_total']])
```

## Type-Specific Analysis

Analyze memory by data type:

```python
import dietpandas as dp

report = dp.get_memory_report(df)

# Group by dtype
by_type = report.groupby('dtype')['memory_mb'].sum().sort_values(ascending=False)

print("Memory usage by data type:")
print(by_type)
print(f"\nTotal: {by_type.sum():.2f} MB")

# Calculate percentages
by_type_pct = (by_type / by_type.sum() * 100).round(1)
print("\nPercentage by type:")
print(by_type_pct)
```

Example output:
```
Memory usage by data type:
dtype
object      450.00
int64        80.00
float64      32.00
Total: 562.00 MB

Percentage by type:
dtype
object     80.1%
int64      14.2%
float64     5.7%
```

## Custom Reports

Create custom analysis:

### High Cardinality Report

Find object columns with many unique values:

```python
import dietpandas as dp
import pandas as pd

def high_cardinality_report(df):
    report = dp.get_memory_report(df)
    
    # Filter to object columns
    obj_cols = report[report['dtype'] == 'object']
    
    # Add cardinality info
    obj_cols = obj_cols.copy()
    obj_cols['unique_values'] = [df[col].nunique() for col in obj_cols['column']]
    obj_cols['unique_pct'] = [
        df[col].nunique() / len(df) * 100 
        for col in obj_cols['column']
    ]
    
    return obj_cols.sort_values('memory_mb', ascending=False)

report = high_cardinality_report(df)
print(report)
```

### Optimization Potential Report

Estimate potential savings:

```python
import dietpandas as dp
import numpy as np

def optimization_potential(df):
    report = dp.get_memory_report(df)
    
    estimates = []
    for _, row in report.iterrows():
        col = row['column']
        dtype = row['dtype']
        current_mb = row['memory_mb']
        
        # Estimate optimized size
        if dtype == 'int64':
            # Assume can reduce to int16 on average
            estimated_mb = current_mb * 2 / 8  # 2 bytes vs 8
        elif dtype == 'float64':
            # Assume float32
            estimated_mb = current_mb / 2
        elif dtype == 'object':
            # Assume 50% can become category
            if df[col].nunique() / len(df) < 0.5:
                estimated_mb = current_mb * 0.2  # 80% savings
            else:
                estimated_mb = current_mb
        else:
            estimated_mb = current_mb
        
        savings_mb = current_mb - estimated_mb
        savings_pct = (savings_mb / current_mb * 100) if current_mb > 0 else 0
        
        estimates.append({
            'column': col,
            'current_mb': current_mb,
            'estimated_mb': estimated_mb,
            'savings_mb': savings_mb,
            'savings_pct': savings_pct
        })
    
    return pd.DataFrame(estimates).sort_values('savings_mb', ascending=False)

potential = optimization_potential(df)
print(potential)
print(f"\nTotal potential savings: {potential['savings_mb'].sum():.2f} MB")
```

## Pandas Built-in Methods

Diet Pandas complements pandas' built-in memory tools:

### DataFrame.info()

```python
df.info(memory_usage='deep')
```

Shows memory per column but less detailed than `get_memory_report()`.

### DataFrame.memory_usage()

```python
# Total memory
total = df.memory_usage(deep=True).sum()
print(f"Total memory: {total / 1e6:.1f} MB")

# Per column
per_col = df.memory_usage(deep=True)
print(per_col)
```

### sys.getsizeof()

```python
import sys

# Less accurate for DataFrames
size = sys.getsizeof(df)
print(f"DataFrame object size: {size} bytes")
```

## Real-World Example

Complete analysis workflow:

```python
import dietpandas as dp
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("large_dataset.csv")

print("=" * 70)
print("INITIAL ANALYSIS")
print("=" * 70)

# Basic info
print(f"Shape: {df.shape}")
print(f"Total memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

# Detailed report
report_before = dp.get_memory_report(df)
print("Top 10 memory users:")
print(report_before.head(10))

# Type breakdown
print("\nMemory by type:")
print(report_before.groupby('dtype')['memory_mb'].sum().sort_values(ascending=False))

# High cardinality objects
obj_cols = report_before[report_before['dtype'] == 'object']
if len(obj_cols) > 0:
    print(f"\nObject columns: {len(obj_cols)}")
    print(f"Total object memory: {obj_cols['memory_mb'].sum():.2f} MB")

print("\n" + "=" * 70)
print("OPTIMIZING...")
print("=" * 70)

# Optimize
df_optimized = dp.diet(df, verbose=True)

# After report
report_after = dp.get_memory_report(df_optimized)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Compare totals
before_total = report_before['memory_mb'].sum()
after_total = report_after['memory_mb'].sum()
savings = before_total - after_total
savings_pct = (savings / before_total * 100)

print(f"Before:  {before_total:.2f} MB")
print(f"After:   {after_total:.2f} MB")
print(f"Savings: {savings:.2f} MB ({savings_pct:.1f}%)")

# Biggest wins
print("\nBiggest memory reductions:")
comparison = pd.DataFrame({
    'column': report_before['column'],
    'before_mb': report_before['memory_mb'].values,
    'after_mb': report_after['memory_mb'].values
})
comparison['savings_mb'] = comparison['before_mb'] - comparison['after_mb']
comparison['savings_pct'] = (comparison['savings_mb'] / comparison['before_mb'] * 100)

top_savings = comparison.sort_values('savings_mb', ascending=False).head(10)
print(top_savings[['column', 'before_mb', 'after_mb', 'savings_pct']])
```

## Tips for Memory Analysis

1. **Always use `deep=True`**: Essential for accurate object column measurement
2. **Check cardinality**: High unique counts in object columns = memory waste
3. **Look for patterns**: Repeated values -> category or sparse opportunities
4. **Prioritize largest columns**: 80/20 rule - optimize the biggest first
5. **Compare dtypes**: int64 and object columns usually have most potential

## Next Steps

- Learn about [Advanced Optimization](advanced.md)
- Review [Performance Benchmarks](../performance/benchmarks.md)
- See [API Reference](../api/core.md)
