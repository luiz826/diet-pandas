# API Reference: Analysis Functions

This page documents the analysis and inspection functions in Diet Pandas.

## Analysis Functions

### analyze()

Analyze a DataFrame and return optimization recommendations without modifying it.

::: dietpandas.analysis.analyze
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'id': range(1000),
    'amount': [1.1, 2.2, 3.3] * 333 + [1.1],
    'category': ['A', 'B', 'C'] * 333 + ['A']
})

# Get detailed analysis
analysis_df = dp.analyze(df)
print(analysis_df)
#      column current_dtype recommended_dtype  current_memory_mb  optimized_memory_mb  savings_mb  savings_percent                  reasoning
# 0        id         int64             uint16               0.008                0.002       0.006            75.0    Integer range fits in uint16
# 1    amount       float64            float32               0.008                0.004       0.004            50.0      Standard float optimization
# 2  category        object           category               0.057                0.001       0.056            98.2  Low cardinality (3 unique values)
```

---

### get_optimization_summary()

Get summary statistics from an analysis DataFrame.

::: dietpandas.analysis.get_optimization_summary
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'id': range(1000),
    'value': [1.5, 2.5, 3.5] * 333 + [1.5]
})

analysis = dp.analyze(df)
summary = dp.get_optimization_summary(analysis)

print(summary)
# {
#     'total_columns': 2,
#     'optimizable_columns': 2,
#     'current_memory_mb': 0.016,
#     'optimized_memory_mb': 0.006,
#     'total_savings_mb': 0.010,
#     'total_savings_percent': 62.5
# }
```

---

### estimate_memory_reduction()

Quickly estimate potential memory reduction percentage.

::: dietpandas.analysis.estimate_memory_reduction
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'int_col': [1, 2, 3, 4, 5] * 200,
    'float_col': [1.1, 2.2, 3.3, 4.4, 5.5] * 200,
    'str_col': ['A', 'B', 'C', 'A', 'B'] * 200
})

# Quick estimate without detailed analysis
reduction = dp.estimate_memory_reduction(df)
print(f"Estimated reduction: {reduction:.1f}%")
# Estimated reduction: 78.3%

# Compare with full analysis
analysis = dp.analyze(df)
summary = dp.get_optimization_summary(analysis)
print(f"Actual reduction: {summary['total_savings_percent']:.1f}%")
```

## Workflow Example

### Analyze Before Optimizing

```python
import pandas as pd
import dietpandas as dp

# Load your data
df = pd.read_csv("data.csv")

# 1. Quick estimate
print(f"Expected reduction: {dp.estimate_memory_reduction(df):.1f}%")

# 2. Detailed analysis
analysis = dp.analyze(df)
print(analysis)

# 3. Review summary
summary = dp.get_optimization_summary(analysis)
print(f"Total savings: {summary['total_savings_mb']:.2f} MB")
print(f"Reduction: {summary['total_savings_percent']:.1f}%")

# 4. Apply optimization
df_optimized = dp.diet(df)
```

### Aggressive Mode Analysis

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'metric': [1.123456789] * 1000
})

# Compare normal vs aggressive mode
normal_analysis = dp.analyze(df, aggressive=False)
aggressive_analysis = dp.analyze(df, aggressive=True)

print("Normal mode:")
print(normal_analysis)
# float64 -> float32 (50% reduction)

print("\nAggressive mode:")
print(aggressive_analysis)
# float64 -> float16 (75% reduction, but possible precision loss)
```

## See Also

- [Core Functions](core.md) - Main optimization functions
- [I/O Functions](io.md) - File reading with automatic optimization
- [Exceptions](exceptions.md) - Custom warnings and exceptions
