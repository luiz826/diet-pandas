# API Reference: Core Functions

This page documents all core optimization functions in Diet Pandas.

## Main Functions

### diet()

Optimize a pandas DataFrame by downcasting data types to reduce memory usage.

**NEW in v0.5.0:** Supports parallel processing with `parallel` and `max_workers` parameters for 2-4x speedup.

::: dietpandas.core.diet
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({'col': [1, 2, 3]})

# Standard optimization
df_optimized = dp.diet(df)

# Parallel processing (default, 2-4x faster)
df_optimized = dp.diet(df, parallel=True)

# Control number of threads
df_optimized = dp.diet(df, parallel=True, max_workers=4)

# Sequential processing
df_optimized = dp.diet(df, parallel=False)
```

---

### optimize_int()

Optimize integer columns to the smallest safe integer type.

::: dietpandas.core.optimize_int
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
from dietpandas import optimize_int

s = pd.Series([1, 2, 3, 4, 5])  # int64
s_optimized = optimize_int(s)    # uint8
```

---

### optimize_float()

Optimize float columns to smaller precision when safe.

::: dietpandas.core.optimize_float
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
from dietpandas import optimize_float

s = pd.Series([1.1, 2.2, 3.3])  # float64
s_optimized = optimize_float(s)  # float32
```

---

### optimize_obj()

Optimize object columns by converting low-cardinality strings to category type.

::: dietpandas.core.optimize_obj
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
from dietpandas import optimize_obj

s = pd.Series(['A', 'B', 'A', 'B', 'C'] * 100)  # object
s_optimized = optimize_obj(s)                     # category
```

---

### optimize_datetime()

Optimize datetime columns for better memory efficiency.

::: dietpandas.core.optimize_datetime
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
from dietpandas import optimize_datetime

# Object column with datetime strings
s = pd.Series(['2020-01-01', '2020-02-01', '2020-03-01'])
s_optimized = optimize_datetime(s)  # datetime64[ns]
```

---

### optimize_sparse()

Convert columns with many repeated values to sparse format.

::: dietpandas.core.optimize_sparse
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
from dietpandas import optimize_sparse

# Column with 95% zeros
s = pd.Series([0] * 950 + [1] * 50)
s_optimized = optimize_sparse(s)  # Sparse[int8, 0]
```

---

### get_memory_report()

Generate a detailed memory usage report for a DataFrame.

::: dietpandas.core.get_memory_report
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({
    'a': range(1000),
    'b': ['text'] * 1000
})

report = dp.get_memory_report(df)
print(report)
#   column    dtype  memory_bytes  memory_mb  percent_of_total
# 0      b   object         59000      0.059              88.1
# 1      a    int64          8000      0.008              11.9
```

---

## Type Optimization Rules

### Integer Optimization

| Value Range | Optimized Type | Bytes Saved per Value |
|-------------|----------------|----------------------|
| 0 to 255 | `uint8` | 7 bytes (from int64) |
| 0 to 65,535 | `uint16` | 6 bytes |
| -128 to 127 | `int8` | 7 bytes |
| -32,768 to 32,767 | `int16` | 6 bytes |

### Float Optimization

| Mode | Conversion | Precision | Use Case |
|------|------------|-----------|----------|
| Safe | float64 → float32 | ~7 decimal digits | Most ML tasks |
| Aggressive | float64 → float16 | ~3 decimal digits | Extreme compression |

### Object Optimization

| Condition | Optimization | Memory Savings |
|-----------|--------------|----------------|
| Unique ratio < 50% | object → category | 50-90% typical |
| All datetime strings | object → datetime64 | 50-70% typical |
| High cardinality | No change | Keep as object |

### Sparse Optimization

| Condition | Optimization | Memory Savings |
|-----------|--------------|----------------|
| >90% repeated values | Dense → Sparse | 90-99% typical |
| Binary features (0/1) | Dense → Sparse[int8] | ~96% typical |
| Low sparsity | No change | Keep as dense |

## See Also

- [I/O Functions API](io.md)
- [Basic Usage Guide](../guide/basic-usage.md)
- [Advanced Optimization](../guide/advanced.md)
