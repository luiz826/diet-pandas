# API Reference: I/O Functions

This page documents all file input/output functions in Diet Pandas.

All I/O functions automatically optimize the loaded DataFrame and return a standard pandas DataFrame.

## Read Functions

### read_csv()

Read a CSV file with automatic memory optimization. Uses Polars engine for 5-10x faster parsing.

::: dietpandas.io.read_csv
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp

# Basic usage
df = dp.read_csv("data.csv")

# Disable optimization
df = dp.read_csv("data.csv", optimize=False)

# Aggressive mode
df = dp.read_csv("data.csv", aggressive=True)
```

---

### read_parquet()

Read a Parquet file with automatic memory optimization.

::: dietpandas.io.read_parquet
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp

df = dp.read_parquet("data.parquet")
```

---

### read_excel()

Read an Excel file with automatic memory optimization.

::: dietpandas.io.read_excel
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp

# Read specific sheet
df = dp.read_excel("data.xlsx", sheet_name="Sheet1")

# Read all sheets
dfs = dp.read_excel("data.xlsx", sheet_name=None)
```

---

### read_json()

Read a JSON file with automatic memory optimization.

::: dietpandas.io.read_json
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp

# Read JSON lines format
df = dp.read_json("data.jsonl", lines=True)

# Read standard JSON
df = dp.read_json("data.json")
```

---

### read_hdf()

Read an HDF5 file with automatic memory optimization.

::: dietpandas.io.read_hdf
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp

df = dp.read_hdf("data.h5", key="dataset1")
```

**Note:** Requires optional `tables` dependency:
```bash
pip install "diet-pandas[hdf]"
```

---

### read_feather()

Read a Feather file with automatic memory optimization.

::: dietpandas.io.read_feather
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp

df = dp.read_feather("data.feather")
```

---

## Write Functions

### to_csv_optimized()

Write a DataFrame to CSV with memory optimization.

::: dietpandas.io.to_csv_optimized
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({'col': range(1000)})
dp.to_csv_optimized(df, "output.csv")
```

---

### to_parquet_optimized()

Write a DataFrame to Parquet with memory optimization.

::: dietpandas.io.to_parquet_optimized
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({'col': range(1000)})
dp.to_parquet_optimized(df, "output.parquet")
```

---

### to_feather_optimized()

Write a DataFrame to Feather format with memory optimization.

::: dietpandas.io.to_feather_optimized
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import dietpandas as dp
import pandas as pd

df = pd.DataFrame({'col': range(1000)})
dp.to_feather_optimized(df, "output.feather")
```

---

## Supported File Formats

| Format | Read Function | Write Function | Optional Dependency |
|--------|--------------|----------------|---------------------|
| CSV | `read_csv()` | `to_csv_optimized()` | None (built-in) |
| Parquet | `read_parquet()` | `to_parquet_optimized()` | `pyarrow` |
| Excel | `read_excel()` | N/A | `openpyxl` |
| JSON | `read_json()` | N/A | None (built-in) |
| HDF5 | `read_hdf()` | N/A | `tables` |
| Feather | `read_feather()` | `to_feather_optimized()` | `pyarrow` |

## Performance Comparison

### CSV Reading Performance

```python
import time
import pandas as pd
import dietpandas as dp

# Standard pandas
start = time.time()
df_pandas = pd.read_csv("large_file.csv")
pandas_time = time.time() - start

# Diet pandas
start = time.time()
df_diet = dp.read_csv("large_file.csv")
diet_time = time.time() - start

print(f"Pandas: {pandas_time:.2f}s, Memory: {df_pandas.memory_usage().sum() / 1e6:.1f} MB")
print(f"Diet:   {diet_time:.2f}s, Memory: {df_diet.memory_usage().sum() / 1e6:.1f} MB")
# Pandas: 45.2s, Memory: 2300.0 MB
# Diet:   8.7s, Memory: 750.0 MB
```

## Common Parameters

Most read functions support these common parameters:

- **`optimize`** (bool, default=True): Whether to optimize memory usage
- **`aggressive`** (bool, default=False): Use aggressive optimization mode
- **`**kwargs`**: Additional parameters passed to underlying pandas function

## See Also

- [Core Functions API](core.md)
- [File I/O Guide](../guide/file-io.md)
- [Performance Benchmarks](../performance/benchmarks.md)
