# File I/O

Learn how to load and save data efficiently with Diet Pandas.

## Reading Files

Diet Pandas provides drop-in replacements for pandas I/O functions that automatically optimize memory usage.

### CSV Files

The most common use case - CSV reading with Polars engine for speed:

```python
import dietpandas as dp

# Basic usage - 5-10x faster than pandas.read_csv
df = dp.read_csv("data.csv")

# Disable optimization if needed
df = dp.read_csv("data.csv", optimize=False)

# Aggressive mode for maximum compression
df = dp.read_csv("data.csv", aggressive=True)

# Pass through pandas arguments
df = dp.read_csv("data.csv", sep=";", encoding="utf-8")
```

### Parquet Files

Fast columnar format with built-in compression:

```python
import dietpandas as dp

df = dp.read_parquet("data.parquet")

# Still optimizes further!
# 🥗 Diet Complete: Memory reduced by 45.2%
```

### Excel Files

```python
import dietpandas as dp

# Read specific sheet
df = dp.read_excel("data.xlsx", sheet_name="Sales")

# Read all sheets
dfs = dp.read_excel("data.xlsx", sheet_name=None)
# Returns dict of optimized DataFrames
```

**Note:** Requires `openpyxl`:
```bash
pip install "diet-pandas[excel]"
```

### JSON Files

```python
import dietpandas as dp

# JSON Lines format (recommended for large files)
df = dp.read_json("data.jsonl", lines=True)

# Standard JSON
df = dp.read_json("data.json")
```

### HDF5 Files

Hierarchical data format for large datasets:

```python
import dietpandas as dp

df = dp.read_hdf("data.h5", key="dataset1")
```

**Note:** Requires `tables`:
```bash
pip install "diet-pandas[hdf]"
```

### Feather Files

Apache Arrow format - extremely fast:

```python
import dietpandas as dp

df = dp.read_feather("data.feather")
# Fastest format for pandas data!
```

## Writing Files

Save optimized DataFrames to disk:

### CSV

```python
import dietpandas as dp

dp.to_csv_optimized(df, "output.csv")

# Pass pandas arguments
dp.to_csv_optimized(df, "output.csv", index=False, sep="|")
```

### Parquet

```python
import dietpandas as dp

dp.to_parquet_optimized(df, "output.parquet")

# With compression
dp.to_parquet_optimized(df, "output.parquet", compression="gzip")
```

### Feather

```python
import dietpandas as dp

dp.to_feather_optimized(df, "output.feather")
```

## Performance Comparison

Loading a 500MB CSV file:

| Method | Time | Memory | Notes |
|--------|------|--------|-------|
| `pd.read_csv()` | 45s | 2.3 GB | Standard |
| `pd.read_csv()` + `diet()` | 47s | 750 MB | Manual opt |
| `dp.read_csv()` | 8s | 750 MB | **Best!** |

## Choosing a File Format

| Format | Speed | Compression | Use Case |
|--------|-------|-------------|----------|
| **CSV** | Medium | None | Human-readable, universal |
| **Parquet** | Fast | Excellent | Long-term storage |
| **Feather** | Very Fast | Good | Temporary storage |
| **Excel** | Slow | None | Business reports |
| **JSON** | Medium | None | Web APIs |
| **HDF5** | Fast | Good | Scientific data |

**Recommendations:**
- **Fast iteration:** Feather
- **Long-term storage:** Parquet
- **Sharing with non-Python users:** CSV
- **Large datasets:** Parquet or HDF5

## Working with Multiple Files

### Loading Multiple CSVs

```python
import dietpandas as dp
import glob

dfs = []
for filepath in glob.glob("data/*.csv"):
    df = dp.read_csv(filepath)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
```

### Batch Processing

```python
import dietpandas as dp

def process_file(filepath):
    df = dp.read_csv(filepath)
    # Process
    result = df.groupby('category')['sales'].sum()
    return result

results = [process_file(f) for f in file_list]
```

## Chunked Reading

For files too large to fit in memory:

```python
import pandas as pd
import dietpandas as dp

# Read in chunks
for chunk in pd.read_csv("huge_file.csv", chunksize=10000):
    # Optimize each chunk
    chunk = dp.diet(chunk)
    # Process
    process(chunk)
```

## URL and Cloud Storage

Diet Pandas works with URLs and cloud storage:

```python
import dietpandas as dp

# From URL
df = dp.read_csv("https://example.com/data.csv")

# From S3 (with s3fs)
df = dp.read_csv("s3://bucket/data.csv")

# From Google Cloud Storage (with gcsfs)
df = dp.read_parquet("gs://bucket/data.parquet")
```

## Compression

Reading compressed files:

```python
import dietpandas as dp

# Automatic detection
df = dp.read_csv("data.csv.gz")
df = dp.read_csv("data.csv.bz2")
df = dp.read_csv("data.csv.zip")

# Still optimized!
```

## Advanced Patterns

### Pipeline Pattern

```python
import dietpandas as dp

def load_and_prepare(filepath):
    return (
        dp.read_csv(filepath)
          .dropna()
          .query("age > 18")
          .reset_index(drop=True)
    )

df = load_and_prepare("users.csv")
```

### Caching Pattern

```python
import dietpandas as dp
import os

def load_with_cache(filepath, cache_path):
    if os.path.exists(cache_path):
        # Load from fast format
        return dp.read_feather(cache_path)
    else:
        # Load and cache
        df = dp.read_csv(filepath)
        dp.to_feather_optimized(df, cache_path)
        return df

df = load_with_cache("data.csv", "cache/data.feather")
```

## Troubleshooting

### Polars Engine Fails

If Polars engine fails, Diet Pandas automatically falls back to pandas:

```python
df = dp.read_csv("complex_file.csv")
# ⚠️ Warning: Polars engine failed, falling back to pandas
# This is automatic - no action needed
```

### Memory Issues

For very large files:

```python
# Load without optimization first
df = dp.read_csv("huge.csv", optimize=False)

# Drop unnecessary columns
df = df[['col1', 'col2', 'col3']]

# Then optimize
df = dp.diet(df)
```

### Encoding Issues

```python
df = dp.read_csv("data.csv", encoding="latin-1")
```

## Next Steps

- Learn about [Advanced Optimization](advanced.md)
- Check [Memory Reports](memory-reports.md)
- See [API Reference](../api/io.md)
