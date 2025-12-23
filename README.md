# Diet Pandas 🐼🥗

**Tagline:** *Same Pandas taste, half the calories (RAM).*

[![PyPI version](https://badge.fury.io/py/diet-pandas.svg)](https://pypi.org/project/diet-pandas/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://luiz826.github.io/diet-pandas/)

## 🎯 The Problem

Pandas is built for safety and ease of use, not memory efficiency. When you load a CSV, standard Pandas defaults to "safe" but wasteful data types:

* **`int64`** for small integers (wasting 75%+ memory per number)
* **`float64`** for simple metrics (wasting 50% memory per number)
* **`object`** for repetitive strings (wasting massive amounts of memory and CPU)

**Diet Pandas** solves this by acting as a strict nutritionist for your data. It aggressively analyzes data distributions and "downcasts" types to the smallest safe representation—often reducing memory usage by **50% to 80%** without losing information.

## 🚀 Quick Start

### Installation

```bash
pip install diet-pandas
```

### Basic Usage

```python
import dietpandas as dp

# 1. Drop-in replacement for pandas.read_csv
# Loads faster and uses less RAM automatically
df = dp.read_csv("huge_dataset.csv")
# Diet Complete: Memory reduced by 67.3%
#    450.00MB -> 147.15MB

# 2. Or optimize an existing DataFrame
import pandas as pd
df_heavy = pd.DataFrame({
    'year': [2020, 2021, 2022], 
    'revenue': [1.1, 2.2, 3.3]
})

print(df_heavy.info())
# year       int64   (8 bytes each)
# revenue    float64 (8 bytes each)

df_light = dp.diet(df_heavy)
# Diet Complete: Memory reduced by 62.5%
#    0.13MB -> 0.05MB

print(df_light.info())
# year       uint16  (2 bytes each)
# revenue    float32 (4 bytes each)
```

## ✨ Features

### 🏃 Fast Loading with Polars Engine

Diet Pandas uses [Polars](https://www.pola.rs/) (a blazing-fast DataFrame library) to parse CSV files, then automatically converts to optimized Pandas DataFrames.

```python
import dietpandas as dp

# 5-10x faster than pandas.read_csv AND uses less memory
df = dp.read_csv("large_file.csv")
```

### 🎯 Intelligent Type Optimization

```python
import dietpandas as dp

# Automatic optimization
df = dp.diet(df_original)

# See detailed memory report
report = dp.get_memory_report(df)
print(report)
#         column    dtype  memory_bytes  memory_mb  percent_of_total
# 0  large_text  category      12589875      12.59              45.2
# 1     user_id     uint32       4000000       4.00              14.4
```

### 🔥 Aggressive Mode (Keto Diet)

For maximum compression, use aggressive mode:

```python
# Safe mode: float64 -> float32 (lossless for most ML tasks)
df = dp.diet(df, aggressive=False)

# Keto mode: float64 -> float16 (extreme compression, some precision loss)
df = dp.diet(df, aggressive=True)
# Diet Complete: Memory reduced by 81.2%
```

### 📊 Multiple File Format Support

```python
import dietpandas as dp

# CSV with fast Polars engine
df = dp.read_csv("data.csv")

# Parquet
df = dp.read_parquet("data.parquet")

# Excel
df = dp.read_excel("data.xlsx")

# JSON
df = dp.read_json("data.json")

# HDF5
df = dp.read_hdf("data.h5", key="dataset1")

# Feather
df = dp.read_feather("data.feather")

# All readers automatically optimize memory usage!
```

### 🗜️ Sparse Data Optimization

For data with many repeated values (zeros, NaNs, or any repeated value):

```python
# Enable sparse optimization for columns with >90% repeated values
df = dp.diet(df, optimize_sparse_cols=True)
# Perfect for: binary features, indicator variables, sparse matrices
```

### 📅 DateTime Optimization

Automatically optimizes datetime columns for better memory efficiency:

```python
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=1000000),
    'value': range(1000000)
})

df_optimized = dp.diet(df, optimize_datetimes=True)
# DateTime columns automatically optimized
```

### ✓ Boolean Optimization

Automatically detects and optimizes boolean-like columns:

```python
df = pd.DataFrame({
    'is_active': [0, 1, 1, 0, 1],           # int64 -> boolean (87.5% memory reduction)
    'has_data': ['yes', 'no', 'yes', 'no', 'yes'],  # object -> boolean
    'approved': ['True', 'False', 'True', 'False', 'True']  # object -> boolean
})

df_optimized = dp.diet(df, optimize_bools=True)
# All three columns converted to memory-efficient boolean type!
```

Supports multiple boolean representations:
- **Numeric**: `0`, `1`
- **Strings**: `'true'`/`'false'`, `'yes'`/`'no'`, `'y'`/`'n'`, `'t'`/`'f'`
- Case-insensitive detection

### 🎛️ Column-Specific Control

**NEW in v0.3.0!** Fine-grained control over optimization:

```python
# Skip specific columns (e.g., IDs, UUIDs)
df = dp.diet(df, skip_columns=['user_id', 'uuid'])

# Force categorical conversion on high-cardinality columns
df = dp.diet(df, force_categorical=['country_code', 'product_sku'])

# Use aggressive mode only for specific columns
df = dp.diet(df, force_aggressive=['approximation_field', 'estimated_value'])

# Combine multiple controls
df = dp.diet(
    df,
    skip_columns=['id'],
    force_categorical=['category'],
    force_aggressive=['approx_price']
)
```

### 🔍 Pre-Flight Analysis

**NEW in v0.3.0!** Analyze your DataFrame before optimization to see what changes will be made:

```python
import pandas as pd
import dietpandas as dp

df = pd.DataFrame({
    'id': range(1000),
    'amount': [1.1, 2.2, 3.3] * 333 + [1.1],
    'category': ['A', 'B', 'C'] * 333 + ['A']
})

# Analyze without modifying the DataFrame
analysis = dp.analyze(df)
print(analysis)
#
#      column current_dtype recommended_dtype  current_memory_mb  optimized_memory_mb  savings_mb  savings_percent                  reasoning
# 0        id         int64             uint16               0.008                0.002       0.006            75.0    Integer range 0-999 fits in uint16
# 1    amount       float64            float32               0.008                0.004       0.004            50.0      Standard float optimization
# 2  category        object           category               0.057                0.001       0.056            98.2  Low cardinality (3 unique values)

# Get summary statistics
summary = dp.get_optimization_summary(analysis)
print(summary)
# {
#     'total_columns': 3,
#     'optimizable_columns': 3,
#     'current_memory_mb': 0.073,
#     'optimized_memory_mb': 0.007,
#     'total_savings_mb': 0.066,
#     'total_savings_percent': 90.4
# }

# Quick estimate without detailed analysis
reduction_pct = dp.estimate_memory_reduction(df)
print(f"Estimated reduction: {reduction_pct:.1f}%")
# Estimated reduction: 90.4%
```

### ⚠️ Smart Warnings

**NEW in v0.3.0!** Get helpful warnings about potential issues:

```python
import dietpandas as dp

df = pd.DataFrame({
    'id': range(10000),  # High cardinality
    'value': [1.123456789] * 10000,  # Will lose precision in float16
    'empty': [None] * 10000  # All NaN column
})

# Warnings are enabled by default
df_optimized = dp.diet(df, aggressive=True, warn_on_issues=True)
# ⚠️  Warning: Column 'empty' is entirely NaN - consider dropping it
# ⚠️  Warning: Column 'id' has high cardinality (100.0%) - may not benefit from categorical
# ⚠️  Warning: Aggressive mode on column 'value' may lose precision (float64 -> float16)

# Disable warnings if you know what you're doing
df_optimized = dp.diet(df, aggressive=True, warn_on_issues=False)
```

```python
import dietpandas as dp

# CSV (with Polars acceleration)
df = dp.read_csv("data.csv")

# Parquet (with Polars acceleration)
df = dp.read_parquet("data.parquet")

# Excel
df = dp.read_excel("data.xlsx")

# All return optimized Pandas DataFrames
```

## 🧪 Technical Details

### How It Works

Diet Pandas uses a **"Trojan Horse"** architecture:

1. **Ingestion Layer (The Fast Lane):**
   - Uses **Polars** or **PyArrow** for multi-threaded CSV parsing (5-10x faster)

2. **Optimization Layer (The Metabolism):**
   - Calculates min/max for numeric columns
   - Analyzes string cardinality (unique values ratio)
   - Maps stats to smallest safe numpy types

3. **Conversion Layer (The Result):**
   - Returns a standard `pandas.DataFrame` (100% compatible)
   - Works seamlessly with Scikit-Learn, PyTorch, XGBoost, Matplotlib

### Optimization Rules

| Original Type | Optimization | Example |
|--------------|--------------|---------|
| `int64` with only 0/1 | `boolean` | **NEW!** Flags, indicators (87.5% reduction) |
| `object` with 'yes'/'no' | `boolean` | **NEW!** Survey responses |
| `int64` with values 0-255 | `uint8` | User ages, small counts |
| `int64` with values -100 to 100 | `int8` | Temperature data |
| `float64` | `float32` | Most ML features |
| `object` with <50% unique | `category` | Country names, product categories |

## 📈 Real-World Performance

```python
import pandas as pd
import dietpandas as dp

# Standard Pandas
df = pd.read_csv("sales_data.csv")  # 2.3 GB, 45 seconds
print(df.memory_usage(deep=True).sum() / 1e9)  # 2.3 GB

# Diet Pandas
df = dp.read_csv("sales_data.csv")  # 0.8 GB, 8 seconds
print(df.memory_usage(deep=True).sum() / 1e9)  # 0.8 GB
# Diet Complete: Memory reduced by 65.2%
#    2300.00MB -> 800.00MB
```

## 🎛️ Advanced Usage

### Column-Specific Control **NEW!**

```python
# Skip optimization for specific columns
df = dp.diet(df, skip_columns=['user_id', 'uuid'])

# Force categorical conversion for high-cardinality columns
df = dp.diet(df, force_categorical=['country_code'])

# Apply aggressive optimization only to specific columns
df = dp.diet(df, force_aggressive=['estimated_value'])
```

### Custom Categorical Threshold

```python
# Convert to category if <30% unique values (default is 50%)
df = dp.diet(df, categorical_threshold=0.3)
```

### Disable Boolean Optimization

```python
# Keep binary columns as integers instead of converting to boolean
df = dp.diet(df, optimize_bools=False)
```

### In-Place Optimization

```python
# Modify DataFrame in place (saves memory)
dp.diet(df, inplace=True)
```

### Disable Optimization for Specific Columns

```python
import pandas as pd
import dietpandas as dp

df = dp.read_csv("data.csv", optimize=False)  # Load without optimization
df = df.drop(columns=['id_column'])  # Remove high-cardinality columns
df = dp.diet(df)  # Now optimize
```

### Verbose Mode

```python
df = dp.diet(df, verbose=True)
# Diet Complete: Memory reduced by 67.3%
#    450.00MB -> 147.15MB
```

## 🧩 Integration with Data Science Stack

Diet Pandas returns standard Pandas DataFrames, so it works seamlessly with:

```python
import dietpandas as dp
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load optimized data
df = dp.read_csv("train.csv")

# Works with Scikit-Learn
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier()
model.fit(X, y)

# Works with Matplotlib
df['revenue'].plot()
plt.show()

# Works with any Pandas operation
result = df.groupby('category')['sales'].sum()
```

## 🆚 Comparison with Alternatives

| Solution | Speed | Memory Savings | Pandas Compatible | Learning Curve |
|----------|-------|----------------|-------------------|----------------|
| **Diet Pandas** | ⚡⚡⚡ Fast | 🎯 50-80% | ✅ 100% | ✅ None |
| Manual downcasting | 🐌 Slow | 🎯 50-80% | ✅ Yes | ❌ High |
| Polars | ⚡⚡⚡ Very Fast | 🎯 60-90% | ❌ No | ⚠️ Medium |
| Dask | ⚡⚡ Medium | 🎯 Varies | ⚠️ Partial | ⚠️ Medium |

## 🛠️ Development

### Setup

```bash
git clone https://github.com/yourusername/diet-pandas.git
cd diet-pandas

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Examples

```bash
python scripts/examples.py

# Or run the interactive demo
python scripts/demo.py
```

### Project Structure

```
diet-pandas/
├── src/
│   └── dietpandas/
│       ├── __init__.py      # Public API
│       ├── core.py          # Optimization logic
│       └── io.py            # Fast I/O with Polars
├── tests/
│   ├── test_core.py         # Core function tests
│   └── test_io.py           # I/O function tests
├── scripts/
│   ├── demo.py              # Interactive demo
│   ├── examples.py          # Usage examples
│   └── quickstart.py        # Setup script
├── pyproject.toml           # Project configuration
├── README.md                # Documentation
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # Contribution guide
└── LICENSE                  # MIT License
```

## 📝 API Reference

### Core Functions

#### `diet(df, verbose=True, aggressive=False, categorical_threshold=0.5, inplace=False)`

Optimize an existing DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to optimize
- `verbose` (bool): Print memory reduction statistics
- `aggressive` (bool): Use float16 instead of float32 (may lose precision)
- `categorical_threshold` (float): Convert to category if unique_ratio < threshold
- `inplace` (bool): Modify DataFrame in place

**Returns:** Optimized pd.DataFrame

#### `get_memory_report(df)`

Get detailed memory usage report per column.

**Returns:** DataFrame with memory statistics

### I/O Functions

#### `read_csv(filepath, optimize=True, aggressive=False, verbose=False, use_polars=True, **kwargs)`

Read CSV with automatic optimization.

#### `read_parquet(filepath, optimize=True, aggressive=False, verbose=False, use_polars=True, **kwargs)`

Read Parquet with automatic optimization.

#### `read_excel(filepath, optimize=True, aggressive=False, verbose=False, **kwargs)`

Read Excel with automatic optimization.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of the excellent [Pandas](https://pandas.pydata.org/) library
- Uses [Polars](https://www.pola.rs/) for high-speed CSV parsing
- Inspired by the need for memory-efficient data science workflows

## 📬 Contact

- GitHub: [@luiz826](https://github.com/luiz826)
- Issues: [GitHub Issues](https://github.com/luiz826/diet-pandas/issues)

---

**Remember:** A lean DataFrame is a happy DataFrame! 🐼🥗
