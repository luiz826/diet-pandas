# Diet Pandas 🐼🥗

**Tagline:** *Same Pandas taste, half the calories (RAM).*

[![PyPI version](https://badge.fury.io/py/diet-pandas.svg)](https://pypi.org/project/diet-pandas/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 The Problem

Pandas is built for safety and ease of use, not memory efficiency. When you load a CSV, standard Pandas defaults to "safe" but wasteful data types:

* **`int64`** for small integers (wasting 75%+ memory per number)
* **`float64`** for simple metrics (wasting 50% memory per number)
* **`object`** for repetitive strings (wasting massive amounts of memory and CPU)

**Diet Pandas** solves this by acting as a strict nutritionist for your data. It aggressively analyzes data distributions and "downcasts" types to the smallest safe representation—often reducing memory usage by **50% to 80%** without losing information.

## 🚀 Quick Example

```python
import dietpandas as dp

# Drop-in replacement for pandas.read_csv
# Loads faster and uses less RAM automatically
df = dp.read_csv("huge_dataset.csv")
# 🥗 Diet Complete: Memory reduced by 67.3%
#    450.00MB -> 147.15MB

# Or optimize an existing DataFrame
import pandas as pd
df_heavy = pd.DataFrame({
    'year': [2020, 2021, 2022], 
    'revenue': [1.1, 2.2, 3.3]
})
df_light = dp.diet(df_heavy)
# 🥗 Diet Complete: Memory reduced by 62.5%
```

## ✨ Key Features

- **🏃 Fast Loading**: Uses Polars engine for 5-10x faster CSV parsing
- **🎯 Smart Optimization**: Automatically downcasts numeric types to smallest safe representation
- **🗜️ Sparse Support**: Optimizes columns with many repeated values (95%+ reduction)
- **📅 DateTime Handling**: Efficient datetime column optimization
- **📊 Multiple Formats**: CSV, Parquet, Excel, JSON, HDF5, Feather
- **🔥 Aggressive Mode**: Optional extreme compression for maximum memory savings
- **📈 Memory Reports**: Detailed analysis of memory usage per column
- **✅ 100% Pandas Compatible**: Works seamlessly with all pandas operations

## 📚 Documentation

- [Installation](getting-started/installation.md) - Get started in seconds
- [Quick Start](getting-started/quickstart.md) - Basic usage examples
- [User Guide](guide/basic-usage.md) - Comprehensive tutorials
- [API Reference](api/core.md) - Complete API documentation
- [Performance](performance/benchmarks.md) - Benchmark results

## 🎯 When to Use Diet Pandas

**✅ Perfect for:**
- Loading large CSV files (>100MB)
- Working with limited RAM environments
- Training ML models on large datasets
- ETL pipelines with memory constraints
- Web applications serving data

**⚠️ Consider alternatives for:**
- Tiny datasets (<1MB) - optimization overhead not worth it
- Streaming data pipelines - consider Polars directly
- When you need maximum precision (financial calculations)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](about/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](about/license.md) file for details.

## 🔗 Links

- [GitHub Repository](https://github.com/luiz826/diet-pandas)
- [PyPI Package](https://pypi.org/project/diet-pandas/)
- [Issue Tracker](https://github.com/luiz826/diet-pandas/issues)
