# Diet-Pandas Benchmarks

This directory contains performance benchmarks demonstrating diet-pandas effectiveness on both synthetic and real-world datasets.

## 📁 Structure

```
benchmarks/
├── README.md                      # This file
├── synthetic_benchmark.py         # Automated synthetic data benchmarks
├── enem_real_benchmark.py        # Real-world ENEM dataset benchmarks
├── enem_benchmark.ipynb          # Interactive ENEM analysis notebook
└── results/
    ├── synthetic_results.md      # Synthetic benchmark results
    ├── enem_results.md           # ENEM benchmark results
    └── enem_results.json         # Raw ENEM metrics (JSON)
```

## 🚀 Quick Start

### Run Synthetic Benchmarks

```bash
# From diet-pandas root directory
uv run python benchmarks/synthetic_benchmark.py
```

**Tests:**
- Multiple dataset sizes (10K to 500K rows)
- Mixed data types (integers, floats, categories, sparse, datetime)
- File I/O performance
- Sparse data optimization

**Results:** 85%+ memory reduction, <1% time overhead

### Run Real-World ENEM Benchmarks

```bash
# Requires ENEM 2024 data in ../bench-diet-pandas/
uv run python benchmarks/enem_real_benchmark.py
```

**Tests:**
- 4.3 million student records
- Real government dataset
- Geographic/categorical data
- Multiple large CSV files

**Results:** 63-96% memory reduction, 2.7-5.4 GB saved per file

## 📊 Benchmark Results

### Synthetic Data Summary

| Dataset Size | Memory Reduction | Optimization Time |
|--------------|------------------|-------------------|
| 10K rows     | 82.3%           | 0.009 sec        |
| 50K rows     | 85.8%           | 0.033 sec        |
| 100K rows    | 86.3%           | 0.061 sec        |
| 500K rows    | 86.6%           | 0.304 sec        |

[Full Results →](results/synthetic_results.md)

### Real-World Data Summary

| Dataset | Rows | Memory Reduction | Memory Saved |
|---------|------|------------------|--------------|
| ENEM Results | 4.3M | **62.7%** | **2.7 GB** |
| ENEM Participants | 4.3M | **96.2%** | **5.4 GB** |

[Full Results →](results/enem_results.md)

## 🎯 Key Findings

### What Works Best

1. **Categorical Strings** → 96-98% reduction
   - State codes, city names, categories
   - Repeated values across millions of rows

2. **Small Integers** → 87.5% reduction
   - Type indicators (0-9)
   - Small counts, flags
   - int64 → uint8/uint16

3. **Sparse Data** → 75-78% reduction
   - Binary features (0/1)
   - Many zeros or NaNs

4. **Nullable Integers** → Preserves NaN while optimizing
   - Optional fields with missing data
   - Uses Int8, UInt8, etc.

### Trade-offs

✅ **Pros:**
- Massive memory savings (60-96%)
- Consistent across dataset sizes
- Handles real-world data (NaN, mixed types)
- Minimal overhead (<1% for typical workflows)

⚠️ **Cons:**
- Initial load 2-3x slower on some files
- Best for iterative analysis (load once, query many times)
- Less benefit on already-optimized data

## 🔬 Running Your Own Benchmarks

### Custom Synthetic Data

```python
from benchmarks.synthetic_benchmark import create_test_dataframe, measure_memory
import dietpandas as dp

# Create test data
df = create_test_dataframe(size=100000)

# Benchmark
original_memory = measure_memory(df)
df_optimized = dp.diet(df)
optimized_memory = measure_memory(df_optimized)

reduction = (1 - optimized_memory/original_memory) * 100
print(f"Memory reduction: {reduction:.1f}%")
```

### Custom CSV File

```python
import pandas as pd
import dietpandas as dp
import time

# Benchmark pandas
start = time.time()
df_pandas = pd.read_csv("your_file.csv")
pandas_time = time.time() - start
pandas_mem = df_pandas.memory_usage(deep=True).sum() / 1024**2

# Benchmark diet-pandas
start = time.time()
df_diet = dp.read_csv("your_file.csv")
diet_time = time.time() - start
diet_mem = df_diet.memory_usage(deep=True).sum() / 1024**2

print(f"Pandas:  {pandas_time:.2f}s, {pandas_mem:.2f} MB")
print(f"Diet:    {diet_time:.2f}s, {diet_mem:.2f} MB")
print(f"Savings: {(1 - diet_mem/pandas_mem)*100:.1f}%")
```

## 📝 Adding New Benchmarks

1. Create benchmark script in `benchmarks/`
2. Follow naming convention: `{name}_benchmark.py`
3. Save results to `benchmarks/results/`
4. Update this README with summary

**Template:**
```python
import time
import pandas as pd
import dietpandas as dp

def benchmark_your_dataset():
    # Load data
    df_pandas = pd.read_csv("your_data.csv")
    df_diet = dp.read_csv("your_data.csv")
    
    # Measure and compare
    # ... your benchmark code ...
    
    return results

if __name__ == "__main__":
    benchmark_your_dataset()
```

## 🎓 Understanding the Results

### Memory Reduction %

- **50-70%:** Good - Typical for numeric data
- **70-85%:** Great - Mixed types with some categories
- **85-95%:** Excellent - High cardinality categorical data
- **95%+:** Outstanding - Highly repetitive categorical data

### When It Matters

**8GB Laptop:**
- Standard: Can load ~6GB data (with overhead)
- Diet: Can load ~20-30GB raw data (optimized to ~6GB)

**16GB Laptop:**
- Standard: Can load ~12GB data
- Diet: Can load ~40-60GB raw data (optimized to ~12GB)

**Impact:** Work on datasets 3-5x larger without upgrading hardware!

## 📚 Related Documentation

- [Main README](../README.md) - Project overview
- [Basic Usage Guide](../docs/guide/basic-usage.md) - How to use diet-pandas
- [API Reference](../docs/api/) - Function documentation

## 🤝 Contributing

Found interesting benchmark results? Please share!

1. Run benchmarks on your dataset
2. Save anonymized results
3. Open PR with benchmark script + results
4. Help others understand real-world performance

---

**Questions?** Open an issue on [GitHub](https://github.com/luiz826/diet-pandas/issues)
