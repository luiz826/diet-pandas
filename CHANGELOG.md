# Changelog

All notable changes to Diet Pandas will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-21

### Added
- **DateTime Optimization**: New `optimize_datetime()` function for efficient datetime handling
- **Sparse Data Support**: New `optimize_sparse()` function for sparse array optimization (up to 96% memory reduction on sparse data)
- **Extended File Format Support**:
  - `read_json()` - JSON file reading with optimization
  - `read_hdf()` - HDF5 file reading with optimization
  - `read_feather()` - Feather file reading with optimization
  - `to_parquet_optimized()` - Optimized Parquet writing
  - `to_feather_optimized()` - Optimized Feather writing
- **Performance Benchmarking**: Comprehensive benchmark script (`scripts/benchmark.py`)
- New parameters for `diet()`:
  - `optimize_datetimes` - Enable/disable datetime optimization (default: True)
  - `optimize_sparse_cols` - Enable sparse optimization (default: False)
  - `sparse_threshold` - Threshold for sparse conversion (default: 0.9)

### Improved
- Enhanced `diet()` function with datetime and sparse support
- Better test coverage with 12 new tests for new features
- Comprehensive documentation updates
- Emojis in output messages for better UX (🥗)

### Performance
- 82-87% memory reduction on typical datasets (tested with 10K-500K rows)
- Up to 96% memory reduction on sparse data
- Optimization time: 0.007-0.16 seconds depending on dataset size

### Documentation
- Updated README with new features
- Improved CONTRIBUTING.md with completed tasks
- Added performance benchmarking guide
- New example code for sparse and datetime optimization

## [0.1.0] - 2025-12-19

### Added
- Initial release of Diet Pandas
- Core optimization engine with intelligent type downcasting
- `diet()` function for optimizing existing DataFrames
- `optimize_int()` for integer optimization (int64 → int8/int16/uint8/uint16)
- `optimize_float()` for float optimization (float64 → float32/float16)
- `optimize_obj()` for string to category conversion
- `get_memory_report()` for detailed memory usage analysis
- Fast I/O module with Polars integration
- `read_csv()` with automatic optimization (5-10x faster)
- `read_parquet()` with automatic optimization
- `read_excel()` with automatic optimization
- `to_csv_optimized()` for saving optimized DataFrames
- Aggressive mode for maximum compression (float16)
- Customizable categorical threshold
- In-place optimization option
- Verbose mode with memory reduction statistics
- Comprehensive test suite (95%+ coverage)
- Detailed documentation and examples
- Quick reference card
- Development guide
- Contributing guidelines

### Features
- 50-80% memory reduction on typical datasets
- 5-10x faster CSV loading with Polars
- 100% Pandas compatibility
- Automatic fallback to standard Pandas if Polars unavailable
- Safe, lossless optimization by default
- Optional aggressive mode for maximum compression

### Documentation
- Complete README with examples
- Quick reference card (QUICKREF.md)
- Development guide (DEVELOPMENT.md)
- Contributing guidelines (CONTRIBUTING.md)
- Project summary (PROJECT_SUMMARY.md)
- 5 comprehensive examples (examples.py)
- Inline docstring examples
- API documentation

### Testing
- Unit tests for all core functions
- Integration tests for I/O operations
- Edge case testing (NaN, empty data, etc.)
- Cross-platform testing (Linux, macOS, Windows)
- Python 3.10+ compatibility testing

### Infrastructure
- PyPI-ready package structure
- GitHub Actions CI/CD workflow
- Makefile for common tasks
- Quick setup script
- MIT License

## [Unreleased]

### Planned for 0.2.0
- DateTime optimization
- Boolean optimization
- Sparse data handling
- JSON format support
- HDF5 format support
- Feather format support

### Planned for 0.3.0
- Parallel processing support
- Streaming optimization
- Custom optimization profiles
- Performance benchmarking tools

### Future Considerations
- Web dashboard for visualization
- Jupyter notebook extension
- VS Code extension
- Auto-optimization on DataFrame creation
- Integration with Dask for big data
- GPU acceleration support

## Release Notes

### Version 0.1.0

This is the initial release of Diet Pandas, a memory optimization library for Pandas DataFrames.

**Key Highlights:**
- Reduces DataFrame memory usage by 50-80% without data loss
- Loads CSV files 5-10x faster than standard Pandas
- Fully compatible with existing Pandas workflows
- Easy to use: just replace `pd.read_csv()` with `dp.read_csv()`

**Example Usage:**

```python
import dietpandas as dp

# Fast loading with automatic optimization
df = dp.read_csv("large_file.csv")
# 🥗 Diet Complete: Memory reduced by 67.3%

# Or optimize existing DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
df = dp.diet(df)
```

**Technical Details:**
- Uses Polars for multi-threaded CSV parsing
- Intelligent type downcasting algorithms
- Automatic categorical conversion for low-cardinality strings
- Comprehensive test coverage
- Cross-platform support

**Installation:**

```bash
pip install diet-pandas
```

**Documentation:**
- See README.md for complete documentation
- See QUICKREF.md for quick reference
- See examples.py for usage examples

---

For more information, visit: https://github.com/yourusername/diet-pandas
