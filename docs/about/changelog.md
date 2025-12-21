# Changelog

All notable changes to Diet Pandas are documented here.

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
- **Performance Benchmarking**: Comprehensive benchmark script
- New parameters for `diet()`:
  - `optimize_datetimes` - Enable/disable datetime optimization (default: True)
  - `optimize_sparse_cols` - Enable sparse optimization (default: False)
  - `sparse_threshold` - Threshold for sparse conversion (default: 0.9)
- **pytables Dependency**: Added optional HDF5 support

### Improved
- Enhanced `diet()` function with datetime and sparse support
- Better test coverage (46 tests, 100% passing)
- Test file consolidation for cleaner structure
- Comprehensive documentation updates
- Emojis in output messages for better UX (🥗)

### Performance
- 82-87% memory reduction on typical datasets (10K-500K rows)
- Up to 96% memory reduction on sparse data
- Optimization time: 0.007-0.16 seconds depending on dataset size

## [0.1.0] - 2025-12-19

### Added
- Initial release of Diet Pandas
- Core optimization engine with intelligent type downcasting
- `diet()` function for optimizing existing DataFrames
- `optimize_int()` for integer optimization
- `optimize_float()` for float optimization
- `optimize_obj()` for string to category conversion
- `get_memory_report()` for detailed memory analysis
- Fast I/O module with Polars integration
- `read_csv()` with automatic optimization (5-10x faster)
- `read_parquet()` with automatic optimization
- `read_excel()` with automatic optimization
- `to_csv_optimized()` for saving optimized DataFrames
- Aggressive mode for maximum compression
- Comprehensive test suite (95%+ coverage)
- Complete documentation and examples

### Features
- 50-80% memory reduction on typical datasets
- 5-10x faster CSV loading with Polars
- 100% Pandas compatibility
- Safe, lossless optimization by default

## Links

- [GitHub Repository](https://github.com/luiz826/diet-pandas)
- [PyPI Package](https://pypi.org/project/diet-pandas/)
- [Issue Tracker](https://github.com/luiz826/diet-pandas/issues)
