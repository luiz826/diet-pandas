# Changelog

All notable changes to Diet Pandas will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-12-23

### Added
- **Smart Float-to-Integer Conversion**: Automatically detect and convert float columns to integers when they contain only whole numbers
  - Detects floats with no decimal part (e.g., 1.0, 2.0, 3.0)
  - Converts to smallest appropriate integer type (int8, int16, uint8, etc.)
  - Preserves NaN values using nullable integer types (Int8, UInt8, etc.)
  - New `float_to_int` parameter for `diet()` and `optimize_float()` (default: True)
  - Can be disabled with `float_to_int=False` to preserve float types
  - Significant memory savings for ID columns, year fields, counts, and categorical codes
  - 9 comprehensive test cases covering edge cases and NaN handling

### Improved
- Enhanced `optimize_float()` with intelligent integer detection logic
- Updated documentation with float-to-int examples and use cases
- Added `float_to_int_demo.py` script demonstrating the feature

### Performance
- Up to 50% memory reduction for datasets with float-typed integer columns
- Common in CSV files where numeric columns are loaded as float64 by default

### Tests
- 128 total tests passing (9 new float-to-int conversion tests)

## [0.3.0] - 2025-12-23

### Added
- **Automatic Chunked Reading**: Memory-aware CSV reading
  - Automatically switches to chunked reading for large files
  - Estimates file size and available memory with `psutil`
  - Configurable `memory_threshold` (default: 70% of available RAM)
  - `auto_chunk` parameter to enable/disable (default: True)
  - Works seamlessly with schema persistence
  - Prevents out-of-memory errors on large datasets

- **Schema Persistence**: Save and reuse optimization schemas
  - `save_schema()` - Save DataFrame schema to JSON
  - `load_schema()` - Load schema from JSON file
  - `apply_schema()` - Apply saved schema to DataFrame
  - `auto_schema_path()` - Generate schema file paths automatically
  - Skip re-analysis on repeated loads for faster processing
  - Integration with `read_csv()` via `schema_path` and `save_schema` parameters
  - 16 comprehensive tests for schema operations

### Improved
- Enhanced `read_csv()` with automatic memory management
- Added `psutil>=5.9.0` dependency for memory monitoring
- Better handling of large files with automatic chunking
- Schema-based optimization eliminates redundant analysis

### Performance
- Automatic memory-aware chunking prevents out-of-memory errors
- Schema reuse eliminates redundant analysis overhead
- Faster repeated loads when using schema persistence

### Tests
- 139 total tests passing (23 new schema/chunking tests)
- Cross-platform compatibility (Windows, macOS, Linux)
- Fixed Windows file permission issues in tests
- All code formatted with black and verified with flake8/isort

## [0.2.2] - 2025-12-22

### Fixed
- Removed broken logo and favicon references from documentation site

## [0.2.1] - 2025-12-21

### Added
- **Documentation Site**: Complete documentation website at https://luiz826.github.io/diet-pandas/
  - Getting Started guides (Installation, Quick Start)
  - User Guide (Basic Usage, File I/O, Advanced, Memory Reports)
  - API Reference with auto-generated docs
  - Performance Benchmarks
  - Changelog, Contributing, and License pages

### Fixed
- GitHub Actions workflow for documentation deployment
- CI now properly installs package before building docs

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
