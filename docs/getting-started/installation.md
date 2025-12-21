# Installation

## Using pip (Recommended)

Install Diet Pandas from PyPI:

```bash
pip install diet-pandas
```

This installs the core library with all essential dependencies.

## Optional Dependencies

Diet Pandas supports multiple file formats through optional dependencies:

### Install All Optional Dependencies

```bash
pip install "diet-pandas[all]"
```

This includes support for Excel, Parquet, HDF5, and Feather formats.

### Install Specific Format Support

#### Excel Support

```bash
pip install "diet-pandas[excel]"
```

#### Parquet Support

```bash
pip install "diet-pandas[parquet]"
```

#### HDF5 Support

```bash
pip install "diet-pandas[hdf]"
```

## Development Installation

To contribute to Diet Pandas, clone the repository and install in development mode:

```bash
git clone https://github.com/luiz826/diet-pandas.git
cd diet-pandas
pip install -e ".[dev]"
```

This installs all development dependencies including:
- Testing tools (pytest, pytest-cov)
- Code formatters (black, isort)
- Linters (flake8)
- All optional format dependencies

## Requirements

- **Python**: 3.10 or higher
- **pandas**: 1.5.0 or higher
- **numpy**: 1.20.0 or higher
- **polars**: 0.17.0 or higher

## Verify Installation

After installation, verify it works:

```python
import dietpandas as dp
print(dp.__version__)
# Should print: 0.2.0
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade diet-pandas
```

## Troubleshooting

### Import Error

If you get an import error, ensure pandas and numpy are properly installed:

```bash
pip install --upgrade pandas numpy polars
```

### File Format Issues

If you encounter errors with specific file formats:

```bash
# For Excel errors
pip install openpyxl

# For Parquet errors
pip install pyarrow

# For HDF5 errors
pip install tables
```

## Next Steps

Now that you have Diet Pandas installed, check out the [Quick Start Guide](quickstart.md) to learn basic usage.
