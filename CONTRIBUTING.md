# Contributing to Diet Pandas

Thank you for your interest in contributing to Diet Pandas! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, package versions)

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Potential implementation approach (optional)

### Code Contributions

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/diet-pandas.git
   cd diet-pandas
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set Up Development Environment**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Make Your Changes**
   - Write clean, readable code
   - Follow existing code style
   - Add docstrings to new functions
   - Include type hints where appropriate

5. **Add Tests**
   ```bash
   # Add tests to tests/test_core.py or tests/test_io.py
   pytest tests/ -v
   ```

6. **Run Tests**
   ```bash
   # Make sure all tests pass
   pytest tests/ -v
   
   # Check coverage
   pytest tests/ --cov=dietpandas
   ```

7. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

8. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub.

## 📝 Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Keep functions focused and small
- Maximum line length: 100 characters

### Docstring Style

Use Google-style docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Brief description of what the function does.
    
    More detailed explanation if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Examples:
        >>> my_function(5, "test")
        True
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Union, List
import pandas as pd

def process_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_optimize_int_converts_to_uint8`
- Test both success and failure cases
- Test edge cases (empty data, NaN values, etc.)

### Test Structure

```python
import pytest
import pandas as pd
from dietpandas.core import my_function

class TestMyFunction:
    """Tests for my_function."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        input_data = pd.Series([1, 2, 3])
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result.dtype == expected_dtype
        
    def test_edge_case_empty_series(self):
        """Test with empty series."""
        input_data = pd.Series([])
        result = my_function(input_data)
        assert len(result) == 0
```

## 📚 Documentation

### Updating Documentation

When adding features:
1. Update `README.md` with usage examples
2. Update `QUICKREF.md` if it's a common use case
3. Add docstrings to all new functions
4. Update `DEVELOPMENT.md` if it affects development workflow

### Example Documentation

Include examples in docstrings:

```python
def diet(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.
    
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> optimized = diet(df)
        🥗 Diet Complete: Memory reduced by 62.5%
    """
    pass
```

## 🎯 Areas for Contribution

### High Priority

- [x] Additional file format support (JSON, HDF5, Feather) ✅ **COMPLETED**
- [x] DateTime optimization ✅ **COMPLETED**
- [x] Sparse data handling ✅ **COMPLETED**
- [x] Performance benchmarks ✅ **COMPLETED**

### Medium Priority

- [ ] More comprehensive tests
- [ ] Additional optimization strategies
- [ ] Documentation improvements
- [ ] Example notebooks
- [ ] Automated backend selection based on data size
- [ ] Integration with Dask/Modin for distributed computing

### Low Priority

- [ ] Additional utility functions
- [ ] Performance profiling tools
- [ ] CI/CD pipeline improvements

## 🔍 Code Review Process

### What We Look For

- ✅ Code works correctly
- ✅ Tests are included and pass
- ✅ Documentation is updated
- ✅ Code style is consistent
- ✅ No breaking changes (or clearly documented)

### Review Timeline

- We aim to review PRs within 3-5 business days
- Complex changes may take longer
- Feel free to ping if no response after a week

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create git tag
5. Build and publish to PyPI

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- `README.md` contributors section
- Release notes
- Project documentation

## 💬 Getting Help

- **Questions**: Open a GitHub discussion
- **Bugs**: Open a GitHub issue
- **Chat**: (Add Discord/Slack link if available)

## 📞 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

Thank you for contributing to Diet Pandas! 🐼🥗
