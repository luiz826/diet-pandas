# Contributing

See the main [CONTRIBUTING.md](https://github.com/luiz826/diet-pandas/blob/master/CONTRIBUTING.md) file for detailed contribution guidelines.

## Quick Start

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Development Setup

```bash
git clone https://github.com/luiz826/diet-pandas.git
cd diet-pandas
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use:
- **Black** for formatting
- **isort** for import sorting
- **flake8** for linting

## Documentation

Help improve the docs! All documentation is in the `docs/` folder.

To build docs locally:
```bash
pip install -e ".[docs]"
mkdocs serve
```

Then visit http://localhost:8000

## Links

- [GitHub Repository](https://github.com/luiz826/diet-pandas)
- [Issue Tracker](https://github.com/luiz826/diet-pandas/issues)
