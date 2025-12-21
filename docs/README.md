# Diet Pandas Documentation

This directory contains the source files for the Diet Pandas documentation website.

## Local Development

To build and serve the documentation locally:

1. **Install documentation dependencies:**
   ```bash
   pip install -e ".[docs]"
   ```

2. **Serve the docs locally:**
   ```bash
   mkdocs serve
   ```

3. **Open in browser:**
   Visit http://localhost:8000

The server will auto-reload when you make changes to the documentation files.

## Building the Docs

To build the static site:

```bash
mkdocs build
```

The static site will be generated in the `site/` directory.

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `master` branch.

The deployment is handled by the GitHub Actions workflow in `.github/workflows/docs.yml`.

## Documentation Structure

```
docs/
├── index.md                          # Home page
├── getting-started/
│   ├── installation.md               # Installation guide
│   └── quickstart.md                 # Quick start tutorial
├── guide/
│   ├── basic-usage.md                # Basic usage patterns
│   ├── file-io.md                    # File I/O guide
│   ├── advanced.md                   # Advanced optimization
│   └── memory-reports.md             # Memory analysis
├── api/
│   ├── core.md                       # Core functions API
│   └── io.md                         # I/O functions API
├── performance/
│   └── benchmarks.md                 # Performance benchmarks
└── about/
    ├── changelog.md                  # Version history
    ├── contributing.md               # Contribution guide
    └── license.md                    # License information
```

## Writing Documentation

### Markdown

All documentation is written in Markdown with some extensions:

- **Code blocks with syntax highlighting**
- **Admonitions** for notes, warnings, tips
- **Tabbed content** for alternative approaches
- **Emoji support**

### Code Examples

Always include working code examples:

```python
import dietpandas as dp

df = dp.read_csv("data.csv")
```

### API Documentation

API docs are auto-generated from docstrings using mkdocstrings:

```markdown
::: dietpandas.core.diet
    options:
      show_root_heading: true
```

## Configuration

Documentation is configured in `mkdocs.yml` at the repository root.

Key configuration:
- **Theme**: Material for MkDocs
- **Plugins**: mkdocstrings for API docs
- **Extensions**: pymdown-extensions for enhanced Markdown

## Contributing

To contribute to the documentation:

1. Fork the repository
2. Make your changes in the `docs/` folder
3. Test locally with `mkdocs serve`
4. Submit a pull request

## Links

- [Live Documentation](https://luiz826.github.io/diet-pandas/)
- [GitHub Repository](https://github.com/luiz826/diet-pandas)
- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
