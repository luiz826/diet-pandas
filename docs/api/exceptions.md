# API Reference: Exceptions & Warnings

This page documents the custom warnings and exceptions in Diet Pandas.

## Warning Classes

All Diet Pandas warnings inherit from `DietPandasWarning`, which inherits from `UserWarning`.

### DietPandasWarning

Base class for all Diet Pandas warnings.

::: dietpandas.exceptions.DietPandasWarning
    options:
      show_root_heading: true
      heading_level: 4

---

### HighCardinalityWarning

Issued when attempting to convert a high-cardinality column to categorical.

::: dietpandas.exceptions.HighCardinalityWarning
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
import dietpandas as dp
import warnings

df = pd.DataFrame({
    'id': range(10000)  # 10000 unique values in 10000 rows
})

# This will trigger a HighCardinalityWarning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dp.diet(df, warn_on_issues=True)
    
    if w:
        print(w[0].category)  # <class 'dietpandas.exceptions.HighCardinalityWarning'>
        print(w[0].message)   # "Column 'id' has high cardinality (100.0%)..."
```

---

### PrecisionLossWarning

Issued when aggressive optimization may cause precision loss.

::: dietpandas.exceptions.PrecisionLossWarning
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
import dietpandas as dp
import warnings

df = pd.DataFrame({
    'value': [1.123456789, 2.234567890, 3.345678901]
})

# Aggressive mode converts float64 -> float16, which may lose precision
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dp.diet(df, aggressive=True, warn_on_issues=True)
    
    if w:
        print(w[0].category)  # <class 'dietpandas.exceptions.PrecisionLossWarning'>
```

---

### OptimizationSkippedWarning

Issued when a column is skipped due to issues (e.g., all NaN values).

::: dietpandas.exceptions.OptimizationSkippedWarning
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
import pandas as pd
import numpy as np
import dietpandas as dp
import warnings

df = pd.DataFrame({
    'good_col': [1, 2, 3],
    'empty_col': [np.nan, np.nan, np.nan]
})

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dp.diet(df, warn_on_issues=True)
    
    # Will warn about empty_col being all NaN
    for warning in w:
        if issubclass(warning.category, OptimizationSkippedWarning):
            print(warning.message)  # "Column 'empty_col' is entirely NaN..."
```

## Exception Classes

### OptimizationError

Raised when optimization fails unexpectedly.

::: dietpandas.exceptions.OptimizationError
    options:
      show_root_heading: true
      heading_level: 4

**Example:**

```python
from dietpandas.exceptions import OptimizationError

try:
    # Some optimization operation
    result = optimize_something(data)
except OptimizationError as e:
    print(f"Optimization failed: {e}")
    # Handle the error appropriately
```

## Controlling Warnings

### Enable/Disable Warnings

```python
import dietpandas as dp

# Enable warnings (default)
df = dp.diet(df, warn_on_issues=True)

# Disable warnings
df = dp.diet(df, warn_on_issues=False)
```

### Filtering Specific Warnings

```python
import warnings
from dietpandas.exceptions import HighCardinalityWarning, PrecisionLossWarning

# Ignore high cardinality warnings
warnings.filterwarnings('ignore', category=HighCardinalityWarning)

# Ignore precision loss warnings
warnings.filterwarnings('ignore', category=PrecisionLossWarning)

# Apply optimization
df = dp.diet(df, aggressive=True, warn_on_issues=True)
```

### Treat Warnings as Errors

```python
import warnings
from dietpandas.exceptions import PrecisionLossWarning

# Make precision loss warnings raise errors
warnings.filterwarnings('error', category=PrecisionLossWarning)

try:
    df = dp.diet(df, aggressive=True, warn_on_issues=True)
except PrecisionLossWarning as e:
    print(f"Cannot proceed: {e}")
    # Use non-aggressive mode instead
    df = dp.diet(df, aggressive=False)
```

## Warning Messages

All warnings include:
- **Column name** that triggered the warning
- **Specific issue** detected
- **Actionable recommendation** for how to address it

Example warning messages:

```
⚠️  HighCardinalityWarning: Column 'id' has high cardinality (98.5%) - 
    may not benefit from categorical conversion. Consider skip_columns=['id']

⚠️  PrecisionLossWarning: Aggressive mode on column 'measurement' may lose 
    precision (float64 -> float16). Verify acceptable for your use case.

⚠️  OptimizationSkippedWarning: Column 'empty' is entirely NaN - 
    consider dropping it with df.drop(columns=['empty'])
```

## See Also

- [Core Functions](core.md) - Main optimization functions with warning support
- [Analysis Functions](analysis.md) - Pre-flight analysis to preview issues
