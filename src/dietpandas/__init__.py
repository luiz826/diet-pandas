"""
Diet Pandas 🐼🥗

Same Pandas taste, half the calories (RAM).

Diet Pandas automatically optimizes Pandas DataFrames to use 50-80% less memory
without losing information, by intelligently downcasting numeric types and
converting repetitive strings to categories.
"""

__version__ = "0.4.0"

from .analysis import (
    analyze,
    estimate_memory_reduction,
    get_optimization_summary,
)
from .core import (
    diet,
    get_memory_report,
    optimize_bool,
    optimize_datetime,
    optimize_float,
    optimize_int,
    optimize_obj,
    optimize_sparse,
)
from .exceptions import (
    DietPandasWarning,
    HighCardinalityWarning,
    OptimizationError,
    OptimizationSkippedWarning,
    PrecisionLossWarning,
)
from .io import (
    read_csv,
    read_excel,
    read_feather,
    read_hdf,
    read_json,
    read_parquet,
    to_csv_optimized,
    to_feather_optimized,
    to_parquet_optimized,
)
from .schema import (
    apply_schema,
    auto_schema_path,
    load_schema,
    save_schema,
)

__all__ = [
    # Core optimization functions
    "diet",
    "optimize_int",
    "optimize_float",
    "optimize_obj",
    "optimize_bool",
    "optimize_datetime",
    "optimize_sparse",
    "get_memory_report",
    # Analysis functions
    "analyze",
    "get_optimization_summary",
    "estimate_memory_reduction",
    # Exceptions and warnings
    "DietPandasWarning",
    "HighCardinalityWarning",
    "PrecisionLossWarning",
    "OptimizationSkippedWarning",
    "OptimizationError",
    # IO functions
    "read_csv",
    "read_parquet",
    "read_excel",
    "read_json",
    "read_hdf",
    "read_feather",
    "to_csv_optimized",
    "to_parquet_optimized",
    "to_feather_optimized",
    # Schema functions
    "save_schema",
    "load_schema",
    "apply_schema",
    "auto_schema_path",
]
