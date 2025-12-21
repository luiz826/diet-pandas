"""
Diet Pandas 🐼🥗

Same Pandas taste, half the calories (RAM).

Diet Pandas automatically optimizes Pandas DataFrames to use 50-80% less memory
without losing information, by intelligently downcasting numeric types and
converting repetitive strings to categories.
"""

__version__ = "0.2.0"

from .core import (
    diet,
    get_memory_report,
    optimize_datetime,
    optimize_float,
    optimize_int,
    optimize_obj,
    optimize_sparse,
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

__all__ = [
    # Core optimization functions
    "diet",
    "optimize_int",
    "optimize_float",
    "optimize_obj",
    "optimize_datetime",
    "optimize_sparse",
    "get_memory_report",
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
]
