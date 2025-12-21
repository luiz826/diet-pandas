"""
Diet Pandas 🐼🥗

Same Pandas taste, half the calories (RAM).

Diet Pandas automatically optimizes Pandas DataFrames to use 50-80% less memory
without losing information, by intelligently downcasting numeric types and
converting repetitive strings to categories.
"""

__version__ = "0.1.0"

from .core import (
    diet,
    optimize_int,
    optimize_float,
    optimize_obj,
    get_memory_report,
)

from .io import (
    read_csv,
    read_parquet,
    read_excel,
    to_csv_optimized,
)

__all__ = [
    # Core optimization functions
    "diet",
    "optimize_int",
    "optimize_float",
    "optimize_obj",
    "get_memory_report",
    
    # IO functions
    "read_csv",
    "read_parquet",
    "read_excel",
    "to_csv_optimized",
]
