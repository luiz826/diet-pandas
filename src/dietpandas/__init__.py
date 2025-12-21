"""
Diet Pandas 🐼🥗

Same Pandas taste, half the calories (RAM).

Diet Pandas automatically optimizes Pandas DataFrames to use 50-80% less memory
without losing information, by intelligently downcasting numeric types and
converting repetitive strings to categories.
"""

__version__ = "0.1.1"

from .core import (
    diet,
    get_memory_report,
    optimize_float,
    optimize_int,
    optimize_obj,
)
from .io import (
    read_csv,
    read_excel,
    read_parquet,
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
