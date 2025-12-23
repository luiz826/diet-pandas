"""
Diet Pandas - Custom Exceptions and Warnings

This module defines custom exceptions and warnings for Diet Pandas operations.
"""


class DietPandasWarning(UserWarning):
    """Base warning class for all Diet Pandas warnings."""

    pass


class HighCardinalityWarning(DietPandasWarning):
    """
    Warning raised when attempting to convert a high-cardinality column to category.

    This warning suggests that categorical conversion may not be beneficial
    for columns with many unique values.
    """

    pass


class PrecisionLossWarning(DietPandasWarning):
    """
    Warning raised when aggressive optimization may cause precision loss.

    This warning is raised when converting float64 to float16, which can
    result in loss of precision for values outside the float16 range.
    """

    pass


class OptimizationSkippedWarning(DietPandasWarning):
    """
    Warning raised when optimization is skipped for a column.

    This can occur when a column contains all NaN values or when
    optimization would not provide any benefit.
    """

    pass


class OptimizationError(Exception):
    """
    Base exception class for Diet Pandas optimization errors.

    Raised when an optimization operation fails and cannot be recovered.
    """

    pass
