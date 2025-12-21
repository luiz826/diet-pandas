"""
Unit tests for Diet Pandas core optimization functions.
"""

import pytest
import pandas as pd
import numpy as np
from dietpandas.core import (
    optimize_int,
    optimize_float,
    optimize_obj,
    diet,
    get_memory_report,
)


class TestOptimizeInt:
    """Tests for integer optimization."""
    
    def test_optimize_small_unsigned_integers(self):
        """Test that small positive integers are converted to uint8."""
        s = pd.Series([0, 1, 2, 100, 255], dtype='int64')
        result = optimize_int(s)
        assert result.dtype == np.uint8
        assert list(result) == [0, 1, 2, 100, 255]
    
    def test_optimize_medium_unsigned_integers(self):
        """Test that medium positive integers are converted to uint16."""
        s = pd.Series([0, 1000, 30000], dtype='int64')
        result = optimize_int(s)
        assert result.dtype == np.uint16
        assert list(result) == [0, 1000, 30000]
    
    def test_optimize_small_signed_integers(self):
        """Test that small signed integers are converted to int8."""
        s = pd.Series([-100, 0, 100], dtype='int64')
        result = optimize_int(s)
        assert result.dtype == np.int8
        assert list(result) == [-100, 0, 100]
    
    def test_optimize_medium_signed_integers(self):
        """Test that medium signed integers are converted to int16."""
        s = pd.Series([-1000, 0, 1000], dtype='int64')
        result = optimize_int(s)
        assert result.dtype == np.int16
        assert list(result) == [-1000, 0, 1000]
    
    def test_no_downcast_when_not_beneficial(self):
        """Test that very large integers stay as int64."""
        s = pd.Series([0, 2**40], dtype='int64')
        result = optimize_int(s)
        assert result.dtype == np.uint64


class TestOptimizeFloat:
    """Tests for float optimization."""
    
    def test_optimize_float_to_float32(self):
        """Test that float64 is converted to float32 in safe mode."""
        s = pd.Series([1.1, 2.2, 3.3], dtype='float64')
        result = optimize_float(s, aggressive=False)
        assert result.dtype == np.float32
    
    def test_optimize_float_to_float16_aggressive(self):
        """Test that float64 is converted to float16 in aggressive mode."""
        s = pd.Series([1.1, 2.2, 3.3], dtype='float64')
        result = optimize_float(s, aggressive=True)
        assert result.dtype == np.float16


class TestOptimizeObj:
    """Tests for object/string optimization."""
    
    def test_optimize_low_cardinality_to_category(self):
        """Test that low cardinality strings are converted to category."""
        s = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'] * 10)
        result = optimize_obj(s, categorical_threshold=0.5)
        assert result.dtype.name == 'category'
        assert list(result.unique()) == ['A', 'B']
    
    def test_no_optimization_high_cardinality(self):
        """Test that high cardinality strings remain as object."""
        s = pd.Series([f'ID_{i}' for i in range(100)])
        result = optimize_obj(s, categorical_threshold=0.5)
        assert result.dtype == 'object'
    
    def test_empty_series(self):
        """Test that empty series are handled correctly."""
        s = pd.Series([], dtype='object')
        result = optimize_obj(s)
        assert result.dtype == 'object'


class TestDiet:
    """Tests for the main diet function."""
    
    def test_diet_reduces_memory(self):
        """Test that diet function reduces memory usage."""
        df = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'category_col': ['A', 'B', 'A', 'B', 'A'],
        })
        
        # Convert to wasteful types
        df['small_int'] = df['small_int'].astype('int64')
        df['float_col'] = df['float_col'].astype('float64')
        
        start_mem = df.memory_usage(deep=True).sum()
        result = diet(df, verbose=False)
        end_mem = result.memory_usage(deep=True).sum()
        
        assert end_mem < start_mem
    
    def test_diet_preserves_data(self):
        """Test that diet function preserves data values."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.1, 2.2, 3.3],
            'c': ['x', 'y', 'z'],
        })
        
        result = diet(df, verbose=False)
        
        # Check that values are preserved
        assert list(result['a']) == [1, 2, 3]
        assert list(result['b']) == pytest.approx([1.1, 2.2, 3.3], rel=1e-5)
        assert list(result['c']) == ['x', 'y', 'z']
    
    def test_diet_inplace(self):
        """Test that diet function can modify DataFrame in place."""
        df = pd.DataFrame({
            'a': pd.Series([1, 2, 3], dtype='int64'),
        })
        
        original_id = id(df)
        diet(df, verbose=False, inplace=True)
        
        # When inplace=True, should modify the original DataFrame
        assert df['a'].dtype == np.uint8
    
    def test_diet_copy(self):
        """Test that diet function returns a copy by default."""
        df = pd.DataFrame({
            'a': pd.Series([1, 2, 3], dtype='int64'),
        })
        
        original_dtype = df['a'].dtype
        result = diet(df, verbose=False, inplace=False)
        
        # Original should be unchanged
        assert df['a'].dtype == original_dtype
        # Result should be optimized
        assert result['a'].dtype == np.uint8
    
    def test_diet_with_nan_values(self):
        """Test that diet handles NaN values correctly."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [1.1, np.nan, 3.3, 4.4],
        })
        
        result = diet(df, verbose=False)
        assert result.isna().sum().sum() == 2  # Should preserve NaN values
    
    def test_diet_aggressive_mode(self):
        """Test that aggressive mode uses float16."""
        df = pd.DataFrame({
            'float_col': [1.1, 2.2, 3.3],
        })
        
        result = diet(df, verbose=False, aggressive=True)
        assert result['float_col'].dtype == np.float16


class TestGetMemoryReport:
    """Tests for memory reporting function."""
    
    def test_memory_report_structure(self):
        """Test that memory report has correct structure."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
        })
        
        report = get_memory_report(df)
        
        assert 'column' in report.columns
        assert 'dtype' in report.columns
        assert 'memory_bytes' in report.columns
        assert 'memory_mb' in report.columns
        assert 'percent_of_total' in report.columns
    
    def test_memory_report_sorted(self):
        """Test that memory report is sorted by memory usage."""
        df = pd.DataFrame({
            'small': [1, 2],
            'large': ['x' * 1000, 'y' * 1000],
        })
        
        report = get_memory_report(df)
        
        # First row should be the largest memory consumer
        assert report.iloc[0]['memory_bytes'] >= report.iloc[1]['memory_bytes']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
