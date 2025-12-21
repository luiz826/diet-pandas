"""
Unit tests for new I/O functions in Diet Pandas.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dietpandas as dp


class TestReadJSON:
    """Tests for read_json function."""

    def test_read_json_basic(self):
        """Test basic JSON reading with optimization."""
        # Create test data
        df = pd.DataFrame({
            'int_col': range(100),
            'str_col': ['A', 'B', 'C'] * 33 + ['A']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            df.to_json(f.name)
            temp_file = f.name
        
        try:
            result = dp.read_json(temp_file, verbose=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            os.unlink(temp_file)

    def test_read_json_no_optimization(self):
        """Test JSON reading without optimization."""
        df = pd.DataFrame({'col': range(10)})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            df.to_json(f.name)
            temp_file = f.name
        
        try:
            result = dp.read_json(temp_file, optimize=False)
            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(temp_file)


class TestReadFeather:
    """Tests for read_feather function."""

    def test_read_feather_basic(self):
        """Test basic Feather reading with optimization."""
        df = pd.DataFrame({
            'int_col': range(100),
            'float_col': np.random.randn(100)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_feather(temp_file)
            result = dp.read_feather(temp_file, verbose=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_feather_no_optimization(self):
        """Test Feather reading without optimization."""
        df = pd.DataFrame({'col': range(10)})
        
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_feather(temp_file)
            result = dp.read_feather(temp_file, optimize=False)
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestReadHDF:
    """Tests for read_hdf function."""

    def test_read_hdf_basic(self):
        """Test basic HDF5 reading with optimization."""
        df = pd.DataFrame({
            'int_col': range(100),
            'float_col': np.random.randn(100)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_hdf(temp_file, key='data', mode='w')
            result = dp.read_hdf(temp_file, key='data', verbose=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_hdf_no_optimization(self):
        """Test HDF5 reading without optimization."""
        df = pd.DataFrame({'col': range(10)})
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_hdf(temp_file, key='data', mode='w')
            result = dp.read_hdf(temp_file, key='data', optimize=False)
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestWriteOptimized:
    """Tests for optimized write functions."""

    def test_to_parquet_optimized(self):
        """Test optimized Parquet writing."""
        df = pd.DataFrame({
            'int_col': range(1000),
            'float_col': np.random.randn(1000)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_file = f.name
        
        try:
            dp.to_parquet_optimized(df, temp_file, optimize_before_save=True)
            assert os.path.exists(temp_file)
            
            # Read back and verify
            result = pd.read_parquet(temp_file)
            assert len(result) == 1000
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_to_feather_optimized(self):
        """Test optimized Feather writing."""
        df = pd.DataFrame({
            'int_col': range(1000),
            'float_col': np.random.randn(1000)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as f:
            temp_file = f.name
        
        try:
            dp.to_feather_optimized(df, temp_file, optimize_before_save=True)
            assert os.path.exists(temp_file)
            
            # Read back and verify
            result = pd.read_feather(temp_file)
            assert len(result) == 1000
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_to_csv_optimized_no_optimization(self):
        """Test CSV writing without optimization."""
        df = pd.DataFrame({'col': range(10)})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            dp.to_csv_optimized(df, temp_file, optimize_before_save=False)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
