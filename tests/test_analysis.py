"""
Unit tests for the deforestation analysis module
"""
import pytest
import pandas as pd
import numpy as np
import os


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_dataset_exists(self):
        """Test that the deforestation dataset exists"""
        dataset_path = "deforestation_dataset.csv"
        assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"
    
    def test_dataset_format(self):
        """Test that the dataset is a valid CSV"""
        dataset_path = "deforestation_dataset.csv"
        try:
            df = pd.read_csv(dataset_path)
            assert not df.empty, "Dataset is empty"
            assert len(df) > 0, "No rows in dataset"
        except Exception as e:
            pytest.fail(f"Failed to load CSV: {e}")
    
    def test_dataset_has_columns(self):
        """Test that dataset has expected columns"""
        dataset_path = "deforestation_dataset.csv"
        df = pd.read_csv(dataset_path)
        assert len(df.columns) > 0, "Dataset has no columns"


class TestAnalysisModule:
    """Test analysis module functionality"""
    
    def test_analysis_module_imports(self):
        """Test that analysis module can be imported"""
        try:
            import analysis
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import analysis module: {e}")
    
    def test_basic_dataframe_operations(self):
        """Test basic pandas operations work"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        assert df.shape == (3, 2)
        assert df['col1'].sum() == 6


class TestDependencies:
    """Test that all required dependencies are installed"""
    
    def test_numpy_installed(self):
        """Test numpy is available"""
        import numpy
        assert numpy is not None
    
    def test_pandas_installed(self):
        """Test pandas is available"""
        import pandas
        assert pandas is not None
    
    def test_sklearn_installed(self):
        """Test scikit-learn is available"""
        from sklearn import datasets
        assert datasets is not None
    
    def test_matplotlib_installed(self):
        """Test matplotlib is available"""
        import matplotlib
        assert matplotlib is not None


if __name__ == "__main__":
    pytest.main([__file__])
