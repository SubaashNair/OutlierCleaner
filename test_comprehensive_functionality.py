#!/usr/bin/env python3
"""
Comprehensive functionality test for OutlierCleaner using California Housing dataset.
This script tests all methods and documents any errors encountered.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Any, List
import traceback
import sys
import os

# Add the outlier_cleaner package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sklearn.datasets import fetch_california_housing
except ImportError:
    print("ERROR: scikit-learn not installed. Please install with: pip install scikit-learn")
    sys.exit(1)

try:
    from outlier_cleaner import OutlierCleaner
    from outlier_cleaner.utils import plot_outliers, plot_distribution, plot_boxplot, plot_qq, plot_outlier_analysis
except ImportError as e:
    print(f"ERROR: Could not import OutlierCleaner: {e}")
    sys.exit(1)

class TestResults:
    """Class to track test results and errors."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []
    
    def add_pass(self, test_name: str, details: str = ""):
        self.passed.append({"test": test_name, "details": details})
        print(f"✅ PASS: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append({"test": test_name, "error": error})
        self.errors.append(f"FAIL: {test_name} - {error}")
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {error}")
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(self.passed) + len(self.failed)}")
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")
        
        if self.failed:
            print("\nFAILED TESTS:")
            for i, fail in enumerate(self.failed, 1):
                print(f"{i}. {fail['test']}")
                print(f"   Error: {fail['error']}")
        
        if self.errors:
            print("\nALL ERRORS DOCUMENTED:")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

def load_test_data():
    """Load and prepare California Housing dataset."""
    try:
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['PRICE'] = housing.target
        return df
    except Exception as e:
        raise Exception(f"Failed to load California Housing dataset: {e}")

def test_initialization(results: TestResults, df: pd.DataFrame):
    """Test OutlierCleaner initialization."""
    try:
        # Test basic initialization
        cleaner = OutlierCleaner(df)
        results.add_pass("Basic initialization", f"DataFrame shape: {cleaner.clean_df.shape}")
        
        # Test with preserve_index=True
        cleaner_preserve = OutlierCleaner(df, preserve_index=True)
        results.add_pass("Initialization with preserve_index=True")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        cleaner_empty = OutlierCleaner(empty_df)
        results.add_pass("Initialization with empty DataFrame")
        
    except Exception as e:
        results.add_fail("Initialization", str(e))

def test_set_data(results: TestResults, df: pd.DataFrame):
    """Test set_data method."""
    try:
        cleaner = OutlierCleaner(df)
        
        # Test setting new data
        new_df = df.copy().iloc[:100]  # Smaller subset
        cleaner.set_data(new_df)
        results.add_pass("set_data method", f"New shape: {cleaner.clean_df.shape}")
        
    except Exception as e:
        results.add_fail("set_data method", str(e))

def test_analyze_distribution(results: TestResults, df: pd.DataFrame):
    """Test analyze_distribution method."""
    cleaner = OutlierCleaner(df)
    
    test_columns = ['MedInc', 'AveRooms', 'PRICE']
    
    for column in test_columns:
        try:
            analysis = cleaner.analyze_distribution(column)
            
            # Check required keys
            required_keys = ['skewness', 'kurtosis', 'recommended_method', 'is_normal']
            missing_keys = [key for key in required_keys if key not in analysis]
            
            if missing_keys:
                results.add_fail(f"analyze_distribution({column})", f"Missing keys: {missing_keys}")
            else:
                results.add_pass(f"analyze_distribution({column})", 
                               f"Skewness: {analysis['skewness']:.2f}, Method: {analysis['recommended_method']}")
                
        except Exception as e:
            results.add_fail(f"analyze_distribution({column})", str(e))
    
    # Test with non-existent column
    try:
        analysis = cleaner.analyze_distribution('NonExistentColumn')
        results.add_fail("analyze_distribution(non-existent)", "Should have raised an error")
    except Exception as e:
        results.add_pass("analyze_distribution(non-existent)", f"Correctly raised error: {type(e).__name__}")

def test_outlier_detection_methods(results: TestResults, df: pd.DataFrame):
    """Test all outlier detection methods."""
    cleaner = OutlierCleaner(df)
    test_column = 'MedInc'
    
    # Test IQR method
    try:
        cleaned_df, info = cleaner.remove_outliers_iqr(test_column)
        results.add_pass("remove_outliers_iqr", 
                        f"Removed {info['num_outliers']} outliers, {info['percent_removed']:.1f}%")
    except Exception as e:
        results.add_fail("remove_outliers_iqr", str(e))
    
    # Test Z-score method
    try:
        cleaned_df, info = cleaner.remove_outliers_zscore(test_column)
        results.add_pass("remove_outliers_zscore", 
                        f"Removed {info['num_outliers']} outliers, {info['percent_removed']:.1f}%")
    except Exception as e:
        results.add_fail("remove_outliers_zscore", str(e))
    
    # Test Modified Z-score method
    try:
        cleaned_df, info = cleaner.remove_outliers_modified_zscore(test_column)
        results.add_pass("remove_outliers_modified_zscore", 
                        f"Removed {info['num_outliers']} outliers, {info['percent_removed']:.1f}%")
    except Exception as e:
        results.add_fail("remove_outliers_modified_zscore", str(e))

def test_zscore_operations(results: TestResults, df: pd.DataFrame):
    """Test Z-score related operations."""
    cleaner = OutlierCleaner(df)
    
    # Test add_zscore_columns
    try:
        cleaner.add_zscore_columns()
        zscore_cols = [col for col in cleaner.clean_df.columns if col.endswith('_zscore')]
        results.add_pass("add_zscore_columns", f"Added {len(zscore_cols)} Z-score columns")
    except Exception as e:
        results.add_fail("add_zscore_columns", str(e))
    
    # Test clean_zscore_columns
    try:
        cleaned_df, info = cleaner.clean_zscore_columns(threshold=3.0)
        results.add_pass("clean_zscore_columns", f"Cleaned with threshold 3.0")
    except Exception as e:
        results.add_fail("clean_zscore_columns", str(e))

def test_batch_operations(results: TestResults, df: pd.DataFrame):
    """Test batch cleaning operations."""
    cleaner = OutlierCleaner(df)
    
    test_columns = ['MedInc', 'AveRooms', 'PRICE']
    
    # Test clean_columns with auto method
    try:
        cleaned_df, info = cleaner.clean_columns(columns=test_columns, method='auto', show_progress=True)
        results.add_pass("clean_columns(auto)", f"Cleaned {len(test_columns)} columns")
    except Exception as e:
        results.add_fail("clean_columns(auto)", str(e))
    
    # Test clean_columns with specific methods
    methods = ['iqr', 'zscore', 'modified_zscore']
    for method in methods:
        try:
            cleaned_df, info = cleaner.clean_columns(columns=[test_columns[0]], method=method, show_progress=False)
            results.add_pass(f"clean_columns({method})", f"Method: {method}")
        except Exception as e:
            results.add_fail(f"clean_columns({method})", str(e))

def test_statistics_and_analysis(results: TestResults, df: pd.DataFrame):
    """Test statistical analysis methods."""
    cleaner = OutlierCleaner(df)
    
    test_columns = ['MedInc', 'AveRooms', 'PRICE']
    
    # Test get_outlier_stats
    try:
        stats = cleaner.get_outlier_stats(test_columns)
        if isinstance(stats, pd.DataFrame) and not stats.empty:
            results.add_pass("get_outlier_stats", f"Generated stats for {len(stats)} entries")
        else:
            results.add_fail("get_outlier_stats", "Returned empty or invalid DataFrame")
    except Exception as e:
        results.add_fail("get_outlier_stats", str(e))
    
    # Test compare_methods
    try:
        comparison = cleaner.compare_methods(test_columns, methods=['iqr', 'zscore'])
        if isinstance(comparison, dict) and test_columns[0] in comparison:
            results.add_pass("compare_methods", f"Compared methods for {len(comparison)} columns")
        else:
            results.add_fail("compare_methods", "Invalid comparison result")
    except Exception as e:
        results.add_fail("compare_methods", str(e))

def test_outlier_tracking(results: TestResults, df: pd.DataFrame):
    """Test outlier index tracking."""
    cleaner = OutlierCleaner(df)
    test_column = 'MedInc'
    
    # First remove some outliers
    try:
        cleaned_df, info = cleaner.remove_outliers_iqr(test_column)
        
        # Test get_outlier_indices
        outlier_indices = cleaner.get_outlier_indices(test_column)
        if isinstance(outlier_indices, dict) and test_column in outlier_indices:
            results.add_pass("get_outlier_indices", f"Retrieved {len(outlier_indices[test_column])} outlier indices")
        else:
            results.add_fail("get_outlier_indices", "Invalid outlier indices result")
            
    except Exception as e:
        results.add_fail("get_outlier_indices", str(e))
    
    # Test get_outlier_indices for non-existent column
    try:
        outlier_indices = cleaner.get_outlier_indices('NonExistentColumn')
        if isinstance(outlier_indices, dict) and 'NonExistentColumn' in outlier_indices:
            if len(outlier_indices['NonExistentColumn']) == 0:
                results.add_pass("get_outlier_indices(non-existent)", "Correctly returned empty list")
            else:
                results.add_fail("get_outlier_indices(non-existent)", "Should return empty list")
        else:
            results.add_fail("get_outlier_indices(non-existent)", "Invalid result structure")
    except Exception as e:
        results.add_fail("get_outlier_indices(non-existent)", str(e))

def test_reporting(results: TestResults, df: pd.DataFrame):
    """Test reporting functionality."""
    cleaner = OutlierCleaner(df)
    
    # First perform some operations
    try:
        cleaner.remove_outliers_iqr('MedInc')
        cleaner.remove_outliers_zscore('AveRooms')
        
        # Test get_summary_report
        report = cleaner.get_summary_report()
        if report and ('original_shape' in report or 'status' in report):
            if 'status' in report:
                results.add_pass("get_summary_report", f"Generated report: {report['status']}")
            else:
                results.add_pass("get_summary_report", f"Generated report with shape {report['original_shape']} -> {report['clean_shape']}")
        else:
            results.add_fail("get_summary_report", "Invalid or empty report")
            
    except Exception as e:
        results.add_fail("get_summary_report", str(e))

def test_reset_functionality(results: TestResults, df: pd.DataFrame):
    """Test reset functionality."""
    cleaner = OutlierCleaner(df)
    original_shape = cleaner.clean_df.shape
    
    try:
        # Perform some operations
        cleaner.remove_outliers_iqr('MedInc')
        modified_shape = cleaner.clean_df.shape
        
        # Reset
        cleaner.reset()
        reset_shape = cleaner.clean_df.shape
        
        if reset_shape == original_shape:
            results.add_pass("reset", f"Successfully reset from {modified_shape} to {reset_shape}")
        else:
            results.add_fail("reset", f"Reset failed: {original_shape} -> {modified_shape} -> {reset_shape}")
            
    except Exception as e:
        results.add_fail("reset", str(e))

def test_visualization_methods(results: TestResults, df: pd.DataFrame):
    """Test visualization methods."""
    cleaner = OutlierCleaner(df)
    
    # Test integrated plot_outlier_analysis
    try:
        figures = cleaner.plot_outlier_analysis(['MedInc'])
        if isinstance(figures, dict) and 'MedInc' in figures:
            results.add_pass("plot_outlier_analysis(integrated)", "Generated plots successfully")
            plt.close('all')  # Close figures to save memory
        else:
            results.add_fail("plot_outlier_analysis(integrated)", "Invalid figure result")
    except Exception as e:
        results.add_fail("plot_outlier_analysis(integrated)", str(e))
    
    # Test visualize_outliers
    try:
        cleaner.remove_outliers_iqr('MedInc')  # First create some outliers
        cleaner.visualize_outliers('MedInc')
        results.add_pass("visualize_outliers", "Generated visualization successfully")
        plt.close('all')
    except Exception as e:
        results.add_fail("visualize_outliers", str(e))

def test_standalone_visualization(results: TestResults, df: pd.DataFrame):
    """Test standalone visualization functions."""
    # Create sample outlier mask
    data = df['MedInc'].values
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    
    # Test plot_outliers
    try:
        plot_outliers(data, outliers, title="Test Outliers")
        results.add_pass("plot_outliers(standalone)", "Generated scatter plot")
        plt.close('all')
    except Exception as e:
        results.add_fail("plot_outliers(standalone)", str(e))
    
    # Test plot_distribution
    try:
        plot_distribution(data, outliers, title="Test Distribution")
        results.add_pass("plot_distribution(standalone)", "Generated distribution plot")
        plt.close('all')
    except Exception as e:
        results.add_fail("plot_distribution(standalone)", str(e))
    
    # Test plot_boxplot
    try:
        plot_boxplot(data, outliers, title="Test Boxplot")
        results.add_pass("plot_boxplot(standalone)", "Generated box plot")
        plt.close('all')
    except Exception as e:
        results.add_fail("plot_boxplot(standalone)", str(e))
    
    # Test plot_qq
    try:
        plot_qq(data, outliers, title="Test Q-Q Plot")
        results.add_pass("plot_qq(standalone)", "Generated Q-Q plot")
        plt.close('all')
    except Exception as e:
        results.add_fail("plot_qq(standalone)", str(e))
    
    # Test plot_outlier_analysis (standalone)
    try:
        plot_outlier_analysis(data, outliers, title="Test Analysis")
        results.add_pass("plot_outlier_analysis(standalone)", "Generated analysis dashboard")
        plt.close('all')
    except Exception as e:
        results.add_fail("plot_outlier_analysis(standalone)", str(e))

def test_edge_cases(results: TestResults, df: pd.DataFrame):
    """Test edge cases and error handling."""
    
    # Test with constant column
    try:
        df_constant = df.copy()
        df_constant['constant_col'] = 5.0
        cleaner = OutlierCleaner(df_constant)
        
        cleaned_df, info = cleaner.remove_outliers_iqr('constant_col')
        results.add_pass("constant_column_iqr", "Handled constant column correctly")
    except Exception as e:
        results.add_fail("constant_column_iqr", str(e))
    
    # Test with column containing NaN values
    try:
        df_nan = df.copy()
        df_nan.loc[0:10, 'MedInc'] = np.nan
        cleaner = OutlierCleaner(df_nan)
        
        cleaned_df, info = cleaner.remove_outliers_iqr('MedInc')
        results.add_pass("nan_values", "Handled NaN values correctly")
    except Exception as e:
        results.add_fail("nan_values", str(e))
    
    # Test with single row DataFrame
    try:
        df_single = df.iloc[:1].copy()
        cleaner = OutlierCleaner(df_single)
        
        cleaned_df, info = cleaner.remove_outliers_iqr('MedInc')
        results.add_pass("single_row", "Handled single row DataFrame")
    except Exception as e:
        results.add_fail("single_row", str(e))

def main():
    """Main test function."""
    print("OutlierCleaner Comprehensive Functionality Test")
    print("Using California Housing Dataset")
    print("="*80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    results = TestResults()
    
    try:
        # Load test data
        print("Loading California Housing dataset...")
        df = load_test_data()
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print()
        
        # Run all tests
        print("Running tests...")
        print()
        
        test_initialization(results, df)
        test_set_data(results, df)
        test_analyze_distribution(results, df)
        test_outlier_detection_methods(results, df)
        test_zscore_operations(results, df)
        test_batch_operations(results, df)
        test_statistics_and_analysis(results, df)
        test_outlier_tracking(results, df)
        test_reporting(results, df)
        test_reset_functionality(results, df)
        test_visualization_methods(results, df)
        test_standalone_visualization(results, df)
        test_edge_cases(results, df)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print(traceback.format_exc())
        results.add_fail("Critical Error", str(e))
    
    finally:
        # Print summary
        results.print_summary()
        
        # Save error log if there are errors
        if results.errors:
            error_log_path = "outlier_cleaner_test_errors.log"
            with open(error_log_path, 'w') as f:
                f.write("OutlierCleaner Test Error Log\n")
                f.write("="*50 + "\n\n")
                for error in results.errors:
                    f.write(f"{error}\n")
            print(f"\nError log saved to: {error_log_path}")

if __name__ == "__main__":
    main()