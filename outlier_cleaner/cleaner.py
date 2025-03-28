"""
outlier_cleaner.py - A module for detecting and removing outliers in data

This module provides functions for identifying and removing outliers
using various statistical methods such as IQR and Z-score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class OutlierCleaner:
    """
    A class for detecting and removing outliers from pandas DataFrames.
    
    This class provides methods to clean data using different statistical
    approaches, visualize the outliers, and generate reports on the 
    cleaning process.
    """
    
    def __init__(self, df=None):
        """
        Initialize the OutlierCleaner with an optional DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            The DataFrame to clean
        """
        self.original_df = df.copy() if df is not None else None
        self.clean_df = df.copy() if df is not None else None
        self.outlier_info = {}
        
    def set_data(self, df):
        """
        Set or update the DataFrame to be cleaned.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to clean
        """
        self.original_df = df.copy()
        self.clean_df = df.copy()
        self.outlier_info = {}
        
    def remove_outliers_iqr(self, column, lower_factor=1.5, upper_factor=1.5):
        """
        Remove outliers from a DataFrame column using the IQR method.
        
        Parameters:
        -----------
        column : str
            The name of the column to clean
        lower_factor : float, default=1.5
            The factor to multiply the IQR by for the lower bound
        upper_factor : float, default=1.5
            The factor to multiply the IQR by for the upper bound
            
        Returns:
        --------
        pandas.DataFrame
            A DataFrame with outliers removed
        dict
            Information about the outliers removed
        """
        if self.clean_df is None:
            raise ValueError("No DataFrame has been set. Use set_data() first.")
            
        # Calculate Q1, Q3, and IQR
        Q1 = self.clean_df[column].quantile(0.25)
        Q3 = self.clean_df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds
        lower_bound = Q1 - (lower_factor * IQR)
        upper_bound = Q3 + (upper_factor * IQR)
        
        # Identify outliers
        outliers = self.clean_df[(self.clean_df[column] < lower_bound) | 
                                (self.clean_df[column] > upper_bound)]
        
        # Create a clean DataFrame without outliers
        self.clean_df = self.clean_df[(self.clean_df[column] >= lower_bound) & 
                                     (self.clean_df[column] <= upper_bound)]
        
        # Prepare outlier information
        outlier_info = {
            'method': 'IQR',
            'column': column,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_outliers': len(outliers),
            'num_outliers_below': len(self.original_df[self.original_df[column] < lower_bound]),
            'num_outliers_above': len(self.original_df[self.original_df[column] > upper_bound]),
            'percent_removed': (len(outliers) / len(self.original_df)) * 100
        }
        
        # Store outlier information
        self.outlier_info[column] = outlier_info
        
        return self.clean_df, outlier_info
    
    def remove_outliers_zscore(self, column, threshold=3.0):
        """
        Remove outliers from a DataFrame column using the Z-score method.
        
        Parameters:
        -----------
        column : str
            The name of the column to clean
        threshold : float, default=3.0
            The Z-score threshold above which to consider a point an outlier
            
        Returns:
        --------
        pandas.DataFrame
            A DataFrame with outliers removed
        dict
            Information about the outliers removed
        """
        if self.clean_df is None:
            raise ValueError("No DataFrame has been set. Use set_data() first.")
            
        # Calculate Z-scores
        z_scores = np.abs((self.clean_df[column] - self.clean_df[column].mean()) / 
                          self.clean_df[column].std())
        
        # Identify outliers
        outliers = self.clean_df[z_scores > threshold]
        
        # Create a clean DataFrame without outliers
        self.clean_df = self.clean_df[z_scores <= threshold]
        
        # Prepare outlier information
        outlier_info = {
            'method': 'Z-score',
            'column': column,
            'mean': self.original_df[column].mean(),
            'std': self.original_df[column].std(),
            'threshold': threshold,
            'num_outliers': len(outliers),
            'percent_removed': (len(outliers) / len(self.original_df)) * 100
        }
        
        # Store outlier information
        self.outlier_info[column] = outlier_info
        
        return self.clean_df, outlier_info
    
    def clean_columns(self, method='iqr', columns=None, **kwargs):
        """
        Clean multiple columns in a DataFrame by removing outliers.
        
        Parameters:
        -----------
        method : str, default='iqr'
            The method to use ('iqr' or 'zscore')
        columns : list or None, default=None
            List of columns to clean. If None, all numeric columns will be cleaned.
        **kwargs : 
            Additional parameters to pass to the outlier removal function
            
        Returns:
        --------
        pandas.DataFrame
            A DataFrame with outliers removed from specified columns
        dict
            Information about outliers removed from each column
        """
        if self.clean_df is None:
            raise ValueError("No DataFrame has been set. Use set_data() first.")
            
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self.clean_df.select_dtypes(include=np.number).columns.tolist()
        
        # Select the appropriate outlier removal function
        if method.lower() == 'iqr':
            outlier_func = self.remove_outliers_iqr
        elif method.lower() in ['zscore', 'z-score', 'z_score']:
            outlier_func = self.remove_outliers_zscore
        else:
            raise ValueError("Method must be either 'iqr' or 'zscore'")
        
        # Process each column
        for col in columns:
            if col not in self.clean_df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                continue
                
            if not np.issubdtype(self.clean_df[col].dtype, np.number):
                print(f"Warning: Column '{col}' is not numeric. Skipping.")
                continue
            
            # Apply the outlier removal function
            outlier_func(col, **kwargs)
        
        return self.clean_df, self.outlier_info
    
    def visualize_outliers(self, column):
        """
        Visualize the distribution of data and highlight outliers.
        
        Parameters:
        -----------
        column : str
            The name of the column to visualize
        """
        if self.original_df is None:
            raise ValueError("No DataFrame has been set. Use set_data() first.")
            
        if column not in self.outlier_info:
            raise ValueError(f"No outlier information found for column '{column}'. Run removal first.")
            
        outlier_info = self.outlier_info[column]
        
        plt.figure(figsize=(12, 6))
        
        # Create subplot for the boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(y=self.original_df[column])
        plt.title(f'Boxplot of {column}')
        
        # Create subplot for the histogram
        plt.subplot(1, 2, 2)
        sns.histplot(self.original_df[column], kde=True)
        
        if outlier_info['method'] == 'IQR':
            plt.axvline(outlier_info['lower_bound'], color='r', linestyle='--', 
                       label=f"Lower bound: {outlier_info['lower_bound']:.2f}")
            plt.axvline(outlier_info['upper_bound'], color='r', linestyle='--',
                       label=f"Upper bound: {outlier_info['upper_bound']:.2f}")
        else:  # Z-score
            # Calculate bounds for z-score method for visualization
            mean = outlier_info['mean']
            std = outlier_info['std']
            threshold = outlier_info['threshold']
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            plt.axvline(lower_bound, color='r', linestyle='--',
                       label=f"Lower bound: {lower_bound:.2f}")
            plt.axvline(upper_bound, color='r', linestyle='--',
                       label=f"Upper bound: {upper_bound:.2f}")
        
        plt.title(f'Distribution of {column} with Outlier Bounds')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        self._print_outlier_summary(column)
    
    def _print_outlier_summary(self, column):
        """
        Print a summary of outliers for a specific column.
        
        Parameters:
        -----------
        column : str
            The name of the column to summarize
        """
        if column not in self.outlier_info:
            raise ValueError(f"No outlier information found for column '{column}'")
            
        outlier_info = self.outlier_info[column]
        
        print(f"Outlier Summary ({outlier_info['method']} method):")
        print(f"- Column: {column}")
        print(f"- Number of outliers: {outlier_info['num_outliers']} ({outlier_info['percent_removed']:.2f}%)")
        if outlier_info['method'] == 'IQR':
            print(f"- Outliers below lower bound: {outlier_info['num_outliers_below']}")
            print(f"- Outliers above upper bound: {outlier_info['num_outliers_above']}")
    
    def get_summary_report(self):
        """
        Generate a summary report of all outlier removal operations.
        
        Returns:
        --------
        dict
            Summary report of all operations
        """
        if not self.outlier_info:
            return {"status": "No outlier removal operations performed yet"}
            
        total_rows_before = len(self.original_df)
        total_rows_after = len(self.clean_df)
        percent_removed = ((total_rows_before - total_rows_after) / total_rows_before) * 100
        
        summary = {
            "original_shape": self.original_df.shape,
            "clean_shape": self.clean_df.shape,
            "total_rows_removed": total_rows_before - total_rows_after,
            "percent_removed": percent_removed,
            "columns_processed": list(self.outlier_info.keys()),
            "column_details": self.outlier_info
        }
        
        return summary
    
    def reset(self):
        """
        Reset the cleaner to the original DataFrame.
        """
        if self.original_df is not None:
            self.clean_df = self.original_df.copy()
            self.outlier_info = {}


# Example usage:
def example():
    """
    Example demonstrating how to use the OutlierCleaner class.
    """
    # Create a sample DataFrame
    np.random.seed(42)
    data = {
        'normal_data': np.random.normal(0, 1, 1000),
        'skewed_data': np.random.exponential(2, 1000),
        'categorical': np.random.choice(['A', 'B', 'C'], 1000)
    }
    df = pd.DataFrame(data)
    
    # Add some outliers
    df.loc[0, 'normal_data'] = 15  # Add a high outlier
    df.loc[1, 'normal_data'] = -12  # Add a low outlier
    df.loc[2, 'skewed_data'] = 30  # Add a high outlier
    
    # Create an OutlierCleaner instance
    cleaner = OutlierCleaner(df)
    
    # Method 1: Clean a specific column using IQR
    print("Cleaning 'normal_data' with IQR method:")
    cleaner.remove_outliers_iqr('normal_data')
    cleaner.visualize_outliers('normal_data')
    
    # Reset to original data
    cleaner.reset()
    
    # Method 2: Clean a specific column using Z-score
    print("\nCleaning 'normal_data' with Z-score method:")
    cleaner.remove_outliers_zscore('normal_data', threshold=2.5)
    cleaner.visualize_outliers('normal_data')
    
    # Reset to original data
    cleaner.reset()
    
    # Method 3: Clean multiple columns at once
    print("\nCleaning multiple columns with IQR method:")
    cleaner.clean_columns(method='iqr', columns=['normal_data', 'skewed_data'])
    
    # Get a summary report
    report = cleaner.get_summary_report()
    print("\nSummary Report:")
    for key, value in report.items():
        if key != "column_details":
            print(f"- {key}: {value}")
            
    # Visualize the results for all processed columns
    for column in report["columns_processed"]:
        cleaner.visualize_outliers(column)


if __name__ == "__main__":
    example()