# OutlierCleaner

A Python package for detecting and removing outliers in data using various statistical methods.

## Features

- Clean outliers using IQR (Interquartile Range) method
- Clean outliers using Z-score method
- Add Z-score columns for analysis
- Clean multiple columns using pre-calculated Z-scores
- Batch clean all Z-score columns at once
- Advanced outlier analysis and statistics
- Comprehensive visualization tools
- Method comparison and agreement analysis

## Installation

```bash
pip install outlier-cleaner
```

## Usage

```python
import pandas as pd
from outlier_cleaner import OutlierCleaner

# Create sample data
df = pd.DataFrame({
    'height': [170, 175, 160, 180, 250, 165, 170],  # 250 is an outlier
    'weight': [70, 75, 60, 80, 180, 65, 72]  # 180 is an outlier
})

# Initialize cleaner
cleaner = OutlierCleaner(df)

# Get comprehensive outlier statistics
stats = cleaner.get_outlier_stats(['height', 'weight'])
print(stats['height']['iqr']['potential_outliers'])  # Number of potential outliers
print(stats['height']['zscore']['outlier_indices'])  # Indices of outliers

# Visualize outlier analysis
figures = cleaner.plot_outlier_analysis(['height', 'weight'])
# figures['height'].show()  # Display the figure for height

# Compare different outlier detection methods
comparison = cleaner.compare_methods(['height', 'weight'])
print(comparison['height']['summary'])  # Print comparison summary

# Add Z-score columns
cleaner.add_zscore_columns()

# Clean all columns with Z-scores at once
cleaned_df, outlier_info = cleaner.clean_zscore_columns(threshold=3.0)

# Clean specific columns using IQR method
cleaned_df, outlier_info = cleaner.remove_outliers_iqr('height', lower_factor=1.5, upper_factor=1.5)
```

## Methods

### get_outlier_stats(columns=None, methods=['iqr', 'zscore'])
Get comprehensive statistics about potential outliers without removing them.
- Returns detailed statistics for each column and method
- Includes counts, percentages, and indices of outliers

### plot_outlier_analysis(columns=None, methods=['iqr', 'zscore'])
Create comprehensive visualizations for outlier analysis.
- Generates box plots, distributions, and Z-score plots
- Shows outlier thresholds and boundaries
- Returns dictionary of matplotlib figures

### compare_methods(columns=None, methods=['iqr', 'zscore'])
Compare different outlier detection methods and their agreement.
- Calculates agreement percentage between methods
- Identifies common outliers and method-specific outliers
- Provides detailed comparison summary

### add_zscore_columns(columns=None)
Add Z-score columns to the DataFrame for specified columns.
- Creates new columns with '_zscore' suffix
- Useful for outlier detection and analysis

### clean_zscore_columns(threshold=3.0)
Clean all columns that have associated Z-score columns.
- Removes outliers based on Z-score threshold
- Returns cleaned DataFrame and outlier information

### remove_outliers_iqr(column, lower_factor=1.5, upper_factor=1.5)
Remove outliers using the IQR method.
- Configurable factors for lower and upper bounds
- Returns cleaned DataFrame and outlier information

### remove_outliers_zscore(column, threshold=3.0)
Remove outliers using the Z-score method.
- Uses existing Z-score columns if available
- Returns cleaned DataFrame and outlier information

## Requirements

- numpy
- pandas
- matplotlib
- seaborn

## Author

Subashan Annai

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 