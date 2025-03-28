# OutlierCleaner

A Python package for detecting and removing outliers in data using various statistical methods such as IQR and Z-score.

## Features

- Remove outliers using IQR (Interquartile Range) method
- Remove outliers using Z-score method
- Visualize outliers with boxplots and histograms
- Generate detailed reports on outlier removal
- Support for cleaning multiple columns at once
- Comprehensive documentation and examples

## Installation

```bash
pip install outlier-cleaner
```

## Usage

```python
from outlier_cleaner import OutlierCleaner
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {
    'normal_data': np.random.normal(0, 1, 1000),
    'skewed_data': np.random.exponential(2, 1000)
}
df = pd.DataFrame(data)

# Create an OutlierCleaner instance
cleaner = OutlierCleaner(df)

# Method 1: Clean using IQR method
cleaner.remove_outliers_iqr('normal_data')
cleaner.visualize_outliers('normal_data')

# Method 2: Clean using Z-score method
cleaner.reset()  # Reset to original data
cleaner.remove_outliers_zscore('normal_data', threshold=2.5)
cleaner.visualize_outliers('normal_data')

# Method 3: Clean multiple columns at once
cleaner.reset()
cleaner.clean_columns(method='iqr', columns=['normal_data', 'skewed_data'])

# Get a summary report
report = cleaner.get_summary_report()
print(report)
```

## Methods

### IQR Method
```python
cleaner.remove_outliers_iqr(column, lower_factor=1.5, upper_factor=1.5)
```

### Z-score Method
```python
cleaner.remove_outliers_zscore(column, threshold=3.0)
```

### Clean Multiple Columns
```python
cleaner.clean_columns(method='iqr', columns=None, **kwargs)
```

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.2.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Subashanan Nair

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 