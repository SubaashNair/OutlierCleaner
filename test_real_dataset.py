import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from outlier_cleaner import OutlierCleaner

# Load the California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target variable
df['PRICE'] = housing.target

print("Original Dataset Shape:", df.shape)
print("\nFeatures:", housing.feature_names)
print("\nFeature Descriptions:")
for name, desc in zip(housing.feature_names, housing.feature_names):
    print(f"- {name}")

# Test index preservation
print("\nTesting index preservation...")
# Create a non-sequential index
df.index = range(100, len(df) + 100)

# Initialize OutlierCleaner with index preservation
cleaner_preserve = OutlierCleaner(df, preserve_index=True)
cleaner_reset = OutlierCleaner(df, preserve_index=False)

# First, let's analyze the distribution of each column
print("\nAnalyzing distributions...")
for column in df.columns:
    analysis = cleaner_preserve.analyze_distribution(column)
    print(f"\n{column}:")
    print(f"- Skewness: {analysis['skewness']:.2f}")
    print(f"- Kurtosis: {analysis['kurtosis']:.2f}")
    print(f"- Normal distribution: {'Yes' if analysis['is_normal'] else 'No'}")
    print(f"- Recommended method: {analysis['recommended_method']}")
    if analysis['recommended_method'] == 'iqr':
        print(f"- Recommended thresholds: lower={analysis['recommended_threshold']['lower_factor']}, "
              f"upper={analysis['recommended_threshold']['upper_factor']}")
    else:
        print(f"- Recommended threshold: {analysis['recommended_threshold']}")

# Let's analyze a few interesting columns with both methods
interesting_columns = ['MedInc', 'AveRooms', 'PRICE']  # Median income, average rooms, and house price

# Clean with automatic method selection and preserved index
print("\nCleaning data with preserved index...")
cleaned_preserve, info_preserve = cleaner_preserve.clean_columns(
    method='auto', 
    columns=interesting_columns,
    show_progress=True
)
print("\nPreserved Index Cleaning Summary:")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_preserve.shape}")
print("Index preserved:", cleaned_preserve.index.min() >= 100)

# Clean with automatic method selection and reset index
print("\nCleaning data with reset index...")
cleaned_reset, info_reset = cleaner_reset.clean_columns(
    method='auto', 
    columns=interesting_columns,
    show_progress=True
)
print("\nReset Index Cleaning Summary:")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_reset.shape}")
print("Index reset:", cleaned_reset.index.min() == 0)

# Test outlier indices tracking
print("\nOutlier Indices Summary:")
for column in interesting_columns:
    outlier_indices = cleaner_preserve.get_outlier_indices(column)
    print(f"\n{column}:")
    print(f"Number of outliers: {len(outlier_indices[column])}")
    print(f"First few outlier indices: {outlier_indices[column][:5]}")

# Test zero MAD handling
print("\nTesting zero MAD handling...")
# Create a column with many repeated values (potential zero MAD)
df_test = pd.DataFrame({
    'constant': [1] * 95 + [100] * 5,  # Most values are 1, with a few outliers
    'normal': np.random.normal(0, 1, 100)
})
cleaner_test = OutlierCleaner(df_test)
cleaned_test, info_test = cleaner_test.clean_columns(
    method='modified_zscore',
    columns=['constant', 'normal']
)
print("\nZero MAD Test Results:")
print(f"Original shape: {df_test.shape}")
print(f"Cleaned shape: {cleaned_test.shape}")

# Compare distributions before and after cleaning
plt.figure(figsize=(15, 5))
for i, column in enumerate(interesting_columns):
    plt.subplot(1, 3, i+1)
    sns.kdeplot(data=df, x=column, label='Original', alpha=0.5)
    sns.kdeplot(data=cleaned_preserve, x=column, label='Preserved Index', alpha=0.5)
    sns.kdeplot(data=cleaned_reset, x=column, label='Reset Index', alpha=0.5)
    plt.title(f'{column} Distribution')
    plt.legend()

plt.tight_layout()
plt.show() 