import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from pipeline.config import FEATURE_COLUMNS
from pipeline.config import EDA_OVERVIEW_PNG
from pipeline.data_loader import load_raw_dataset
from pipeline.hierarchical_model import run_hierarchical_workflow
from pipeline.kmeans_model import run_kmeans_workflow
from pipeline.preprocessing import (
    build_clustering_inputs,
    preprocess_customer_dataset,
    save_clean_feature_dataset,
)
from pipeline.visualization import create_eda_overview

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')


def main() -> None:
    raw_dataset, dataset_path = load_raw_dataset()
    print(f'Loaded dataset from: {dataset_path}')
    print(f'Dataset shape: {raw_dataset.shape}')
    print(f"\nColumn names:\n{list(raw_dataset.columns)}")

    print('Missing values per column:')
    missing = raw_dataset.isnull().sum()
    print(missing[missing > 0])
    print(f"\nTotal missing values: {raw_dataset.isnull().sum().sum()}")
    print(f"\nData types:\n{raw_dataset.dtypes.value_counts()}")

    cleaned_dataset = preprocess_customer_dataset(raw_dataset)
    print(f"Clean dataset shape: {cleaned_dataset.shape}")
    print('\nNew features created:')
    print(f"  Age range: {cleaned_dataset['Age'].min()} - {cleaned_dataset['Age'].max()}")
    print(
        f"  Total Spent range: ${cleaned_dataset['Total_Spent'].min()} - ${cleaned_dataset['Total_Spent'].max()}"
    )
    print(
        f"  Total Purchases range: {cleaned_dataset['Total_Purchases'].min()} - {cleaned_dataset['Total_Purchases'].max()}"
    )

    create_eda_overview(cleaned_dataset, EDA_OVERVIEW_PNG)

    df_clean, feature_frame = build_clustering_inputs(cleaned_dataset)
    clean_csv_path = save_clean_feature_dataset(feature_frame)
    print(f'Clean feature dataset saved to: {clean_csv_path}')
    print(f'Customers after removing outliers: {len(df_clean)}')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_frame)
    print(f'Features selected: {FEATURE_COLUMNS}')
    print(f"\nScaling complete — mean: {X_scaled.mean():.2f}, std: {X_scaled.std():.2f}")

    kmeans_results = run_kmeans_workflow(X_scaled, df_clean)
    print('\nK-means metrics:')
    print(kmeans_results['metrics_df'])
    print('\nK-means cluster profile:')
    print(kmeans_results['profile_df'])

    hierarchical_results = run_hierarchical_workflow(X_scaled, df_clean)
    print('\nHierarchical metrics:')
    print(hierarchical_results['metrics_df'])
    print('\nHierarchical cluster profile:')
    print(hierarchical_results['profile_df'])


if __name__ == '__main__':
    main()