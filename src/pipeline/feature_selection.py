from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation import numeric_columns


def select_numeric_features(df: pd.DataFrame) -> list[str]:
    return numeric_columns(df)


def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No numeric features available for variance filtering.")

    variances = numeric_df.var(numeric_only=True)
    selected = variances[variances > threshold].index.tolist()
    return df[selected].copy(), selected


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
) -> tuple[pd.DataFrame, list[str]]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No numeric features available for correlation filtering.")

    correlation_matrix = numeric_df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    selected_columns = [column for column in numeric_df.columns if column not in columns_to_drop]
    return df[selected_columns].copy(), selected_columns
