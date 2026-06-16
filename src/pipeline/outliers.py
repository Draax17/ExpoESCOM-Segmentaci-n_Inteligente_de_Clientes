from __future__ import annotations

import pandas as pd


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


def outlier_mask_iqr(df: pd.DataFrame, columns: list[str], multiplier: float = 1.5) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        lower, upper = iqr_bounds(df[column], multiplier=multiplier)
        mask = mask | df[column].lt(lower) | df[column].gt(upper)
    return mask


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], multiplier: float = 1.5) -> pd.DataFrame:
    mask = outlier_mask_iqr(df, columns, multiplier=multiplier)
    return df.loc[~mask].reset_index(drop=True)
