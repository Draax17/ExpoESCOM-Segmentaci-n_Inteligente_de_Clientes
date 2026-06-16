from __future__ import annotations

import pandas as pd


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    duplicated_rows = df.duplicated().sum()
    return {
        "total_rows": len(df),
        "duplicate_rows": int(duplicated_rows),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)
