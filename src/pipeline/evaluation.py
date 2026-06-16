from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def clustering_metrics(df: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    if labels.nunique() < 2:
        return {
            "silhouette_score": float("nan"),
            "calinski_harabasz_score": float("nan"),
            "davies_bouldin_score": float("nan"),
        }

    return {
        "silhouette_score": float(silhouette_score(df, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(df, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(df, labels)),
    }


def summarize_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    summary = df.copy()
    summary["cluster"] = labels.values
    return summary.groupby("cluster").agg(["mean", "median", "std"])


def best_score_from_report(report: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    if report.empty:
        return {"k": 0, score_column: float("nan")}

    best_row = report.sort_values(score_column, ascending=False).iloc[0]
    return {"k": int(best_row["k"]), score_column: float(best_row[score_column])}
