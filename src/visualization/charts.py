from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def elbow_chart(report: pd.DataFrame) -> go.Figure:
    figure = px.line(report, x="k", y="inertia", markers=True, title="Metodo del codo")
    figure.update_layout(template="plotly_white")
    return figure


def silhouette_chart(report: pd.DataFrame) -> go.Figure:
    figure = px.line(report, x="k", y="silhouette_score", markers=True, title="Silhouette Score")
    figure.update_layout(template="plotly_white")
    return figure


def cluster_scatter(df: pd.DataFrame, x_column: str, y_column: str, cluster_column: str) -> go.Figure:
    figure = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=cluster_column,
        title="Segmentacion de clientes",
        opacity=0.8,
    )
    figure.update_layout(template="plotly_white")
    return figure
