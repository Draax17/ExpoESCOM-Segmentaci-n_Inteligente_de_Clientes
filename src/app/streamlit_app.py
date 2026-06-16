from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Segmentación Inteligente de Clientes",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH_1 = PROJECT_ROOT / "data" / "marketing_campaign.csv"
DATA_PATH_2 = PROJECT_ROOT / "data" / "marketing_campaign_kaggle.csv"


# ============================================================
# FUNCIONES
# ============================================================

@st.cache_data
def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file, sep="\t")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)

    if DATA_PATH_1.exists():
        return pd.read_csv(DATA_PATH_1, sep="\t")

    if DATA_PATH_2.exists():
        try:
            return pd.read_csv(DATA_PATH_2, sep="\t")
        except Exception:
            return pd.read_csv(DATA_PATH_2)

    raise FileNotFoundError("No se encontró el dataset en la carpeta data/.")


def iqr_outlier_summary(dataframe, columns):
    rows = []

    for col in columns:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = dataframe[(dataframe[col] < lower) | (dataframe[col] > upper)]

        rows.append({
            "Variable": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Límite inferior": round(lower, 2),
            "Límite superior": round(upper, 2),
            "Cantidad de outliers": len(outliers)
        })

    return pd.DataFrame(rows)


def preprocess_data(df):
    df = df.copy()

    duplicated_count = df.duplicated().sum()
    missing_total = df.isnull().sum().sum()

    df = df.drop_duplicates()
    df = df.dropna()

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Age"] = 2026 - df["Year_Birth"]
    df = df[df["Age"] < 100].copy()

    df["Total_Spent"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    df["Total_Purchases"] = (
        df["NumDealsPurchases"]
        + df["NumWebPurchases"]
        + df["NumCatalogPurchases"]
        + df["NumStorePurchases"]
    )

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Tenure"] = (pd.to_datetime("2026-01-01") - df["Dt_Customer"]).dt.days

    features = [
        "Income",
        "Age",
        "Total_Spent",
        "Total_Purchases",
        "Total_Children",
        "Tenure",
        "Recency",
        "NumWebVisitsMonth"
    ]

    outlier_summary = iqr_outlier_summary(df, features)

    df_clean = df[df["Income"] < 200000].copy()

    return df_clean, features, missing_total, duplicated_count, outlier_summary


def run_models(df_clean, features, k_final):
    X = df_clean[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = range(2, 11)

    results = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        results.append({
            "k": k,
            "Inercia": model.inertia_,
            "Silhouette": silhouette_score(X_scaled, labels),
            "Davies-Bouldin": davies_bouldin_score(X_scaled, labels),
            "Calinski-Harabasz": calinski_harabasz_score(X_scaled, labels)
        })

    metrics_df = pd.DataFrame(results)

    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    hierarchical = AgglomerativeClustering(n_clusters=k_final, linkage="ward")
    labels_hierarchical = hierarchical.fit_predict(X_scaled)

    df_result = df_clean.copy()
    df_result["Cluster_KMeans"] = labels_kmeans
    df_result["Cluster_Hierarchical"] = labels_hierarchical

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "PCA 1": X_pca[:, 0],
        "PCA 2": X_pca[:, 1],
        "Cluster K-Means": labels_kmeans.astype(str),
        "Cluster Jerárquico": labels_hierarchical.astype(str),
        "Income": df_clean["Income"].values,
        "Total_Spent": df_clean["Total_Spent"].values,
        "Total_Purchases": df_clean["Total_Purchases"].values
    })

    comparison = pd.DataFrame({
        "Modelo": ["K-Means", "Clustering Jerárquico"],
        "Número de clusters": [k_final, k_final],
        "Silhouette": [
            silhouette_score(X_scaled, labels_kmeans),
            silhouette_score(X_scaled, labels_hierarchical)
        ],
        "Davies-Bouldin": [
            davies_bouldin_score(X_scaled, labels_kmeans),
            davies_bouldin_score(X_scaled, labels_hierarchical)
        ],
        "Calinski-Harabasz": [
            calinski_harabasz_score(X_scaled, labels_kmeans),
            calinski_harabasz_score(X_scaled, labels_hierarchical)
        ]
    })

    profile_kmeans = df_result.groupby("Cluster_KMeans")[features].mean().round(2)
    profile_hierarchical = df_result.groupby("Cluster_Hierarchical")[features].mean().round(2)

    return metrics_df, comparison, profile_kmeans, profile_hierarchical, df_result, df_pca


# ============================================================
# INTERFAZ
# ============================================================

st.title("📊 Sistema de Segmentación Inteligente de Clientes")
st.write(
    "Esta aplicación permite cargar un dataset de clientes, realizar preprocesamiento, "
    "detectar outliers, aplicar K-Means y Clustering Jerárquico, y visualizar los segmentos obtenidos."
)

st.sidebar.header("Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Cargar dataset CSV",
    type=["csv"]
)

k_final = st.sidebar.slider(
    "Número de clusters para el modelo final",
    min_value=2,
    max_value=10,
    value=4,
    step=1
)

try:
    df_original = load_dataset(uploaded_file)
except Exception as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()

df_clean, features, missing_total, duplicated_count, outlier_summary = preprocess_data(df_original)
metrics_df, comparison, profile_kmeans, profile_hierarchical, df_result, df_pca = run_models(
    df_clean,
    features,
    k_final
)

# ============================================================
# SECCIÓN 1: RESUMEN DEL DATASET
# ============================================================

st.header("1. Resumen del dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros originales", df_original.shape[0])
col2.metric("Columnas originales", df_original.shape[1])
col3.metric("Valores faltantes", missing_total)
col4.metric("Duplicados encontrados", duplicated_count)

st.subheader("Primeras filas del dataset")
st.dataframe(df_original.head(), use_container_width=True)

st.subheader("Dataset limpio")
st.write(f"Después de eliminar faltantes, duplicados y edades no realistas, quedan **{df_clean.shape[0]} registros**.")
st.dataframe(df_clean[features].head(), use_container_width=True)


# ============================================================
# SECCIÓN 2: OUTLIERS
# ============================================================

st.header("2. Detección de outliers")

st.write("La detección de outliers se realizó mediante el método del rango intercuartílico, también conocido como IQR.")
st.dataframe(outlier_summary, use_container_width=True)


# ============================================================
# SECCIÓN 3: EXPLORACIÓN VISUAL
# ============================================================

st.header("3. Exploración visual de clientes")

col_a, col_b = st.columns(2)

fig_age = px.histogram(
    df_clean,
    x="Age",
    nbins=30,
    title="Distribución de edad de los clientes"
)
col_a.plotly_chart(fig_age, use_container_width=True)

fig_income = px.histogram(
    df_clean,
    x="Income",
    nbins=30,
    title="Distribución de ingresos"
)
col_b.plotly_chart(fig_income, use_container_width=True)

fig_spent = px.scatter(
    df_clean,
    x="Income",
    y="Total_Spent",
    title="Relación entre ingreso y gasto total",
    hover_data=["Age", "Total_Purchases", "Total_Children"]
)
st.plotly_chart(fig_spent, use_container_width=True)


# ============================================================
# SECCIÓN 4: MÉTODO DEL CODO Y MÉTRICAS
# ============================================================

st.header("4. Evaluación de K-Means")

st.subheader("Método del codo")

fig_elbow = px.line(
    metrics_df,
    x="k",
    y="Inercia",
    markers=True,
    title="Método del codo para seleccionar k"
)
st.plotly_chart(fig_elbow, use_container_width=True)

st.subheader("Métricas por número de clusters")
st.dataframe(metrics_df.round(4), use_container_width=True)

col_m1, col_m2, col_m3 = st.columns(3)

fig_sil = px.line(
    metrics_df,
    x="k",
    y="Silhouette",
    markers=True,
    title="Silhouette Score"
)
col_m1.plotly_chart(fig_sil, use_container_width=True)

fig_db = px.line(
    metrics_df,
    x="k",
    y="Davies-Bouldin",
    markers=True,
    title="Davies-Bouldin"
)
col_m2.plotly_chart(fig_db, use_container_width=True)

fig_ch = px.line(
    metrics_df,
    x="k",
    y="Calinski-Harabasz",
    markers=True,
    title="Calinski-Harabasz"
)
col_m3.plotly_chart(fig_ch, use_container_width=True)


# ============================================================
# SECCIÓN 5: RESULTADOS DE SEGMENTACIÓN
# ============================================================

st.header("5. Segmentación de clientes")

st.subheader("Visualización de clusters con PCA")

tab1, tab2 = st.tabs(["K-Means", "Clustering Jerárquico"])

with tab1:
    fig_kmeans = px.scatter(
        df_pca,
        x="PCA 1",
        y="PCA 2",
        color="Cluster K-Means",
        title="Clusters K-Means visualizados con PCA",
        hover_data=["Income", "Total_Spent", "Total_Purchases"]
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

    st.subheader("Perfil promedio de clusters K-Means")
    st.dataframe(profile_kmeans, use_container_width=True)

with tab2:
    fig_hier = px.scatter(
        df_pca,
        x="PCA 1",
        y="PCA 2",
        color="Cluster Jerárquico",
        title="Clusters Jerárquicos visualizados con PCA",
        hover_data=["Income", "Total_Spent", "Total_Purchases"]
    )
    st.plotly_chart(fig_hier, use_container_width=True)

    st.subheader("Perfil promedio de clusters jerárquicos")
    st.dataframe(profile_hierarchical, use_container_width=True)


# ============================================================
# SECCIÓN 6: COMPARACIÓN FINAL
# ============================================================

st.header("6. Comparación final de modelos")

st.dataframe(comparison.round(4), use_container_width=True)

best_model = comparison.sort_values(
    by=["Silhouette", "Calinski-Harabasz"],
    ascending=[False, False]
).iloc[0]

st.success(
    f"De acuerdo con las métricas calculadas, el modelo con mejor desempeño general es: "
    f"**{best_model['Modelo']}**."
)


# ============================================================
# SECCIÓN 7: DESCARGA DE RESULTADOS
# ============================================================

st.header("7. Descarga de resultados")

csv_segmentado = df_result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar clientes segmentados en CSV",
    data=csv_segmentado,
    file_name="clientes_segmentados.csv",
    mime="text/csv"
)

st.info(
    "La interfaz permite consultar el dataset limpio, revisar outliers, analizar el método del codo, "
    "comparar métricas y visualizar los segmentos finales de clientes."
)