from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "marketing_campaign.csv"

if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / "data" / "marketing_campaign_kaggle.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("All libraries loaded successfully")


# ============================================================
# 1. CARGA DEL DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "No se encontró el dataset. Coloca marketing_campaign.csv en la carpeta data/"
    )

df = pd.read_csv(DATA_PATH, sep="\t")

print("\nDataset loaded successfully")
print(f"Dataset shape: {df.shape}")

print("\nColumn names:")
print(list(df.columns))

print("\nFirst rows:")
print(df.head())


# ============================================================
# 2. REVISIÓN DE CALIDAD DE DATOS
# ============================================================

print("\n================ DATA QUALITY CHECK ================")

# Valores faltantes
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])

print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Tipos de datos
print("\nData types:")
print(df.dtypes.value_counts())

# Duplicados
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")
else:
    print("No duplicate rows found.")

# Eliminar valores faltantes
df = df.dropna()
print(f"\nDataset shape after removing missing values: {df.shape}")


# ============================================================
# 3. TRANSFORMACIÓN Y CREACIÓN DE VARIABLES
# ============================================================

print("\n================ FEATURE ENGINEERING ================")

# Convertir fecha
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

# Edad del cliente
df["Age"] = 2026 - df["Year_Birth"]

# Eliminar edades no realistas
df = df[df["Age"] < 100].copy()

# Gasto total
df["Total_Spent"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)

# Total de compras
df["Total_Purchases"] = (
    df["NumDealsPurchases"]
    + df["NumWebPurchases"]
    + df["NumCatalogPurchases"]
    + df["NumStorePurchases"]
)

# Total de hijos en casa
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

# Antigüedad como cliente
df["Tenure"] = (pd.to_datetime("2026-01-01") - df["Dt_Customer"]).dt.days

print(f"\nClean dataset shape: {df.shape}")

print("\nNew features created:")
print(f"Age range: {df['Age'].min()} - {df['Age'].max()}")
print(f"Total Spent range: ${df['Total_Spent'].min()} - ${df['Total_Spent'].max()}")
print(f"Total Purchases range: {df['Total_Purchases'].min()} - {df['Total_Purchases'].max()}")
print(f"Tenure range: {df['Tenure'].min()} - {df['Tenure'].max()} days")


# ============================================================
# 4. EXPLORACIÓN VISUAL
# ============================================================

print("\n================ EDA VISUALIZATION ================")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Customer Profile Overview", fontsize=16, fontweight="bold")

# Distribución de edad
axes[0, 0].hist(
    df["Age"],
    bins=30,
    color="steelblue",
    edgecolor="white",
    linewidth=0.8
)
axes[0, 0].set_title("Customer Age Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Count")

# Distribución de gasto
axes[0, 1].hist(
    df["Total_Spent"],
    bins=30,
    color="coral",
    edgecolor="white",
    linewidth=0.8
)
axes[0, 1].set_title("Total Spending Distribution", fontweight="bold")
axes[0, 1].set_xlabel("Total Spent ($)")
axes[0, 1].set_ylabel("Count")

# Distribución de ingreso
axes[1, 0].hist(
    df["Income"],
    bins=30,
    color="mediumseagreen",
    edgecolor="white",
    linewidth=0.8
)
axes[1, 0].set_title("Income Distribution", fontweight="bold")
axes[1, 0].set_xlabel("Income ($)")
axes[1, 0].set_ylabel("Count")

# Ingreso vs gasto
axes[1, 1].scatter(
    df["Income"],
    df["Total_Spent"],
    alpha=0.4,
    color="steelblue",
    s=20
)
axes[1, 1].set_title("Income vs Total Spending", fontweight="bold")
axes[1, 1].set_xlabel("Income ($)")
axes[1, 1].set_ylabel("Total Spent ($)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()

print("EDA chart saved.")


# ============================================================
# 5. SELECCIÓN DE VARIABLES
# ============================================================

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

print("\n================ FEATURE SELECTION ================")
print(f"Features selected: {features}")


# ============================================================
# 6. DETECCIÓN DE OUTLIERS
# ============================================================

print("\n================ OUTLIER DETECTION ================")

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
            "feature": col,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outlier_count": len(outliers)
        })

    return pd.DataFrame(rows)

outlier_summary = iqr_outlier_summary(df, features)

print("\nOutlier summary using IQR method:")
print(outlier_summary.to_string(index=False))

outlier_summary.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

# Eliminamos outliers extremos de ingreso
df_clean = df[df["Income"] < 200000].copy()

print(f"\nCustomers before removing extreme income outliers: {len(df)}")
print(f"Customers after removing extreme income outliers: {len(df_clean)}")

X = df_clean[features].copy()


# ============================================================
# 7. GUARDAR DATASET LIMPIO
# ============================================================

clean_features_df = df_clean[features].copy()
clean_csv_path = OUTPUT_DIR / "marketing_campaign_clean_features.csv"
clean_features_df.to_csv(clean_csv_path, index=False)

print(f"\nClean feature dataset saved to: {clean_csv_path}")


# ============================================================
# 8. ESCALAMIENTO
# ============================================================

print("\n================ SCALING ================")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaling complete — mean: {X_scaled.mean():.2f}, std: {X_scaled.std():.2f}")


# ============================================================
# 9. K-MEANS: MÉTODO DEL CODO Y MÉTRICAS
# ============================================================

print("\n================ K-MEANS BASELINE ================")

inertias = []
silhouette_scores = []
db_scores = []
ch_scores = []

k_range = range(2, 11)

print("Testing different number of clusters...")

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

baseline_results_df = pd.DataFrame({
    "k": list(k_range),
    "inertia": inertias,
    "silhouette": silhouette_scores,
    "davies_bouldin": db_scores,
    "calinski_harabasz": ch_scores
})

print("\nBaseline clustering results:")
print(baseline_results_df.to_string(index=False))

baseline_results_df.to_csv(OUTPUT_DIR / "kmeans_metricas_por_k.csv", index=False)


# ============================================================
# 10. GRÁFICA DEL MÉTODO DEL CODO
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    baseline_results_df["k"],
    baseline_results_df["inertia"],
    marker="o"
)
plt.title("Método del Codo - K-Means")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_metodo_codo.png", dpi=150, bbox_inches="tight")
plt.show()

print("Elbow method chart saved.")


# ============================================================
# 11. GRÁFICAS DE MÉTRICAS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(
    baseline_results_df["k"],
    baseline_results_df["silhouette"],
    marker="o"
)
axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Silhouette")

axes[1].plot(
    baseline_results_df["k"],
    baseline_results_df["davies_bouldin"],
    marker="o",
    color="orange"
)
axes[1].set_title("Davies-Bouldin Index")
axes[1].set_xlabel("k")
axes[1].set_ylabel("DB Index")

axes[2].plot(
    baseline_results_df["k"],
    baseline_results_df["calinski_harabasz"],
    marker="o",
    color="green"
)
axes[2].set_title("Calinski-Harabasz Score")
axes[2].set_xlabel("k")
axes[2].set_ylabel("CH Score")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_metricas_clustering.png", dpi=150, bbox_inches="tight")
plt.show()

print("Clustering metrics charts saved.")


# ============================================================
# 12. MODELO FINAL K-MEANS
# ============================================================

print("\n================ FINAL K-MEANS MODEL ================")

# Nota:
# k=2 puede tener mejor Silhouette, pero k=4 da segmentos más útiles para marketing.
k_final = 4

kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
clusters_kmeans = kmeans_final.fit_predict(X_scaled)

df_clean["Cluster_KMeans"] = clusters_kmeans

print("\nCustomers per K-Means cluster:")
print(df_clean["Cluster_KMeans"].value_counts().sort_index())

print("\nK-Means Cluster Profiles:")
profile_kmeans = df_clean.groupby("Cluster_KMeans")[features].mean().round(2)
print(profile_kmeans)

kmeans_silhouette = silhouette_score(X_scaled, clusters_kmeans)
kmeans_db = davies_bouldin_score(X_scaled, clusters_kmeans)
kmeans_ch = calinski_harabasz_score(X_scaled, clusters_kmeans)

print("\nK-Means final model metrics:")
print(f"Silhouette Score: {kmeans_silhouette:.3f}")
print(f"Davies-Bouldin Index: {kmeans_db:.3f}")
print(f"Calinski-Harabasz Score: {kmeans_ch:.3f}")

df_clean.to_csv(OUTPUT_DIR / "clientes_segmentados_kmeans.csv", index=False)
profile_kmeans.to_csv(OUTPUT_DIR / "perfil_clusters_kmeans.csv")


# ============================================================
# 13. VISUALIZACIÓN K-MEANS CON PCA
# ============================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters_kmeans,
    cmap="viridis",
    alpha=0.6
)
plt.title("Clusters K-Means visualizados con PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_clusters_kmeans_pca.png", dpi=150, bbox_inches="tight")
plt.show()

print("K-Means PCA visualization saved.")


# ============================================================
# 14. CLUSTERING JERÁRQUICO
# ============================================================

print("\n================ HIERARCHICAL CLUSTERING ================")

hierarchical_model = AgglomerativeClustering(
    n_clusters=k_final,
    linkage="ward"
)

clusters_hierarchical = hierarchical_model.fit_predict(X_scaled)

df_clean["Cluster_Hierarchical"] = clusters_hierarchical

print("\nCustomers per Hierarchical cluster:")
print(df_clean["Cluster_Hierarchical"].value_counts().sort_index())

print("\nHierarchical Cluster Profiles:")
profile_hierarchical = df_clean.groupby("Cluster_Hierarchical")[features].mean().round(2)
print(profile_hierarchical)

hier_silhouette = silhouette_score(X_scaled, clusters_hierarchical)
hier_db = davies_bouldin_score(X_scaled, clusters_hierarchical)
hier_ch = calinski_harabasz_score(X_scaled, clusters_hierarchical)

print("\nHierarchical clustering metrics:")
print(f"Silhouette Score: {hier_silhouette:.3f}")
print(f"Davies-Bouldin Index: {hier_db:.3f}")
print(f"Calinski-Harabasz Score: {hier_ch:.3f}")

profile_hierarchical.to_csv(OUTPUT_DIR / "perfil_clusters_hierarchical.csv")
df_clean.to_csv(OUTPUT_DIR / "clientes_segmentados_final.csv", index=False)


# ============================================================
# 15. DENDROGRAMA
# ============================================================

print("\nGenerating dendrogram...")

sample_size = min(300, len(X_scaled))
rng = np.random.RandomState(42)
sample_indices = rng.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_indices]

linked = linkage(X_sample, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title("Dendrograma - Clustering Jerárquico")
plt.xlabel("Clientes")
plt.ylabel("Distancia")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_dendrograma.png", dpi=150, bbox_inches="tight")
plt.show()

print("Dendrogram saved.")


# ============================================================
# 16. COMPARACIÓN FINAL DE MODELOS
# ============================================================

print("\n================ MODEL COMPARISON ================")

comparison = pd.DataFrame({
    "Modelo": ["K-Means", "Clustering Jerárquico"],
    "Numero de clusters": [k_final, k_final],
    "Silhouette": [kmeans_silhouette, hier_silhouette],
    "Davies-Bouldin": [kmeans_db, hier_db],
    "Calinski-Harabasz": [kmeans_ch, hier_ch]
})

print("\nModel comparison:")
print(comparison.to_string(index=False))

comparison.to_csv(OUTPUT_DIR / "comparacion_modelos.csv", index=False)


# ============================================================
# 17. CONCLUSIÓN AUTOMÁTICA
# ============================================================

print("\n================ FINAL CONCLUSION ================")

best_model = comparison.sort_values(
    by=["Silhouette", "Calinski-Harabasz"],
    ascending=[False, False]
).iloc[0]

print(f"Best model according to Silhouette Score: {best_model['Modelo']}")

print("""
Conclusión:
Se realizó un proceso completo de segmentación de clientes utilizando técnicas de minería de datos.
Primero se revisó la calidad del dataset, incluyendo valores faltantes, duplicados y outliers.
Después se crearon variables relevantes para el análisis de comportamiento del cliente, como edad,
gasto total, compras totales, número de hijos y antigüedad.

Posteriormente, las variables fueron escaladas debido a que los algoritmos de clustering trabajan con distancias.
Se aplicó K-Means evaluando distintos valores de k mediante el método del codo, Silhouette Score,
Davies-Bouldin Index y Calinski-Harabasz Score. También se aplicó Clustering Jerárquico y se generó
un dendrograma para analizar la estructura de agrupamiento.

Finalmente, se compararon ambos métodos para identificar cuál ofrece una segmentación más adecuada
para apoyar estrategias de marketing.
""")

print("\nGenerated files saved in outputs/ folder:")
for file in OUTPUT_DIR.iterdir():
    print(f"- {file.name}")