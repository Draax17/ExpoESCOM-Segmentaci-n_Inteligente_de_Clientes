from pathlib import Path

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'marketing_campaign.csv'
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / 'data' / 'marketing_campaign_kaggle.csv'

OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("All libraries loaded successfully")

# Importing the dataset
df = pd.read_csv(DATA_PATH, sep='\t')
print("Dataset loaded successfully")

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")
df.head()

# Check for missing values
print("Missing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])

print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print(f"\nData types:")
print(df.dtypes.value_counts())

# Drop rows with missing values
df = df.dropna()

# Convert Dt_Customer to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

# Customer age
df['Age'] = 2026 - df['Year_Birth']

# Remove unrealistic age outliers
df = df[df['Age'] < 100].copy()

# Total spending across all product categories
df['Total_Spent'] = (
    df['MntWines'] + df['MntFruits'] +
    df['MntMeatProducts'] + df['MntFishProducts'] +
    df['MntSweetProducts'] + df['MntGoldProds']
)

# Total number of purchases
df['Total_Purchases'] = (
    df['NumDealsPurchases'] + df['NumWebPurchases'] +
    df['NumCatalogPurchases'] + df['NumStorePurchases']
)

# Total children at home
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# Customer tenure in days
df['Tenure'] = (pd.to_datetime('2026-01-01') - df['Dt_Customer']).dt.days

print(f"Clean dataset shape: {df.shape}")
print(f"\nNew features created:")
print(f"  Age range: {df['Age'].min()} - {df['Age'].max()}")
print(f"  Total Spent range: ${df['Total_Spent'].min()} - ${df['Total_Spent'].max()}")
print(f"  Total Purchases range: {df['Total_Purchases'].min()} - ${df['Total_Purchases'].max()}")

# Visual exploration of key features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Profile Overview', 
             fontsize=16, fontweight='bold')

# Chart 1: Age distribution
axes[0,0].hist(df['Age'], bins=30, color='steelblue', 
               edgecolor='white', linewidth=0.8)
axes[0,0].set_title('Customer Age Distribution', fontweight='bold')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Count')

# Chart 2: Total spending distribution
axes[0,1].hist(df['Total_Spent'], bins=30, color='coral', 
               edgecolor='white', linewidth=0.8)
axes[0,1].set_title('Total Spending Distribution', fontweight='bold')
axes[0,1].set_xlabel('Total Spent ($)')
axes[0,1].set_ylabel('Count')

# Chart 3: Income distribution
axes[1,0].hist(df['Income'], bins=30, color='mediumseagreen', 
               edgecolor='white', linewidth=0.8)
axes[1,0].set_title('Income Distribution', fontweight='bold')
axes[1,0].set_xlabel('Income ($)')
axes[1,0].set_ylabel('Count')

# Chart 4: Spending vs Income
axes[1,1].scatter(df['Income'], df['Total_Spent'], 
                  alpha=0.4, color='steelblue', s=20)
axes[1,1].set_title('Income vs Total Spending', fontweight='bold')
axes[1,1].set_xlabel('Income ($)')
axes[1,1].set_ylabel('Total Spent ($)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_eda_overview.png', 
            dpi=150, bbox_inches='tight')
plt.show()
print("EDA chart saved")

# Select features for clustering
# We use behavioural features — not demographic ones
features = ['Income', 'Age', 'Total_Spent', 'Total_Purchases', 
            'Total_Children', 'Tenure', 'Recency',
            'NumWebVisitsMonth']

# Extract selected features
X = df[features].copy()

# Remove outliers in Income (anyone earning more than 200k is an outlier)
X = X[X['Income'] < 200000]
df_clean = df[df['Income'] < 200000].copy()

clean_features_df = df_clean[features].copy()
clean_csv_path = OUTPUT_DIR / 'marketing_campaign_clean_features.csv'
clean_features_df.to_csv(clean_csv_path, index=False)
print(f"Clean feature dataset saved to: {clean_csv_path}")

print(f"Customers after removing outliers: {len(X)}")

# Scale the features — K-Means is distance based
# so all features need to be on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features selected: {features}")
print(f"\nScaling complete — mean: {X_scaled.mean():.2f}, std: {X_scaled.std():.2f}")

# Baseline: test different K values
inertias = []
silhouette_scores = []
db_scores = []
ch_scores = []
k_range = range(2, 11)

print("Testing different number of clusters for baseline model...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

baseline_results_df = pd.DataFrame({
    'k': list(k_range),
    'inertia': inertias,
    'silhouette': silhouette_scores,
    'davies_bouldin': db_scores,
    'calinski_harabasz': ch_scores
})

print("Baseline clustering results:")
display(baseline_results_df)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(baseline_results_df['k'], baseline_results_df['silhouette'], marker='o')
axes[0].set_title('Baseline Silhouette Score')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Silhouette')

axes[1].plot(baseline_results_df['k'], baseline_results_df['davies_bouldin'], marker='o', color='orange')
axes[1].set_title('Baseline Davies-Bouldin Index')
axes[1].set_xlabel('k')
axes[1].set_ylabel('DB Index')

axes[2].plot(baseline_results_df['k'], baseline_results_df['calinski_harabasz'], marker='o', color='green')
axes[2].set_title('Baseline Calinski-Harabasz Score')
axes[2].set_xlabel('k')
axes[2].set_ylabel('CH Score')

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_metricas-codo.png', 
            dpi=150, bbox_inches='tight')

# #K-means k=4
# # Train final K-Means model with k=4
# kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
# clusters = kmeans.fit_predict(X_scaled)

# # Add cluster labels back to dataframe
# df_clean = df_clean[df_clean['Income'] < 200000].copy()
# df_clean['Cluster'] = clusters

# # Count customers in each cluster
# print("Customers per cluster:")
# print(df_clean['Cluster'].value_counts().sort_index())

# # Profile each cluster
# print("\nCluster Profiles:")
# profile = df_clean.groupby('Cluster')[features].mean().round(2)
# print(profile)


# # Baseline evaluation metrics
# baseline_silhouette = silhouette_score(X_scaled, clusters)
# baseline_db = davies_bouldin_score(X_scaled, clusters)
# baseline_ch = calinski_harabasz_score(X_scaled, clusters)

# print("\nBaseline final model metrics (k=4):")
# print(f"Silhouette Score: {baseline_silhouette:.3f}")
# print(f"Davies-Bouldin Index: {baseline_db:.3f}")
# print(f"Calinski-Harabasz Score: {baseline_ch:.3f}")


# # Baseline evaluation metrics
# baseline_silhouette = silhouette_score(X_scaled, clusters)
# baseline_db = davies_bouldin_score(X_scaled, clusters)
# baseline_ch = calinski_harabasz_score(X_scaled, clusters)

# print("\nBaseline final model metrics (k=4):")
# print(f"Silhouette Score: {baseline_silhouette:.3f}")
# print(f"Davies-Bouldin Index: {baseline_db:.3f}")
# print(f"Calinski-Harabasz Score: {baseline_ch:.3f}")

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# axes[0].plot(baseline_results_df['k'], baseline_results_df['silhouette'], marker='o')
# axes[0].set_title('Baseline Silhouette Score')
# axes[0].set_xlabel('k')
# axes[0].set_ylabel('Silhouette')

# axes[1].plot(baseline_results_df['k'], baseline_results_df['davies_bouldin'], marker='o', color='orange')
# axes[1].set_title('Baseline Davies-Bouldin Index')
# axes[1].set_xlabel('k')
# axes[1].set_ylabel('DB Index')

# axes[2].plot(baseline_results_df['k'], baseline_results_df['calinski_harabasz'], marker='o', color='green')
# axes[2].set_title('Baseline Calinski-Harabasz Score')
# axes[2].set_xlabel('k')
# axes[2].set_ylabel('CH Score')

# plt.tight_layout()
# plt.show()