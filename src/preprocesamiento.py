# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

print("All libraries loaded successfully")

# Importing the dataset
df = pd.read_csv('/kaggle/input/datasets/imakash3011/customer-personality-analysis/marketing_campaign.csv', sep='\t')
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
plt.savefig('/kaggle/working/01_eda_overview.png', 
            dpi=150, bbox_inches='tight')
plt.show()
print("EDA chart saved")