import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Config:
SHAP_PATH = "/data/shared/alzgene26/data/figures/shap/real_shap_values.csv"
EXPR_PATH = "/data/shared/alzgene26/data/figures/shap/real_expression_values.csv"
SAVE_DIR = "figures/hierarchicalclustering"
os.makedirs(SAVE_DIR, exist_ok=True)

# Identify top 20 genes using SHAP weights
df_shap = pd.read_csv(SHAP_PATH)
top_20_proteins = df_shap.abs().mean().sort_values(ascending=False).head(20).index.tolist()

# Load protein expression data
df_expr_full = pd.read_csv(EXPR_PATH)

# Filter for top 20 proteins
df_top20 = df_expr_full[top_20_proteins]

# Handle severity labels (AD Status)
if 'AD_status' in df_expr_full.columns:
    severity_labels = df_expr_full['AD_status'].map({1.0: 'More severe', 0.0: 'Less severe'})
else:
    # Fallback if status is missing from CSV
    severity_labels = pd.Series(['Unknown'] * len(df_top20))

# Setup colors for clustermap
severity_colors = severity_labels.map({'More severe': 'red', 'Less severe': 'blue'})

# Create long-format data for boxplots
df_melted = df_top20.copy()
df_melted['Severity'] = severity_labels
df_long = df_melted.melt(id_vars='Severity', var_name='Protein', value_name='Quantity')

# Plot 1, Clustermap:
g = sns.clustermap(
    data=df_top20,
    row_colors=severity_colors,
    method='ward',
    metric='euclidean',
    z_score=1,            # Standardize by protein
    cmap='coolwarm',
    figsize=(10, 10)
)
g.ax_heatmap.set_ylabel("Samples (Patients)")
g.ax_heatmap.set_xlabel("Top 20 Proteins")
plt.savefig(f"{SAVE_DIR}/clustermap_final.png")

# Plot 2, Boxplot & Presence Barplot:
fig, (ax_box, ax_bar) = plt.subplots(
    nrows=2, 
    ncols=1, 
    figsize=(12, 10), 
    sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]}
)

# Upper panel: Boxplot with individual data points
sns.boxplot(
    data=df_long, x='Protein', y='Quantity', hue='Severity',
    ax=ax_box, palette=['blue', 'red'], showfliers=False
)
sns.stripplot(
    data=df_long, x='Protein', y='Quantity', hue='Severity',
    ax=ax_box, dodge=True, alpha=0.3, palette=['darkblue', 'darkred'], legend=False
)
ax_box.set_ylabel("Scaled Protein Quantity")
ax_box.set_title("Top Proteins Distribution by Severity")

# Lower panel: Fraction of samples where protein quantity > 0
presence = df_melted.groupby('Severity').agg(lambda x: (x > 0).mean()).reset_index()
df_presence_long = presence.melt(id_vars='Severity', var_name='Protein', value_name='Fraction')

sns.barplot(
    data=df_presence_long, x='Protein', y='Fraction', hue='Severity',
    ax=ax_bar, palette=['blue', 'red']
)
ax_bar.set_ylabel("Presence Fraction")
ax_bar.set_ylim(0, 1.1)
ax_bar.legend_.remove()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/boxplot_presence_final.png")