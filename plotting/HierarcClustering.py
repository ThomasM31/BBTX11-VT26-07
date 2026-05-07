import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Config:
SHAP_PATH = "/data/shared/alzgene26/data/figures/shap/real_shap_values.csv"
EXPR_PATH = "/data/shared/alzgene26/data/figures/shap/real_expression_values.csv"
RESULTS_PATH = "/data/users/lucasant/kandidatarbete/BBTX11-VT26-07/plotting/ModelResults/binn_test_results.csv"  # This file contains the true labels
SAVE_DIR = "figures/hierarchicalclustering"

os.makedirs(SAVE_DIR, exist_ok=True)
print("Starting Hierarchical Clustering Script!")
# Identify top 20 genes based on mean absolute SHAP impact
df_shap = pd.read_csv(SHAP_PATH)
top_20_proteins = df_shap.abs().mean().sort_values(ascending=False).head(20).index.tolist()

# Load protein expression and labels
df_expr_full = pd.read_csv(EXPR_PATH)
df_results = pd.read_csv(RESULTS_PATH) # Get labels from the results file
severity_labels = df_results['y_true'].map({1.0: 'More severe', 0.0: 'Less severe'})
print(f"Data loaded. Found {len(top_20_proteins)} top proteins.")

# Prepare data for plotting
df_top20 = df_expr_full[top_20_proteins]
severity_colors = severity_labels.map({'More severe': 'red', 'Less severe': 'blue'})

# Create long-format for boxplots
df_melted = df_top20.copy()
df_melted['Severity'] = severity_labels
df_long = df_melted.melt(id_vars='Severity', var_name='Protein', value_name='Quantity')

print("Generating plots...")
# Plot 1, Clustermap:
g = sns.clustermap(
    data=df_top20,
    row_colors=severity_colors,
    method='ward',
    metric='euclidean',
    z_score=1,            # Standardize across proteins
    cmap='coolwarm',
    figsize=(12, 12)
)
g.ax_heatmap.set_ylabel("Samples (Patients)")
g.ax_heatmap.set_xlabel("Top 20 Proteins")
plt.savefig(f"{SAVE_DIR}/clustermap_final.png", dpi=300)

# Plot 2, Boxplot & Presence:
fig, (ax_box, ax_bar) = plt.subplots(
    nrows=2, ncols=1, figsize=(14, 12), sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]}
)

# Upper panel: Boxplot with distribution points
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
plt.savefig(f"{SAVE_DIR}/boxplot_presence_final.png", dpi=300)
print(f"Done! Figures saved in {SAVE_DIR}")