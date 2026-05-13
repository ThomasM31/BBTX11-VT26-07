import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Config:
SHAP_PATH = "/data/shared/alzgene26/data/figures/shap/real_shap_values_ROSMAP_260513_1407.csv"
EXPR_PATH = "/data/shared/alzgene26/data/figures/shap/real_expression_values_ROSMAP_260513_1407.csv"
RESULTS_PATH = "/data/shared/alzgene26/data/results/binn_results/binn_test_results_260508_0940.csv" 
SAVE_DIR = "figures/hierarchicalclustering"

os.makedirs(SAVE_DIR, exist_ok=True)
print("Starting Hierarchical Clustering Script!")

# Load SHAP och identify top 20 proteins
df_shap = pd.read_csv(SHAP_PATH)
top_20_proteins = df_shap.abs().mean().sort_values(ascending=False).head(20).index.tolist()

df_expr_full = pd.read_csv(EXPR_PATH)
df_results = pd.read_csv(RESULTS_PATH) 

# Labels: 1.0 -> High (red), 0.0 -> Low (blue)
severity_labels = df_results['y_true'].map({1.0: 'Hög svårighetsgrad', 0.0: 'Låg svårighetsgrad'})
print(f"Data loaded. Found {len(top_20_proteins)} top proteins.")

# Colors: 
HUE_ORDER = ['Hög svårighetsgrad', 'Låg svårighetsgrad']
COLOR_PALETTE = ['red', 'blue']
DOT_PALETTE = ['darkred', 'darkblue']

# config clustermap
df_top20 = df_expr_full[top_20_proteins]
severity_colors = severity_labels.map({'Hög svårighetsgrad': 'red', 'Låg svårighetsgrad': 'blue'})

# config  boxplot (long-format)
df_melted = df_top20.copy()
df_melted['Svårighetsgrad'] = severity_labels
df_long = df_melted.melt(id_vars='Svårighetsgrad', var_name='Protein', value_name='Kvantitet')

print("Generating plots...")

# Clustermap:
g = sns.clustermap(
    data=df_top20,
    row_colors=severity_colors,
    method='ward',
    metric='euclidean',
    z_score=1,            # Z-score: z = (x - medel) / std
    cmap='coolwarm',
    figsize=(12, 12)
)
g.ax_heatmap.set_ylabel("Prover (Patienter)")
g.ax_heatmap.set_xlabel("Topp 20 Proteiner")
g.cax.set_title("Relativt uttryck\n(Z-score)", fontsize=10, pad=15)
g.savefig(f"{SAVE_DIR}/clustermap_final_swe_rosmap.png", dpi=300, bbox_inches='tight')

# Boxplot & Presence:
fig, (ax_box, ax_bar) = plt.subplots(
    nrows=2, ncols=1, figsize=(14, 12), sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]}
)

sns.boxplot(
    data=df_long, x='Protein', y='Kvantitet', hue='Svårighetsgrad',
    hue_order=HUE_ORDER, palette=COLOR_PALETTE,
    ax=ax_box, showfliers=False
)
sns.stripplot(
    data=df_long, x='Protein', y='Kvantitet', hue='Svårighetsgrad',
    hue_order=HUE_ORDER, palette=DOT_PALETTE,
    ax=ax_box, dodge=True, alpha=0.3, legend=False
)
ax_box.set_ylabel("Skalad proteinkvantitet (Z-score)")
ax_box.set_title("Distribution av de viktigaste proteinerna per svårighetsgrad")

# Underpanel: Närvaroandel
presence = df_melted.groupby('Svårighetsgrad').agg(lambda x: (x > 0).mean()).reset_index()
df_presence_long = presence.melt(id_vars='Svårighetsgrad', var_name='Protein', value_name='Andel')

sns.barplot(
    data=df_presence_long, x='Protein', y='Andel', hue='Svårighetsgrad',
    hue_order=HUE_ORDER, palette=COLOR_PALETTE, ax=ax_bar
)
ax_bar.set_ylabel("Närvaroandel")
ax_bar.set_ylim(0, 1.1)
ax_bar.legend_.remove()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/boxplot_presence_final_swe_rosmap.png", dpi=300, bbox_inches='tight')
print(f"Done! Figures saved in {SAVE_DIR}")