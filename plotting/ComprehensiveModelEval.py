# OBS - outputted/created figures are only placeholders for now. Data is currently randomized/pseudo. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data input:
# ---------------------------------------------------------------------
# FAKE DATA BLOCK (DELETE THIS WHEN WE HAVE REAL DATA)
np.random.seed(42)
models = ['Pseudobulk', 'CT Prop', 'Cell-level', 'scAGG', 'scAGG+GAT', 'scAGG+AP', 'scAGG+GAT+AP', 'scRAT']

# 1. Global Metrics (Panels a-e)
global_data = []
for m in models:
    for met in ['Precision', 'Recall', 'Accuracy', 'F1', 'AUC']:
        global_data.append({'Model': m, 'Metric': met, 'Score': np.random.uniform(0.5, 0.95), 'Error': np.random.uniform(0.02, 0.08)})
df_global = pd.DataFrame(global_data)

# 2. Dataset Specific Metrics (Panels f-g)
dataset_data = []
for m in models:
    for ds in ['SeaAD', 'COMBAT']:
        dataset_data.append({'Model': m, 'Dataset': ds, 'AUC': np.random.uniform(0.4, 0.99), 'Error': np.random.uniform(0.02, 0.05)})
df_datasets = pd.DataFrame(dataset_data)

# 3. Cell Type Specific Data (Panels h-i)
cell_types = ['All', ' Neuronal', '  Excitatory', '  Inhibitory', ' Non-neuronal', '  Astrocytes', '  Immune Cells', '  Oligodendrocytes', '  OPCs', '  Vasculature']
df_cell = pd.DataFrame({
    'CellType': cell_types,
    'AUC': np.random.uniform(0.5, 0.95, len(cell_types)),
    'AUC_Error': np.random.uniform(0.05, 0.1, len(cell_types)),
    'Abundance': np.random.uniform(0.01, 1.0, len(cell_types)),
    'Abundance_Error': np.random.uniform(0.01, 0.05, len(cell_types))
})
df_cell.loc[0, 'Abundance'] = 1.0 # "All" is always 100%
df_cell.loc[0, 'Abundance_Error'] = 0.0
# END OF FAKE DATA BLOCK
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# REAL DATA BLOCK
# Load real results from saved CSV files later:
#
# df_global = pd.read_csv("path/to/global_metrics_results.csv")
# df_datasets = pd.read_csv("path/to/dataset_specific_results.csv")
# df_cell = pd.read_csv("path/to/cell_type_results.csv")
# 
# Make sure our CSV files have the same column names as used below!
# END OF REAL DATA BLOCK
# ---------------------------------------------------------------------


# Plotting the dashboard:
fig = plt.figure(figsize=(20, 10))

# We use GridSpec to create a custom layout
# Top row for a-e (5 columns)
gs_top = fig.add_gridspec(nrows=1, ncols=5, top=0.92, bottom=0.55, wspace=0.3)
# Bottom row for f-i (4 columns)
gs_bottom = fig.add_gridspec(nrows=1, ncols=4, top=0.45, bottom=0.08, wspace=0.4)

metrics = ['Precision', 'Recall', 'Accuracy', 'F1', 'AUC']
letters_top = ['a)', 'b)', 'c)', 'd)', 'e)']

# --- Plot Top Row (a-e): Global Metrics ---
for i, met in enumerate(metrics):
    ax = fig.add_subplot(gs_top[0, i])
    subset = df_global[df_global['Metric'] == met]
    
    sns.barplot(data=subset, x='Score', y='Model', ax=ax, hue='Model', palette='Set2', legend=False)
    ax.errorbar(x=subset['Score'], y=range(len(models)), xerr=subset['Error'], fmt='none', c='black', capsize=3)
    
    ax.set_title(f"{letters_top[i]} {met}")
    ax.set_xlim(0.4, 1.0) # Adjust based on our real data later
    ax.set_xlabel('')
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    
    if i > 0: 
        ax.set_ylabel('')
        ax.set_yticks([]) # Hide Y-axis text for cleaner look on middle plots

# --- Plot Bottom Row Left (f-g): Datasets ---
datasets = ['SeaAD', 'COMBAT']
letters_ds = ['f)', 'g)']
for i, ds in enumerate(datasets):
    ax = fig.add_subplot(gs_bottom[0, i])
    subset = df_datasets[df_datasets['Dataset'] == ds]
    
    sns.barplot(data=subset, x='AUC', y='Model', ax=ax, hue='Model', palette='Set2', legend=False)
    ax.errorbar(x=subset['AUC'], y=range(len(models)), xerr=subset['Error'], fmt='none', c='black', capsize=3)
    
    ax.set_title(f"{letters_ds[i]} {ds}")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('AUC')
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    if i > 0:
        ax.set_ylabel('')
        ax.set_yticks([])

# --- Plot Bottom Row Right (h-i): Cell Types ---
# h) Cell Type AUC
ax_h = fig.add_subplot(gs_bottom[0, 2])
sns.barplot(data=df_cell, x='AUC', y='CellType', ax=ax_h, color='#66c2a5')
ax_h.errorbar(x=df_cell['AUC'], y=range(len(cell_types)), xerr=df_cell['AUC_Error'], fmt='none', c='black', capsize=3)
ax_h.set_title("h) Cell-type AUC")
ax_h.set_xlim(0.5, 1.0)
ax_h.set_xlabel('AUC')
ax_h.set_ylabel('')
ax_h.xaxis.grid(True, linestyle='--', alpha=0.6)

# i) Cell Type Abundance
ax_i = fig.add_subplot(gs_bottom[0, 3])
sns.barplot(data=df_cell, x='Abundance', y='CellType', ax=ax_i, color='#66c2a5')
ax_i.errorbar(x=df_cell['Abundance'], y=range(len(cell_types)), xerr=df_cell['Abundance_Error'], fmt='none', c='black', capsize=3)
ax_i.set_title("i) Abundance")
ax_i.set_xlim(0.0, 1.05)
ax_i.set_xlabel('Abundance')
ax_i.set_ylabel('')
ax_i.set_yticks([]) # Share y-axis visually with h)
ax_i.xaxis.grid(True, linestyle='--', alpha=0.6)

# Save the final masterpiece
plt.savefig("figures/comprehensiveeval/placeholder_comprehensive_eval.png", bbox_inches='tight', dpi=300)
# plt.show() # Remember to leave this commented out for Minerva!
