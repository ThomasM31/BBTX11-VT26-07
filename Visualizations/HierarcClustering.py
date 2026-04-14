import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data input:

# ------------------------
# Made up / placeholder data:  (DELETE THIS ENTIRE BLOCK WHEN YOU HAVE REAL DATA)
np.random.seed(42) # Keeps the random numbers the same every run
n_patients = 100
n_proteins = 20
# Simulate patients with different severity
data_less_severe = np.random.normal(loc=5, scale=2, size=(50, n_proteins))
data_more_severe = np.random.normal(loc=10, scale=3, size=(50, n_proteins))
# Combine the data into a 2D matrix
matrix_2d = np.vstack([data_less_severe, data_more_severe])
# Randomly set some values to 0 (simulates missing proteins)
zero_mask = np.random.rand(n_patients, n_proteins) < 0.15 
matrix_2d[zero_mask] = 0
# Convert to Pandas format
protein_names = [f"Protein_{i}" for i in range(1, 21)]
df_top20 = pd.DataFrame(matrix_2d, columns=protein_names)
severity_labels = pd.Series(['Less severe']*50 + ['More severe']*50)
# END OF FAKE DATA BLOCK
# ------------------------


# ------------------------
# REAL DATA BLOCK - for later:
# Load your real SHAP data from a CSV file.
# df_real_data = pd.read_csv("path/to/your/real_shap_data.csv")
#
# Save the severity column to 'severity_labels'
# severity_labels = df_real_data['Patient_Status'] 
#
# Keep ONLY the 20 protein columns in 'df_top20'
# df_top20 = df_real_data.drop(columns=['Patient_Status']) 
# END OF REAL DATA BLOCK
# ------------------------


# Data preparation:
# Create a color map for the clustermap (Red for More Severe, Blue for Less)
severity_colors = severity_labels.map({'More severe': 'red', 'Less severe': 'blue'})

# For the boxplot, we need "long-format" data (melted)
df_melted = df_top20.copy()
df_melted['Severity'] = severity_labels
df_long = df_melted.melt(id_vars='Severity', var_name='Protein', value_name='Quantity')


# Clustermap:
# method='ward' and metric='euclidean' match your reference image
g = sns.clustermap(
    data=df_top20,
    row_colors=severity_colors,
    method='ward',
    metric='euclidean',
    z_score=1,            # Normalize data by protein
    cmap='coolwarm',      # Blue-White-Red scale
    figsize=(10, 10)
)

# Add simple labels
g.ax_heatmap.set_ylabel("Samples (Patients)")
g.ax_heatmap.set_xlabel("Top 20 Proteins")
plt.savefig("figures/placeholder_clustermap.png")
# plt.show() # Only use this if you run it locally on your laptop


# =Boxplot & presence barplot:
# Create a figure with two rows (Top row 3x larger than bottom row)
fig, (ax_box, ax_bar) = plt.subplots(
    nrows=2, 
    ncols=1, 
    figsize=(12, 10), 
    sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]}
)

# Top Panel: Boxplot with individual points (stripplot)
sns.boxplot(
    data=df_long, x='Protein', y='Quantity', hue='Severity',
    ax=ax_box, palette=['blue', 'red'], showfliers=False
)
sns.stripplot(
    data=df_long, x='Protein', y='Quantity', hue='Severity',
    ax=ax_box, dodge=True, alpha=0.3, palette=['darkblue', 'darkred'], legend=False
)
ax_box.set_ylabel("Scaled Protein Quantity")
ax_box.set_title("Placeholder: Top Proteins Distribution by Severity")

# Bottom Panel: Calculate and plot presence fraction (data > 0)
presence = df_melted.groupby('Severity').agg(lambda x: (x > 0).mean()).reset_index()
df_presence_long = presence.melt(id_vars='Severity', var_name='Protein', value_name='Fraction')

sns.barplot(
    data=df_presence_long, x='Protein', y='Fraction', hue='Severity',
    ax=ax_bar, palette=['blue', 'red']
)
ax_bar.set_ylabel("Presence Fraction")
ax_bar.set_ylim(0, 1.1)
ax_bar.legend_.remove() # Remove redundant legend

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("figures/placeholder_boxplot.png")