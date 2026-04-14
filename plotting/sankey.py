import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

def generate_binn_shap_data(n_cells=100):
    # 1. Define Hierarchy Dimensions
    genes = ['APOE', 'TREM2', 'BIN1', 'CLU', 'ABCA7', 'CD33']
    sub_pathways = ['Lipid_Metab', 'Immune_Resp', 'Endocytosis']
    main_pathways = ['Proteostasis', 'Neuroinflammation']
    
    # 2. Map Genes -> Sub-pathways (The Mask)
    # Value represents the "Global Importance" flow
    gene_to_sub = pd.DataFrame({
        'Source': ['APOE', 'ABCA7', 'TREM2', 'CD33', 'BIN1', 'CLU'],
        'Target': ['Lipid_Metab', 'Lipid_Metab', 'Immune_Resp', 'Immune_Resp', 'Endocytosis', 'Endocytosis'],
        'SHAP_Weight': np.random.uniform(-0.5, 2.0, 6) # Simulated global importance
    })

    # 3. Map Sub-pathways -> Main Pathways
    sub_to_main = pd.DataFrame({
        'Source': ['Lipid_Metab', 'Immune_Resp', 'Endocytosis'],
        'Target': ['Proteostasis', 'Neuroinflammation', 'Proteostasis'],
        'SHAP_Weight': np.random.uniform(-1.0, 3.0, 3)
    })

    # 4. Map Main Pathways -> Output
    main_to_out = pd.DataFrame({
        'Source': ['Proteostasis', 'Neuroinflammation'],
        'Target': ['Alzheimers_Status', 'Alzheimers_Status'],
        'SHAP_Weight': np.random.uniform(-2.0, 5.0, 2)
    })

    # Combine into a single "Flow" dataframe for Plotly Sankey
    sankey_data = pd.concat([gene_to_sub, sub_to_main, main_to_out], axis=0)
    
    return sankey_data


def draw_sankey(df, n_top=10, output_filename="sankey_plot.html"):
    # sort nodes by importance, highest first
    importance_map = {}
    all_unique_names = pd.unique(df[['Source', 'Target']].values.ravel('K'))

    for node in all_unique_names:
        in_vol = df[df['Target'] == node]['SHAP_Weight'].abs().sum()
        out_vol = df[df['Source'] == node]['SHAP_Weight'].abs().sum()
        importance_map[node] = in_vol + out_vol

    sorted_nodes = sorted(importance_map.items(), key=lambda x:x[1], reverse=True)

    all_nodes = [node[0] for node in sorted_nodes]
    top_n_names = [node[0] for node in sorted_nodes[:n_top]]
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    # colormaps
    norm = mcolors.TwoSlopeNorm(
        vmin=df['SHAP_Weight'].min(), 
        vcenter=0, 
        vmax=df['SHAP_Weight'].max()
    )
    cmap = plt.get_cmap('RdBu_r') # reverse color - red is pos value / high risk

    def get_rgba_color(val, is_important, is_link=False):
        if not is_important:
            return f'rgba(220, 220, 220, 0.3)' if is_link else 'rgba(180, 180, 180, 1.0)'
        
        rgb = cmap(norm(val))[:3]
        alpha = 0.4 if is_link else 1.0
        return f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})'

    node_colors = []
    for node in all_nodes:
        net_val = df[df['Target'] == node]['SHAP_Weight'].sum()
        if net_val == 0: net_val = df[df['Source'] == node]['SHAP_Weight'].mean()

        is_imp = node in top_n_names
        node_colors.append(get_rgba_color(net_val, is_imp, is_link=False))

    link_colors = []
    for _, row in df.iterrows():
        is_imp = (row['Source'] in top_n_names or row['Target'] in top_n_names)
        link_colors.append(get_rgba_color(row['SHAP_Weight'], is_imp, is_link=True))

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad=15, 
            thickness=20, 
            line= dict(color = "black", width = 0),
            label=all_nodes, 
            color=node_colors),
        link = dict(
            source = df['Source'].map(node_indices),
            target = df['Target'].map(node_indices),
            value  = df['SHAP_Weight'].abs(),
            color = link_colors
        )
    )])

    fig.update_layout(
        title_text="BINN SHAP Flow (Red: Increased AD risk | Blue: Protective)",
        font_size = 12,
        paper_bgcolor = 'white'
    )

    fig.write_html(output_filename)
    print(f"Interactive plot saved to {output_filename}.")

# Generate the data
df_flow = generate_binn_shap_data()
print(df_flow.head(10))

draw_sankey(df_flow, 3)