import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


def generate_data():
    # 1. Define a larger hierarchy
    # Layer 1: Genes (Inputs)
    genes_risk = ['APOE', 'TREM2', 'BIN1', 'ABCA7', 'SORL1']
    genes_prot = ['CD33', 'CLU', 'PICALM', 'CR1', 'EPHA1']
    genes_noise = [f'GENE_{i}' for i in range(10)] # 10 low-impact genes
    
    # Layer 2: Sub-pathways
    sub_pathways = ['Lipid_Metabolism', 'Microglial_Activation', 'Endocytosis', 'Autophagy', 'Complement_System']
    
    # Layer 3: Main Pathways
    main_pathways = ['Neuroinflammation', 'Protein_Clearance']

    # 2. Map Genes -> Sub-pathways
    # We'll assign high absolute weights to 'known' genes and low weights to 'noise' genes
    g2s_rows = []
    for g in genes_risk:
        g2s_rows.append({'Source': g, 'Target': np.random.choice(sub_pathways[:2]), 'SHAP_Weight': np.random.uniform(2, 5)})
    for g in genes_prot:
        g2s_rows.append({'Source': g, 'Target': np.random.choice(sub_pathways[2:4]), 'SHAP_Weight': np.random.uniform(-4, -1)})
    for g in genes_noise:
        g2s_rows.append({'Source': g, 'Target': np.random.choice(sub_pathways), 'SHAP_Weight': np.random.uniform(-0.5, 0.5)})
    
    gene_to_sub = pd.DataFrame(g2s_rows)

    # 3. Map Sub-pathways -> Main Pathways
    s2m_rows = []
    # Most sub-pathways feed into Neuroinflammation or Protein Clearance
    for s in sub_pathways:
        target = 'Neuroinflammation' if 'Micro' in s or 'Comp' in s else 'Protein_Clearance'
        # Scale weight based on incoming gene weights (summing them roughly)
        s2m_rows.append({'Source': s, 'Target': target, 'SHAP_Weight': np.random.uniform(-3, 6)})
    
    sub_to_main = pd.DataFrame(s2m_rows)

    # 4. Map Main Pathways -> Output
    m2o_rows = [
        {'Source': 'Neuroinflammation', 'Target': 'Alzheimers_Status', 'SHAP_Weight': 7.5},
        {'Source': 'Protein_Clearance', 'Target': 'Alzheimers_Status', 'SHAP_Weight': -5.0}
    ]
    
    main_to_out = pd.DataFrame(m2o_rows)

    # Combine
    return pd.concat([gene_to_sub, sub_to_main, main_to_out], axis=0).reset_index(drop=True)

import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts

hv.extension('bokeh')

def plot_sankey(df, n_top=3, filename='sankey_diagram.html'):
    # 1. Setup absolute weights
    df = df.copy()
    df['Abs_Weight'] = df['SHAP_Weight'].abs()
    
    # 2. Dynamic Layer Detection
    all_nodes = set(df['Source']) | set(df['Target'])
    all_sources = set(df['Source'])
    all_targets = set(df['Target'])
    roots = [s for s in all_sources if s not in all_targets]
    
    layer_map = {node: 0 for node in roots}
    processed = False
    while not processed:
        processed = True
        for _, row in df.iterrows():
            if row['Source'] in layer_map:
                target_layer = layer_map[row['Source']] + 1
                if row['Target'] not in layer_map or layer_map[row['Target']] < target_layer:
                    layer_map[row['Target']] = target_layer
                    processed = False
    
    # 3. Apply Top-N Filtering & Grouping
    df['layer'] = df['Source'].map(layer_map)
    final_edges = []
    for layer in sorted(df['layer'].unique()):
        layer_df = df[df['layer'] == layer].copy()
        for target in layer_df['Target'].unique():
            target_group = layer_df[layer_df['Target'] == target].copy()
            target_group = target_group.sort_values('Abs_Weight', ascending=False)
            
            top_n = target_group.head(n_top)
            others = target_group.tail(len(target_group) - n_top)
            
            final_edges.append(top_n)
            if not others.empty:
                other_row = pd.DataFrame({
                    'Source': ['Other connections'],
                    'Target': [target],
                    'SHAP_Weight': [others['SHAP_Weight'].sum()],
                    'Abs_Weight': [others['Abs_Weight'].sum()]
                })
                final_edges.append(other_row)

    df_final = pd.concat(final_edges).reset_index(drop=True)

    # 4. Vertical Ordering Logic
    # We create a node dataset and sort it so 'Other connections' has a high sort-index
    s_imp = df_final.groupby('Source')['Abs_Weight'].sum()
    t_imp = df_final.groupby('Target')['Abs_Weight'].sum()
    node_imp = s_imp.add(t_imp, fill_value=0).reset_index()
    node_imp.columns = ['index', 'importance']
    
    # Identify the layer for each node for sorting purposes
    # A node's layer is either its Source layer or (if it's only a target) Source layer + 1
    node_layers = {}
    for _, row in df_final.iterrows():
        node_layers[row['Source']] = layer_map.get(row['Source'], 0)
        if row['Target'] not in node_layers:
            node_layers[row['Target']] = layer_map.get(row['Source'], 0) + 1

    node_imp['layer'] = node_imp['index'].map(node_layers)
    
    # Create a Sort Key: "Other connections" gets a value higher than any alphabetic name
    node_imp['sort_key'] = node_imp['index'].apply(lambda x: f"ZZZZ_{x}" if x == 'Other connections' else x)
    node_imp = node_imp.sort_values(['layer', 'sort_key']).reset_index(drop=True)
    
    # 5. Styling & Colors
    def get_edge_color(row):
        if row['Source'] == "Other connections": return '#D3D3D3'
        return '#EF553B' if row['SHAP_Weight'] > 0 else '#636EFA'

    df_final['edge_color'] = df_final.apply(get_edge_color, axis=1)
    
    # Neutralize 'Other' nodes importance so they don't dominate the cmap
    node_imp.loc[node_imp['index'] == 'Other connections', 'importance'] = 0
    nodes_ds = hv.Dataset(node_imp, 'index')

    # 6. Plot Generation
    sankey = hv.Sankey((df_final, nodes_ds), kdims=['Source', 'Target'], vdims=['Abs_Weight', 'edge_color'])

    sankey_obj = sankey.opts(
        opts.Sankey(
            width=1000, height=600,
            edge_color='edge_color', edge_alpha=0.6,
            node_color='importance', node_cmap='YlOrRd',
            label_position='outer', node_width=20, node_padding=20,
            iterations=0, # Crucial: Setting to 0 respects our dataframe's vertical order
            show_values=False
        )
    )

    hv.save(sankey_obj, filename, backend='bokeh')
    return sankey_obj

plot_sankey(generate_data(), n_top=3)