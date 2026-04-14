import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

import pandas as pd
import numpy as np

def generate_complex_binn_data():
    # 1. Define a larger hierarchy
    # Layer 1: Genes (Inputs)
    genes_risk = ['APOE', 'TREM2', 'BIN1', 'ABCA7', 'SORL1']
    genes_prot = ['CD33', 'CLU', 'PICALM', 'CR1', 'EPHA1']
    genes_noise = [f'GENE_{i}' for i in range(10)] # 10 low-impact genes
    
    # Layer 2: Sub-pathways
    sub_pathways = ['Lipid_Metabolism', 'Microglial_Activation', 'Endocytosis', 'Autophagy', 'Complement_System']
    
    # Layer 3: Main Pathways
    main_pathways = ['Neuroinflammation', 'Protein_Clearance', 'Synaptic_Function']

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
        {'Source': 'Protein_Clearance', 'Target': 'Alzheimers_Status', 'SHAP_Weight': -5.0},
        {'Source': 'Synaptic_Function', 'Target': 'Alzheimers_Status', 'SHAP_Weight': 1.2}
    ]
    
    main_to_out = pd.DataFrame(m2o_rows)

    # Combine
    return pd.concat([gene_to_sub, sub_to_main, main_to_out], axis=0).reset_index(drop=True)


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

def draw_sankey(df, n_top=3, output_filename="flexible_binn.html"):
    # 1. Build a Graph to find layers automatically
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['SHAP_Weight'])

    # Assign layers based on shortest path from any root (gene)
    roots = [n for n, d in G.in_degree() if d == 0]
    layers_dict = {}
    for root in roots:
        lengths = nx.single_source_shortest_path_length(G, root)
        for node, length in lengths.items():
            layers_dict[node] = max(layers_dict.get(node, 0), length)
    
    max_layer = max(layers_dict.values())

    # 2. Identify Top N per Layer and Mask "Others"
    new_rows = []
    top_nodes_all_layers = set()
    
    for layer_idx in range(max_layer): # We don't mask the final output layer
        layer_nodes = [n for n, l in layers_dict.items() if l == layer_idx]
        
        # Rank by absolute volume
        vol_map = {n: df[(df['Source']==n) | (df['Target']==n)]['SHAP_Weight'].abs().sum() for n in layer_nodes}
        top_in_layer = sorted(vol_map, key=vol_map.get, reverse=True)[:n_top]
        top_nodes_all_layers.update(top_in_layer)

    # 3. Rewrite DataFrame with "Other" grouping
    for _, row in df.iterrows():
        s, t, w = row['Source'], row['Target'], row['SHAP_Weight']
        
        if s not in top_nodes_all_layers and layers_dict[s] < max_layer:
            s = f"Other Layer {layers_dict[s]}"
        if t not in top_nodes_all_layers and layers_dict[t] < max_layer:
            t = f"Other Layer {layers_dict[t]}"
            
        new_rows.append({'Source': s, 'Target': t, 'SHAP_Weight': w})
    
    masked_df = pd.DataFrame(new_rows).groupby(['Source', 'Target']).sum().reset_index()

    # 4. Sorting for Bottom-Heavy Layout
    # Separate Principal nodes from 'Other' nodes
    nodes_in_masked = pd.unique(masked_df[['Source', 'Target']].values.ravel('K'))
    principals = [n for n in nodes_in_masked if "Other Layer" not in n]
    others = sorted([n for n in nodes_in_masked if "Other Layer" in n])
    
    # Final list: Principals first (top), Others last (bottom)
    final_node_list = principals + others
    node_indices = {name: i for i, name in enumerate(final_node_list)}

    # 5. Manual Y-Coordinates to FORCE 'Other' to the bottom
    # x is the layer (normalized 0 to 1), y is the vertical pos
    x_coords = []
    y_coords = []
    for node in final_node_list:
        # Determine x based on layer
        # If it's an "Other" node, extract the layer from the string
        if "Other Layer" in node:
            l_idx = int(node.split()[-1])
            x_coords.append(l_idx / max_layer)
            y_coords.append(0.95) # Push to very bottom
        else:
            l_idx = layers_dict.get(node, max_layer)
            x_coords.append(l_idx / max_layer)
            y_coords.append(None) # Let Plotly decide for Top-N

    # 6. Coloring and Figure (Same RdBu logic as before)
    norm = mcolors.TwoSlopeNorm(vmin=df['SHAP_Weight'].min(), vcenter=0, vmax=df['SHAP_Weight'].max())
    cmap = plt.get_cmap('RdBu_r')

    node_colors = []
    for node in final_node_list:
        if "Other" in node:
            node_colors.append("rgba(180, 180, 180, 1.0)")
        else:
            net_val = masked_df[masked_df['Target']==node]['SHAP_Weight'].sum()
            if net_val == 0: net_val = masked_df[masked_df['Source']==node]['SHAP_Weight'].mean()
            rgb = cmap(norm(net_val))[:3]
            node_colors.append(f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 1.0)')

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node = dict(
            pad=20, thickness=25, label=final_node_list, color=node_colors,
            x=x_coords, y=y_coords # This forces the 'Other' nodes down
        ),
        link = dict(
            source=masked_df['Source'].map(node_indices),
            target=masked_df['Target'].map(node_indices),
            value=masked_df['SHAP_Weight'].abs(),
            color=[ "rgba(200,200,200,0.3)" if "Other" in str(s)+str(t) else 
                   f'rgba({int(cmap(norm(w))[0]*255)}, {int(cmap(norm(w))[1]*255)}, {int(cmap(norm(w))[2]*255)}, 0.4)'
                   for s, t, w in zip(masked_df.Source, masked_df.Target, masked_df.SHAP_Weight)]
        )
    )])

    fig.write_html(output_filename)
    print(f"Flexible Sankey saved to {output_filename}")


# Generate the data
# Generate and test
#df_complex = generate_complex_binn_data()


df_complex = {}
l = []
#df_complex.update({'Source':'', 'Target':'', 'SHAP_Weight':})
l.append({'Source':'G1', 'Target':'SP1', 'SHAP_Weight':2})
l.append({'Source':'G1', 'Target':'SP2', 'SHAP_Weight':-2})
l.append({'Source':'G2', 'Target':'SP3', 'SHAP_Weight':2})
l.append({'Source':'G3', 'Target':'SP3', 'SHAP_Weight':-1.5})
l.append({'Source':'G4', 'Target':'SP4', 'SHAP_Weight':0.3})
l.append({'Source':'G5', 'Target':'SP4', 'SHAP_Weight':3})
l.append({'Source':'G6', 'Target':'SP4', 'SHAP_Weight':-0.1})

l.append({'Source':'SP1', 'Target':'P1', 'SHAP_Weight':2})
l.append({'Source':'SP2', 'Target':'P2', 'SHAP_Weight':3})
l.append({'Source':'SP3', 'Target':'P3', 'SHAP_Weight':-3})
l.append({'Source':'SP4', 'Target':'P3', 'SHAP_Weight':-1})
l.append({'Source':'SP4', 'Target':'P4', 'SHAP_Weight':-1})

l.append({'Source':'P1', 'Target':'out', 'SHAP_Weight':2})
l.append({'Source':'P2', 'Target':'out', 'SHAP_Weight':2})
l.append({'Source':'P3', 'Target':'out', 'SHAP_Weight':3})
l.append({'Source':'P3', 'Target':'out', 'SHAP_Weight':-1})
l.append({'Source':'P4', 'Target':'out', 'SHAP_Weight':-3})

l = pd.DataFrame(l)

df_complex = pd.concat([l], axis=0).reset_index(drop=True)
print(f"Generated {len(df_complex)} connections across 4 layers.")

draw_sankey(df_complex, 3)