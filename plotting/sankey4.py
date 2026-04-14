import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
from collections import deque, defaultdict
hv.extension('bokeh')


def identify_layers(df):
    """
    Identify which layer each node belongs to based on graph topology.
    Returns a dict mapping node -> layer_index
    """
    # Build adjacency list
    graph = defaultdict(set)
    all_sources = set(df['Source'])
    all_targets = set(df['Target'])
    
    # Find root nodes (sources that are not targets)
    roots = [s for s in all_sources if s not in all_targets]
    
    # Build forward edges
    for _, row in df.iterrows():
        graph[row['Source']].add(row['Target'])
    
    # BFS from roots to assign layers
    node_to_layer = {}
    queue = deque([(root, 0) for root in roots])
    
    while queue:
        node, layer = queue.popleft()
        if node not in node_to_layer:
            node_to_layer[node] = layer
            for neighbor in graph[node]:
                queue.append((neighbor, layer + 1))
    
    return node_to_layer


def aggregate_nodes_by_layer(df, n_top):
    """
    Group all but the top n nodes (by absolute SHAP weight) in each layer 
    as 'Other_Layer[N]' nodes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sankey data with columns: Source, Target, SHAP_Weight
    n_top : int
        Number of top nodes to keep per layer
    
    Returns:
    --------
    pd.DataFrame
        Modified dataframe with aggregated "Other" nodes
    """
    df = df.copy()
    df['Abs_Weight'] = np.abs(df['SHAP_Weight'])
    
    # Identify layers
    node_to_layer = identify_layers(df)
    
    # Group nodes by layer
    layers_dict = {}
    for node, layer in node_to_layer.items():
        if layer not in layers_dict:
            layers_dict[layer] = []
        layers_dict[layer].append(node)
    
    # For each layer, identify top n nodes and mark others for aggregation
    layer_aggregations = {}  # layer -> {node -> aggregated_name}
    
    for layer_idx in sorted(layers_dict.keys()):  # Include all layers, including root
        layer_nodes = layers_dict[layer_idx]
        
        # Calculate total absolute weight per node in this layer
        node_weights = {}
        for node in layer_nodes:
            # Sum weights of all edges targeting this node (incoming) or from it (outgoing)
            incoming = df[df['Target'] == node]['Abs_Weight'].sum()
            outgoing = df[df['Source'] == node]['Abs_Weight'].sum()
            node_weights[node] = max(incoming, outgoing) if max(incoming, outgoing) > 0 else 0
        
        # Sort and identify top n
        sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)
        top_nodes = set([node for node, _ in sorted_nodes[:n_top]])
        
        # Create mapping: top nodes stay as-is, others map to aggregated node
        layer_aggregations[layer_idx] = {}
        for node in top_nodes:
            layer_aggregations[layer_idx][node] = node
        for node in layer_nodes:
            if node not in top_nodes:
                layer_aggregations[layer_idx][node] = f"Other_Layer{layer_idx}"
    
    # Rebuild dataframe: map source/target through aggregation
    new_rows = []
    
    for _, row in df.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row['SHAP_Weight']
        
        # Map source and target to their aggregated forms
        source_layer = node_to_layer.get(source, 0)
        target_layer = node_to_layer.get(target, 0)
        
        new_source = layer_aggregations.get(source_layer, {}).get(source, source)
        new_target = layer_aggregations.get(target_layer, {}).get(target, target)
        
        new_rows.append({
            'Source': new_source,
            'Target': new_target,
            'SHAP_Weight': weight
        })
    
    result_df = pd.DataFrame(new_rows)
    
    # Aggregate duplicate edges (same source-target pair) by summing weights
    result_df = result_df.groupby(['Source', 'Target'], as_index=False).agg({
        'SHAP_Weight': 'sum'
    })
    
    return result_df


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

def plot_sankey(df, n_top=3, n_hidden_layers=2, filename="sankey4.html"):
    # Aggregate nodes to keep only top n per layer
    df = aggregate_nodes_by_layer(df, n_top)
    
    df['Abs_Weight'] = df['SHAP_Weight'].abs()
    weights = df.groupby('Source')['Abs_Weight'].sum()

    all_nodes = set(df['Source']) | set(df['Target'])
    
    # Compute SHAP weights for nodes (sum of outgoing edges, or incoming if source is not root)
    node_shap_weights = {}
    for node in all_nodes:
        # For each node, compute its representative SHAP value from incident edges
        outgoing = df[df['Source'] == node]['SHAP_Weight'].sum()
        if outgoing != 0:
            node_shap_weights[node] = outgoing
        else:
            incoming = df[df['Target'] == node]['SHAP_Weight'].sum()
            node_shap_weights[node] = incoming
    
    # Add node SHAP weights to dataframe by mapping source/target to their weights
    df['Source_SHAP'] = df['Source'].map(node_shap_weights)
    df['Target_SHAP'] = df['Target'].map(node_shap_weights)
    
    all_sources = set(df['Source'])
    all_targets = set(df['Target'])

    roots = [s for s in all_sources if s not in all_targets]
    roots.sort(key=lambda x: weights.get(x,0), reverse=True)
    
    output = [s for s in all_nodes if s not in all_sources]

    layers = [roots]
    next_layer = []
    
    # Find nodes on each layer
    current_layer = layers[0]
    for _ in range(n_hidden_layers): # num_hidden_layers
        targets = df[df['Source'].isin(current_layer)]['Target'].unique()
        if len(targets) == 0:
            break

        next_layer = sorted(targets, key=lambda x: weights.get(x, 0), reverse=True)
        
        layers.append(next_layer)
        current_layer = next_layer

    layers.append(output)

    print(layers)

    sankey = hv.Sankey((df), kdims=['Source', 'Target'], vdims=['Abs_Weight', 'SHAP_Weight', 'Source_SHAP', 'Target_SHAP'])

    sankey_obj = sankey.opts(
        opts.Sankey(
            width=1000, height=600,
            edge_color='SHAP_Weight', edge_cmap='RdBu', edge_alpha=0.6,
            node_color='Source_SHAP', node_cmap='RdBu', fill_color='Source_SHAP',
            label_position='outer', node_width=20, node_padding=20,
            iterations=0, # Crucial: Setting to 0 respects our dataframe's vertical order
            show_values=False
        )
    )

    hv.save(sankey_obj, filename, backend='bokeh')
    return sankey_obj

plot_sankey(generate_data(), n_top=3, n_hidden_layers=2)