from pathlib import Path

import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
import matplotlib.colors as mcolors
import logging
import re
from bokeh.models import LinearColorMapper, ColorBar, FixedTicker
hv.extension('bokeh')
#hv.help(hv.Sankey)


import pipeline_paths as ppaths

# Global configuration and directory setup
pp = ppaths.PipelinePaths(True)
save_path: Path = pp.figures_path_shap / 'sankey'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

readable_labels: dict[str, str] = {}
with open('/data/shared/alzgene26/data/PathwayData/ReactomePathways.gmt', "r") as file:
    for line in file:
        line_objs = line.split('\t')
        if len(line_objs) >= 2:
            # line_objs[1] is the R-HSA ID, line_objs[0] is the Name
            readable_labels[line_objs[1]] = line_objs[0]

def format_fname_label(text):
    return "".join('_' if c == ' ' else c for c in text if c.isalnum() or c == ' ')

def wrap(text, words_per_row=5, hyphen_limit=8):
    # wraps node labels over several lines for readability
    def protect(match):
        word = match.group(0)
        if len(word) <= hyphen_limit:
            return word.replace('-', '§') 
        return word.replace('-', '- ')

    protected_text = re.sub(r'\S+-\S+', protect, str(text))
    words = protected_text.split()
    
    lines = []
    for i in range(0, len(words), words_per_row):
        line = ' '.join(words[i:i+words_per_row])
        lines.append(line.replace('§', '-').replace('- ', '-'))
        
    return '\n'.join(lines)

def translate_node_name(full_name):
    """Extracts ID from 'ID_L1', translates it, and reappends the layer."""
    if "_L" not in str(full_name):
        return full_name
    
    parts = full_name.split('_L')
    node_id = parts[0]
    layer = parts[1]
    
    # Translate if ID exists in our map, else keep the ID
    clean_name = readable_labels.get(node_id, node_id)
    return f"{clean_name}_L{layer}"

def get_subnetwork(df, seed_node, direction='downstream'):
    logging.info(f"Subnetwork Trace: Starting {direction} from '{seed_node}'")
    relevant_edges = []
    nodes_to_process = [seed_node]
    processed_nodes = set()

    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        processed_nodes.add(current_node)
        
        if direction == 'downstream':
            new_edges = df[df['Source'] == current_node]
            new_nodes = set(new_edges['Target'].unique()) - processed_nodes
        else:
            new_edges = df[df['Target'] == current_node]
            new_nodes = set(new_edges['Source'].unique()) - processed_nodes
            
        if not new_edges.empty:
            logging.info(f"  - Found {len(new_edges)} edges for node: {current_node}")
            relevant_edges.append(new_edges)
            nodes_to_process.extend(list(new_nodes))

    if not relevant_edges:
        logging.warning(f"!!! CRITICAL: No subnetwork found for {seed_node}. Check if the name/layer matches the dataframe exactly.")
        return pd.DataFrame(columns=df.columns)

    return pd.concat(relevant_edges).drop_duplicates()

def extract_process_lineage(df, target_node, direction='upstream'):
    """
    direction: 'upstream' to find drivers (Genes), 
               'downstream' to find consequences (processes).
    """
    logging.info(f"Extracting {direction} lineage for: {target_node}")
    
    nodes_to_check = [target_node]
    discovered_nodes = {target_node}
    lineage_edges = []

    while nodes_to_check:
        current = nodes_to_check.pop(0)
        
        if direction == 'upstream':
            # Find edges where our target is the target
            step_df = df[df['Target'] == current]
            next_nodes = set(step_df['Source'].unique())
        else:
            # Find edges where our target is the source
            step_df = df[df['Source'] == current]
            next_nodes = set(step_df['Target'].unique())

        if not step_df.empty:
            lineage_edges.append(step_df)
            # Only add nodes we haven't visited to prevent infinite loops
            new_nodes = next_nodes - discovered_nodes
            nodes_to_check.extend(list(new_nodes))
            discovered_nodes.update(new_nodes)

    if not lineage_edges:
        logging.warning(f"No {direction} edges found for {target_node}")
        return pd.DataFrame()

    return pd.concat(lineage_edges).drop_duplicates()

def get_bidirectional_subnetwork(df, target_node):
    logging.info(f"Extracting complete bi-directional path for: {target_node}")
    
    # Get all nodes upstream
    upstream_edges = extract_process_lineage(df, target_node, direction='upstream')
    
    # Get all nodes downstream
    downstream_edges = extract_process_lineage(df, target_node, direction='downstream')
    
    # Combine and remove duplicates
    combined_df = pd.concat([upstream_edges, downstream_edges]).drop_duplicates()
    
    if combined_df.empty:
        logging.error(f"Target '{target_node}' found no connections.")
        return combined_df

    logging.info(f"Path extracted: {len(combined_df)} unique edges.")
    return combined_df

def plot_sankey(
        df: pd.DataFrame, 
        n_top: int = 10, 
        filename: Path | str = save_path / 'sankey_diagram.html', 
        is_subnetwork_plot: bool =False, 
        output_node_label: str ='Output Node',
        no_labels: bool = False
        ):
    logging.info(f"Starting Sankey plot generation with n_top={n_top}")
    
    df = df.copy()
    df['Abs_Weight'] = df['SHAP_Weight'].abs()

    df['Source'] = df['Source'] + "_L" + df['Source_Layer'].astype(str)
    df['Target'] = df['Target'] + "_L" + df['Target_Layer'].astype(str)

    # Dynamic Layer Detection
    all_sources = set(df['Source'])
    all_targets = set(df['Target'])
    roots = [s for s in all_sources if s not in all_targets]
    
    layer_map = {node: 0 for node in roots}
    processed = False
    iteration = 0
    while not processed:
        processed = True
        iteration += 1
        for _, row in df.iterrows():
            if row['Source'] in layer_map:
                target_layer = layer_map[row['Source']] + 1
                if row['Target'] not in layer_map or layer_map[row['Target']] < target_layer:
                    layer_map[row['Target']] = target_layer
                    processed = False
        if iteration > 1000: break

    # Apply Global Layer-wise Top-N Filtering
    df['layer'] = df['Source'].map(layer_map)
    
    # Pre-calculate the Top-N biological nodes for EVERY layer first
    top_nodes_by_layer = {}
    for layer in sorted(df['layer'].unique()):
        # Importance is sum of Abs_Weight for any node appearing in this layer (Source or Target)
        # But usually, we care about the Source importance for the flow
        layer_df = df[df['layer'] == layer]
        importance = layer_df.groupby('Source')['Abs_Weight'].sum().sort_values(ascending=False)
        top_nodes_by_layer[layer] = set(importance.head(n_top).index.tolist())

    final_edges: list[pd.DataFrame] = []
    
    for layer in sorted(df['layer'].unique()):
        layer_df = df[df['layer'] == layer].copy()
        
        # Calculate the total weight in this layer
        layer_total = layer_df['Abs_Weight'].sum()
        
        # Normalize so the layer sum is 1.0 (prevents the "thinning" effect)
        if layer_total > 0:
            layer_df['Abs_Weight'] = layer_df['Abs_Weight'] / layer_total
        # --------------------------

        current_top_sources = top_nodes_by_layer[layer]
        next_layer = layer + 1
        next_top_nodes = top_nodes_by_layer.get(next_layer, set())

        # Force every Target to be Top-N or 'Other'
        layer_df['Target'] = layer_df['Target'].apply(
            lambda x: x if x in next_top_nodes or "output_node" in x else f'Other in Layer {next_layer}'
        )
        
        # Force every Source to be Top-N or 'Other'
        layer_df['Source'] = layer_df['Source'].apply(
            lambda x: x if x in current_top_sources or "Other in Layer" in x else f'Other in Layer {layer}'
        )

        grouped = layer_df.groupby(['Source', 'Target']).agg({
            'SHAP_Weight': 'sum', 
            'Abs_Weight': 'sum'
        }).reset_index()
        
        final_edges.append(grouped)

    df_final = pd.concat(final_edges).reset_index(drop=True)
    
    # If we have "Other -> Other" connections, group them
    df_final = df_final.groupby(['Source', 'Target']).agg({
        'SHAP_Weight': 'sum', 
        'Abs_Weight': 'sum'
    }).reset_index()
    logging.info(f"Filtering complete. Final edge count: {len(df_final)}")

    # Vertical Ordering & Net Importance Logic
    logging.info("Calculating node importance and weights...")
    
    # Vectorized grouping
    s_net = df_final.groupby('Source')['SHAP_Weight'].sum()
    t_net = df_final.groupby('Target')['SHAP_Weight'].sum()
    node_net = s_net.add(t_net, fill_value=0).reset_index()
    node_net.columns = ['index', 'net_weight']

    logging.info("Node weights calculated. Merging absolute importance...")
    s_abs = df_final.groupby('Source')['Abs_Weight'].sum()
    t_abs = df_final.groupby('Target')['Abs_Weight'].sum()
    node_abs = s_abs.add(t_abs, fill_value=0).reset_index()
    node_abs.columns = ['index', 'importance']
    
    node_imp = pd.merge(node_net, node_abs, on='index')
    
    # Get layers for all Sources 
    # For Targets, they are simply the Source's layer + 1
    source_layers = df_final[['Source']].drop_duplicates()
    source_layers['layer'] = source_layers['Source'].map(layer_map)
    
    target_layers = df_final[['Target', 'Source']].drop_duplicates('Target')
    target_layers['layer'] = target_layers['Source'].map(layer_map) + 1
    
    # Combine them into a single mapping dictionary
    combined_layers = pd.concat([
        source_layers[['Source', 'layer']].rename(columns={'Source': 'node'}),
        target_layers[['Target', 'layer']].rename(columns={'Target': 'node'})
    ]).drop_duplicates('node').set_index('node')['layer'].to_dict()

    node_layers = combined_layers
    logging.info("Layer mapping complete.")

    # Define the palette
    n_steps = 256
    max_val = 0.8
    nodes = [0.0, 0.5, 0.5, 1.0]
    rdbu_colors = ["#0a9cf1", "#cce4f3", "#f4c4cc", "#f44b67"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", list(zip(nodes, rdbu_colors)))
    palette = [mcolors.to_hex(c) for c in cmap(np.linspace(0, 1, n_steps))]

    def get_color(val, name):
        if name and "Other" in str(name): 
            return '#D3D3D3' # Grey
        
        if name and "output_node" in str(name): 
            return '#333333' # Dark Grey/Black
        
        norm_val = (np.clip(val, -max_val, max_val) + max_val) / (2 * max_val)

        # Map the 0.0-1.0 value to an index in our 256-color palette
        idx = int(norm_val * (n_steps - 1))
        return palette[idx]

    # Apply to nodes and edges
    node_imp['node_color'] = node_imp.apply(
        lambda x: get_color(x['net_weight'], x['index']), axis=1
    )
    df_final['edge_color'] = df_final.apply(
        lambda row: get_color(row['SHAP_Weight'], row['Source']), axis=1
    )

    # Sorting
    node_imp['layer'] = node_imp['index'].map(node_layers)
    node_imp = node_imp.sort_values(['layer', 'importance'], ascending=[True, False]).reset_index(drop=True)
    
    node_imp['label'] = node_imp['index'].apply(lambda x: x.split('_L')[0] if '_L' in str(x) else x)
    
    '''max_layer = node_imp['layer'].max()
    node_imp['label'] = node_imp.apply(
        lambda row: output_node_label if (
            row['layer'] == max_layer and 
            (row['label'] == 'output_node' or 'Other in Layer' in row['label'])
        ) else row['label'], axis=1
    )'''

    max_layer = node_imp['layer'].max()

    def process_label(row):
        # Global toggle for no labels
        if no_labels:
            return ' '
        
        current_label = str(row['label'])
        current_layer = row['layer']

        # Specific replacement for the final layer (Layer 5/Max Layer)
        if current_layer == max_layer and (current_label == 'output_node' or 'Other in Layer' in current_label) :
            return output_node_label
        
        if 'Other in Layer' in current_label and current_layer != max_layer:
            layer_num = current_label.split('Layer ')[-1]
            return f'Övriga i lager {layer_num}'
        
        return current_label

    node_imp['label'] = node_imp.apply(process_label, axis=1)

    if not no_labels:
        node_imp['label'] = node_imp['label'].apply(lambda x: readable_labels.get(x, x))
        node_imp['label'] = node_imp['label'].apply(lambda x: wrap(x, 3))

    #node_imp['label'] = node_imp['label'].apply(lambda x: readable_labels.get(x, x))
    #node_imp['label'] = node_imp['label'].apply(lambda x: wrap(x, 3)) # Shorter wrap for subnets

    node_imp = node_imp.sort_values('layer').reset_index(drop=True)

    nodes_ds = hv.Dataset(node_imp, kdims=['index'], vdims=['node_color', 'importance', 'layer', 'label'])

    sankey = hv.Sankey((df_final, nodes_ds), kdims=['Source', 'Target'], vdims=['Abs_Weight', 'edge_color'])

    sankey_obj = sankey.opts(
        opts.Sankey(
            label_index='label',
            width=1800, 
            height=800,
            node_color='node_color',
            edge_color='edge_color',
            node_line_color='#ffffff',
            node_line_width=1,
            node_size=0,
            node_fill_alpha=1,
            edge_fill_alpha=1,
            edge_alpha=1,
            node_width=20,
            node_padding=25,
            label_text_font_size='12pt',
            label_position='outer',
            iterations=0,        
            show_values=False,
            color_index=None 
        )
    )
    
    color_mapper = LinearColorMapper(palette=palette, low=-max_val, high=max_val)

    # ColorBar widget
    c_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=12,
        border_line_color=None,
        location=(0,0),
        ticker=FixedTicker(ticks=[-max_val, max_val]),
        major_label_overrides={max_val: 'High', -max_val: 'Low'},
        major_label_text_font_size='12pt'
    )

    # Update the mapper with dynamic low and high
    color_mapper = LinearColorMapper(palette=palette, low=-max_val, high=max_val)

    # Update the ticker to show the actual max values on the legend
    c_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=FixedTicker(ticks=[-max_val, 0, max_val]),
        major_label_overrides={
            max_val: f'High ({max_val:.2f})', 
            0: '0', 
            -max_val: f'Low ({-max_val:.2f})'
        },
        label_standoff=12,
        border_line_color=None,
        location=(0,0)
    )

    # Create a host for the colorbar
    colorbar_plot = hv.Curve([]).opts(
        width=30, 
        height=150, 
        xaxis=None, 
        yaxis=None, 
        show_frame=False, 
        toolbar=None,
        hooks=[lambda plot, element: plot.state.add_layout(c_bar, 'right')]
    )

    #Combined Layout
    layout = (sankey_obj + colorbar_plot).opts(
        opts.Layout(shared_axes=False, merge_tools=False)
    )

    hv.save(layout, filename, backend='bokeh')
    return layout


shap_data = pd.read_pickle('/data/shared/alzgene26/data/results/binn_model/shap_explanation_layered_260508_0940.pkl')

#plot_sankey(shap_data, 10, save_path / 'sankey_top_10.html', output_node_label='alzheimer')

'''
genes: list[tuple[str, int]] = [('UBB', 10), ('UBC', 10), ('PSMA1', 10), ('FYN', 10), ('ERBB4', 10)]
for gene, n_top in genes:
    downstream_df = get_subnetwork(shap_data, gene, direction='downstream')
    fname_label = format_fname_label(gene)
    plot_sankey(
    downstream_df, 
        n_top=n_top, 
        filename=save_path / f'plots/{fname_label}_downstream.html', 
    )
'''

draw_upstream: list[tuple[str, int]] = [('R-HSA-162582', 10), ('R-HSA-69278', 10), ('R-HSA-5693606', 10), ('R-HSA-1500620', 10), ("R-HSA-5693606", 10)]
draw_upstream: list[tuple[str, int]] = [('R-HSA-8953854', 10)]
for process, n_top in draw_upstream:
    df: pd.DataFrame = get_subnetwork(shap_data, process, direction='upstream')
    fname_label = format_fname_label(readable_labels[process])
    readable_label = readable_labels[process]
    plot_sankey(
        df, 
        n_top=n_top, 
        filename= save_path / f'{fname_label}_upstream_subnetwork.html',
        output_node_label=readable_label
    )



'''draw_downstream: list[tuple[str, int]] = [('R-HSA-69278', 10)]
for process, n_top in draw_downstream:
    df: pd.DataFrame = get_subnetwork(shap_data, process, direction='downstream')
    fname_label = format_fname_label(readable_labels[process])
    readable_label = readable_labels[process]
    plot_sankey(
        df, 
        n_top=n_top, 
        filename=save_path / f'{fname_label}_downstream_subnetwork.html',
        output_node_label=readable_label
    )'''

'''
target = "R-HSA-69278" # cell cycle, meiotic
readable_label = readable_labels[target]

upstream_df = extract_process_lineage(shap_data, target, direction='upstream')
downstream_df = extract_process_lineage(shap_data, target, direction='downstream')

# full lineage up- and downstream for a specific process 
bidirectional_df = get_bidirectional_subnetwork(shap_data, target)
plot_sankey(
    bidirectional_df, 
    n_top=100, 
    filename=save_path / 'meiosis_bidirectional_analysis.html', 
    is_subnetwork_plot=True
    )

plot_sankey(
    upstream_df, 
    n_top=15, 
    filename=save_path / 'meiosis_upstream_analysis.html', 
    is_subnetwork_plot=True, 
    output_node_label=readable_label
    )

target = "R-HSA-5693606" # DNA double strand break response
readable_label = readable_labels[target]
bidirectional_df = get_bidirectional_subnetwork(shap_data, target)
plot_sankey(
    bidirectional_df, 
    n_top=100, 
    filename=save_path / f'{readable_label}_bidirectional_analysis.html', 
    is_subnetwork_plot=True
    )

target = "R-HSA-162582" # Signal Transduction
readable_label = readable_labels[target]
upstream_df = extract_process_lineage(shap_data, target, direction='upstream')
plot_sankey(
    upstream_df, 
    n_top=10, 
    filename=save_path / 'signal_transduction_upstream_analysis.html', 
    is_subnetwork_plot=True, 
    output_node_label=readable_label
    )
'''