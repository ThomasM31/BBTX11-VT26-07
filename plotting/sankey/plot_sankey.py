import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
import matplotlib.colors as mcolors
import logging
hv.extension('bokeh')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_sankey(df, n_top=3, filename='sankey_diagram.html'):
    logging.info(f"Starting Sankey plot generation with n_top={n_top}")
    
    # Setup absolute weights
    df = df.copy()
    df['Abs_Weight'] = df['SHAP_Weight'].abs()

    df['Source'] = df['Source'] + "_L" + df['Source_Layer'].astype(str)
    df['Target'] = df['Target'] + "_L" + df['Target_Layer'].astype(str)

    print(df.head())
    
    # Dynamic Layer Detection
    all_sources = set(df['Source'])
    all_targets = set(df['Target'])
    roots = [s for s in all_sources if s not in all_targets]
    
    layer_map = {node: 0 for node in roots}
    processed = False
    iteration = 0
    max_iterations = 1000

    logging.info("Starting layer mapping loop...")
    while not processed:
        processed = True
        iteration += 1
        for _, row in df.iterrows():
            if row['Source'] in layer_map:
                target_layer = layer_map[row['Source']] + 1
                if row['Target'] not in layer_map or layer_map[row['Target']] < target_layer:
                    layer_map[row['Target']] = target_layer
                    processed = False

        if iteration > max_iterations:
            logging.error("Layer detection stuck in infinite loop! Check for cycles in your Source/Target data.")
            break
    logging.info(f"Layer mapping complete after {iteration} iterations. Max layer: {max(layer_map.values()) if layer_map else 0}")
    
    logging.info("--- SHAP Weight Sparsity Analysis ---")
    for layer in sorted(df['Source_Layer'].unique()):
        layer_df = df[df['Source_Layer'] == layer]
        total_rows = len(layer_df)
        
        # Count rows where the weight is exactly 0
        zero_rows = (layer_df['SHAP_Weight'] == 0).sum()
        
        # Calculate percentage
        percent_zero = (zero_rows / total_rows) * 100 if total_rows > 0 else 0
        
        logging.info(f"Layer {layer}: {percent_zero:6.2f}% of connections have 0.0 weight ({zero_rows}/{total_rows})")
    logging.info("-------------------------------------")

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
    logging.info("Step 4: Calculating node importance and weights...")
    
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
    
    # --- OPTIMIZED LAYER MAPPING ---
    logging.info(f"Mapping layers for {len(node_imp)} unique nodes...")
    
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

    # Styling & Colors

    # Calculate max_val and min_val based on the largest absolute weight in the final data
    dynamic_max = df_final['SHAP_Weight'].max()
    dynamic_min = df_final['SHAP_Weight'].min()

    # Filter out "Other" paths to find the real biological extremes
    biological_edges = df_final[
        (~df_final['Source'].str.contains("Other in Layer", na=False)) & 
        (~df_final['Target'].str.contains("Other in Layer", na=False))
    ]

    # If all edges were "Other" (unlikely), fallback to the full dataframe
    if not biological_edges.empty:
        dynamic_max = biological_edges['SHAP_Weight'].max()
        dynamic_min = biological_edges['SHAP_Weight'].min()
    else:
        dynamic_max = df_final['SHAP_Weight'].max()
        dynamic_min = df_final['SHAP_Weight'].min()

    max_val = dynamic_max
    min_val = dynamic_min
    
    # Define the palette
    rdbu_colors = ["#0571b0", "#C17DEC", "#ca0020"]
    n_steps = 256
    # Create a list of 256 hex codes from Blue to White to Red
    full_palette = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list("", rdbu_colors)(np.linspace(-1, 1, n_steps))]

    def get_color_intensity(val, name):
        if name and "Other in Layer" in str(name): 
            return '#D3D3D3'
        
        if name and "output_node" in str(name): 
            return '#333333'

        denominator = dynamic_max - dynamic_min
        if denominator == 0:
            norm_val = 0.5
        else:
            norm_val = (val - dynamic_min)

        norm_val = max(0, min(1, norm_val))

        # Map the 0.0-1.0 value to an index in our 256-color palette
        idx = int(norm_val * (n_steps - 1))
        return full_palette[idx]

    # Apply to nodes and edges
    node_imp['node_color'] = node_imp.apply(
        lambda x: get_color_intensity(x['net_weight'], x['index']), axis=1
    )
    df_final['edge_color'] = df_final.apply(
        lambda row: get_color_intensity(row['SHAP_Weight'], row['Source']), axis=1
    )
    
    # Sorting
    node_imp['layer'] = node_imp['index'].map(node_layers)
    # Sort by layer then importance to push top nodes to the top of the stack
    node_imp = node_imp.sort_values(['layer', 'importance'], ascending=[True, False]).reset_index(drop=True)
    node_imp['label'] = node_imp['index'].apply(lambda x: x.split('_L')[0] if '_L' in str(x) else x)

    readable_labels = {}
    with open('/data/shared/alzgene26/data/PathwayData/ReactomePathways.gmt', "r") as file:
        for line in file:
            line_objs = line.split('\t')
            if len(line_objs) >= 2:
                readable_labels[line_objs[1]] = line_objs[0]

    # Create a clean label
    node_imp['label'] = node_imp['index'].apply(lambda x: x.split('_L')[0] if '_L' in str(x) else x)
    
    # replace each clean label with corresponding value in readable labels
    node_imp['label'] = node_imp['label'].apply(lambda x: readable_labels.get(x, x))

    nodes_ds = hv.Dataset(node_imp, kdims=['index'], vdims=['node_color', 'importance', 'layer', 'label'])

    sankey = hv.Sankey((df_final, nodes_ds), kdims=['Source', 'Target'], vdims=['Abs_Weight', 'edge_color'])

    sankey_obj = sankey.opts(
        opts.Sankey(
            label_index='label',
            width=1400, 
            height=1000,
            node_color='node_color',
            edge_color='edge_color',
            node_line_color=None,
            node_size=0,
            node_fill_alpha=0.6,
            edge_alpha=0.6,
            node_width=20,
            node_padding=20,
            label_position='outer',
            iterations=0,        
            show_values=False,
            color_index=None 
        )
    )
    
    from bokeh.models import LinearColorMapper, ColorBar, FixedTicker

    # 1. Create the smooth RdBu palette
    # We expand the 3-color RdBu into 256 steps for a perfect gradient
    palette = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list("", rdbu_colors)(np.linspace(0, 1, 256))]
    color_mapper = LinearColorMapper(palette=palette, low=min_val, high=max_val)

    # 2. Explicitly define the ColorBar widget
    c_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=12,
        border_line_color=None,
        location=(0,0),
        ticker=FixedTicker(ticks=[-min_val, max_val]),
        major_label_overrides={max_val: 'High', min_val: 'Low'}
    )

    # Update the mapper with dynamic low and high
    color_mapper = LinearColorMapper(palette=palette, low=min_val, high=max_val)

    # Update the ticker to show the actual max values on the legend
    c_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=FixedTicker(ticks=[min_val, 0, max_val]),
        major_label_overrides={
            max_val: f'High ({max_val:.2f})', 
            0: '0', 
            -max_val: f'Low ({-max_val:.2f})'
        },
        label_standoff=12,
        border_line_color=None,
        location=(0,0)
    )

    # 3. Create a host for the colorbar
    # Using an empty Curve prevents phantom plots from appearing
    colorbar_plot = hv.Curve([]).opts(
        width=30, 
        height=150, 
        xaxis=None, 
        yaxis=None, 
        show_frame=False, 
        toolbar=None,
        hooks=[lambda plot, element: plot.state.add_layout(c_bar, 'right')]
    )

    # 4. Final Combined Layout
    # Removed 'spacing' to fix the ValueError
    layout = (sankey_obj + colorbar_plot).opts(
        opts.Layout(shared_axes=False, merge_tools=False)
    )

    hv.save(layout, filename, backend='bokeh')
    return layout

shap_data = pd.read_pickle('/data/shared/alzgene26/data/results/binn_model/shap_explanation_layered_260508_0940.pkl')

plot_sankey(shap_data, n_top=10)