import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
import matplotlib.colors as mcolors
hv.extension('bokeh')

import mock_shap_data as mockshap

def plot_sankey(df, n_top=3, filename='sankey_diagram.html'):
    # 1. Setup absolute weights
    df = df.copy()
    df['Abs_Weight'] = df['SHAP_Weight'].abs()
    
    # 2. Dynamic Layer Detection
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
    
    # 3. Apply Global Layer-wise Top-N Filtering
    df['layer'] = df['Source'].map(layer_map)
    final_edges: list[pd.DataFrame] = []
    
    for layer in sorted(df['layer'].unique()):
        layer_df = df[df['layer'] == layer].copy()
        
        # Calculate total importance for every unique Source in this layer
        source_importance = layer_df.groupby('Source')['Abs_Weight'].sum().sort_values(ascending=False)
        
        # Identify which sources stay and which get grouped
        top_sources = source_importance.head(n_top).index.tolist()
        
        # Split the dataframe
        top_df = layer_df[layer_df['Source'].isin(top_sources)]
        others_df = layer_df[~layer_df['Source'].isin(top_sources)]
        
        final_edges.append(top_df)
        
        if not others_df.empty:
            # Group all 'Other' sources but keep their specific targets
            # This prevents the "Other" node from becoming a massive bottleneck 
            # by preserving where the flow is going.
            grouped_others = others_df.groupby(['Target']).agg({
                'SHAP_Weight': 'sum',
                'Abs_Weight': 'sum'
            }).reset_index()
            
            grouped_others['Source'] = f'Other in Layer {layer}'
            final_edges.append(grouped_others)

    df_final = pd.concat(final_edges).reset_index(drop=True)

    # 4. Vertical Ordering & Net Importance Logic
    s_net = df_final.groupby('Source')['SHAP_Weight'].sum()
    t_net = df_final.groupby('Target')['SHAP_Weight'].sum()
    node_net = s_net.add(t_net, fill_value=0).reset_index()
    node_net.columns = ['index', 'net_weight']

    s_abs = df_final.groupby('Source')['Abs_Weight'].sum()
    t_abs = df_final.groupby('Target')['Abs_Weight'].sum()
    node_abs = s_abs.add(t_abs, fill_value=0).reset_index()
    node_abs.columns = ['index', 'importance']
    
    node_imp = pd.merge(node_net, node_abs, on='index')
    
    node_layers = {}
    for _, row in df_final.iterrows():
        node_layers[row['Source']] = layer_map.get(row['Source'], 0)
        if row['Target'] not in node_layers:
            node_layers[row['Target']] = layer_map.get(row['Source'], 0) + 1

    # 5. Styling & Colors

    # Calculate max_val based on the largest absolute weight in the final data
    # We use .abs().max() to ensure the scale covers both extreme positive and negative values
    dynamic_max = df_final['SHAP_Weight'].abs().max()
    
    # Optional: Add a small buffer (e.g., 5%) so the darkest colors aren't 
    # only at a single point, or keep it as is for strict scaling.
    max_val = dynamic_max if dynamic_max > 0 else 1.0

    # 1. Define the palette (Sync this with your colorbar logic)
    rdbu_colors = ["#0571b0", "#FFFFFF", "#ca0020"]
    n_steps = 256
    # Create a list of 256 hex codes from Blue to White to Red
    full_palette = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list("", rdbu_colors)(np.linspace(0, 1, n_steps))]

    def get_color_intensity(val, name):
        if name and "Other in Layer" in str(name): 
            return '#D3D3D3'
        
        if name and "Alzheimer" in str(name): 
            return '#333333'

        # Normalize the value from [-max_val, max_val] to [0, 1]
        # This aligns exactly with how the colorbar is mapped
        norm_val = (np.clip(val, -max_val, max_val) + max_val) / (2 * max_val)

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
    node_imp['sort_key'] = node_imp['index'].apply(lambda x: f"ZZZZ_{x}" if x == 'Other connections' else x)
    node_imp = node_imp.sort_values(['layer', 'sort_key']).reset_index(drop=True)

    # Define vdims so the Sankey object "sees" the color column
    nodes_ds = hv.Dataset(node_imp, kdims=['index'], vdims=['node_color', 'importance'])

    # 6. Plot Generation
    # Ensure the nodes are strictly sorted by layer to hint the layout engine
    node_imp = node_imp.sort_values('layer').reset_index(drop=True)
    
    nodes_ds = hv.Dataset(node_imp, kdims=['index'], vdims=['node_color', 'importance', 'layer'])

    sankey = hv.Sankey((df_final, nodes_ds), kdims=['Source', 'Target'], vdims=['Abs_Weight', 'edge_color'])

    sankey_obj = sankey.opts(
        opts.Sankey(
            width=1000, 
            height=600,
            node_color='node_color',
            edge_color='edge_color',
            node_line_color=None,
            node_size=0,
            edge_alpha=0.6,
            node_width=20,
            node_padding=20,
            label_position='outer',
            # INCREASE THIS: 0 prevents the layout engine from moving nodes to correct columns
            iterations=32,        
            show_values=False,
            color_index=None 
        )
    )
    
    from bokeh.models import LinearColorMapper, ColorBar, FixedTicker

    # 1. Create the smooth RdBu palette
    # We expand the 3-color RdBu into 256 steps for a perfect gradient
    palette = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list("", rdbu_colors)(np.linspace(0, 1, 256))]
    color_mapper = LinearColorMapper(palette=palette, low=-max_val, high=max_val)

    # 2. Explicitly define the ColorBar widget
    c_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=12,
        border_line_color=None,
        location=(0,0),
        ticker=FixedTicker(ticks=[-max_val, max_val]),
        major_label_overrides={max_val: 'High', -max_val: 'Low'}
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

plot_sankey(mockshap.generate_data(), n_top=7)