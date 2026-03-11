# ================================
# Binn pathway masks: 
# Baserad på Hartman et al. (2025)
# ================================

import itertools
import re
import os
from typing import List, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

# Interna hjälpfkt:er (biologisk logik)

def _subset_mapping(input_data, mapping):
    return [m for m in mapping if m[0] in input_data]

def _subset_pathways_on_idx(pathways, mapping):
    def add_pathways(idx_list, target):
        if not target: return idx_list
        updated_idx_list = idx_list + target
        subsetted = [p for p in pathways if p[0] in target]
        new_target = list({p[1] for p in subsetted})
        return add_pathways(updated_idx_list, new_target)
    original_target = list({m[1] for m in mapping})
    idx_list = add_pathways([], original_target)
    return [p for p in pathways if p[0] in idx_list]

def _get_mapping_to_all_layers(pathways, mapping):
    graph = nx.DiGraph()
    graph.add_edges_from(pathways)
    components = {"input": [], "connections": []}
    unique_inputs = {m[0] for m in mapping}
    total = len(unique_inputs)
    print(f"      -> Mappar {total} gener till Reactome-stigar...")
    for i, inp in enumerate(unique_inputs):
        if i % 5000 == 0 and i > 0: print(f"         ...bearbetat {i}/{total} gener")
        translation_nodes = [m[1] for m in mapping if m[0] == inp]
        for node_id in translation_nodes:
            if graph.has_node(node_id):
                reachable_nodes = nx.single_source_shortest_path(graph, node_id).keys()
                for conn in reachable_nodes:
                    components["input"].append(inp); components["connections"].append(conn)
    return pd.DataFrame(components).drop_duplicates()

def _get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys())
    all_inputs = list(itertools.chain.from_iterable(layer_dict.values()))
    unique_inputs = list(np.unique(all_inputs))
    df = pd.DataFrame(index=pathways, columns=unique_inputs)
    for pathway, inputs in layer_dict.items():
        df.loc[pathway, inputs] = 1
    return df.infer_objects(copy=False).fillna(0).T

def _add_edges(G, node, n_layers):
    source = node
    for level in range(n_layers):
        target = f"{node}_copy{level + 1}"
        G.add_edge(source, target)
        source = target
    return G

def _complete_network(G, n_layers=4):
    sub_graph = nx.ego_graph(G, "output_node", radius=n_layers)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        dist = len(nx.shortest_path(sub_graph, source="output_node", target=node))
        if dist <= n_layers:
            _add_edges(sub_graph, node, n_layers - dist + 1)
    return sub_graph

def _get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, "output_node", radius=distance))
    if distance >= 1:
        nodes -= set(nx.ego_graph(net, "output_node", radius=distance - 1))
    return list(nodes)

def _get_layers_from_net(net, n_layers):
    layers = []
    for dist in range(n_layers):
        nodes = _get_nodes_at_level(net, dist)
        l_map = {re.sub(r"_copy.*", "", n): [re.sub(r"_copy.*", "", s) for s in net.successors(n)] for n in nodes}
        layers.append(l_map)
    return layers

# Huvudklass:
class PathwayNetwork:
    def __init__(self, input_data, pathways, mapping):
        self.input_data = input_data
        self.mapping = _subset_mapping(self.input_data, mapping)
        self.pathways = _subset_pathways_on_idx(pathways, self.mapping)
        self.mapping = _get_mapping_to_all_layers(self.pathways, self.mapping)
        self.inputs = sorted(set(self.mapping["input"].tolist()))
        
        G = nx.DiGraph()
        G.add_edges_from(self.pathways)
        self.pathway_graph = G.reverse()
        for node in [n for n, d in self.pathway_graph.in_degree() if d == 0]:
            self.pathway_graph.add_edge("output_node", node)

    def get_connectivity_matrices(self, n_layers):
        comp_graph = _complete_network(self.pathway_graph, n_layers)
        layers = _get_layers_from_net(comp_graph, n_layers)
        
        term_nodes = [n for n, d in comp_graph.out_degree() if d == 0]
        term_map = {re.sub(r"_copy.*", "", n): self.mapping.loc[self.mapping["connections"] == re.sub(r"_copy.*", "", n), "input"].unique().tolist() for n in term_nodes}
        layers.append(term_map)

        matrices, curr_in = [], self.inputs
        for i, l_dict in enumerate(layers[::-1]):
            print(f"      ...beräknar matris för lager {i}")
            mat = _get_map_from_layer(l_dict)
            if i == 0: curr_in = sorted(mat.index); self.inputs = curr_in
            merged = pd.DataFrame(index=curr_in).merge(mat, right_index=True, left_index=True, how="inner")
            merged = merged.reindex(sorted(merged.columns), axis=1).sort_index()
            matrices.append(merged)
            curr_in = list(mat.columns)
        return matrices


# Paths:
INPUT_DIR = '/data/shared/alzgene26/data/conv_data/'
OUTPUT_DIR = '/data/shared/alzgene26/PathwayData/'
CONNECTIVITY_FILE = '/data/shared/alzgene26/PathwayData/binn_connectivity.csv'

def main():
    print("=== STARTING FULL MASK GENERATION ===")
    
    # Read the network connectivities
    # includes both gene to pathway and pathway to pathway
    print("Reading connectivities")
    network_df = pd.read_csv(CONNECTIVITY_FILE).dropna(subset=['child', 'parent']).astype(str)
    
    # True if the child element is a pathway (pathway-pathway entries)
    is_pathway = network_df['child'].str.contains('R-HSA-')

    pathway_pathway_mapping = network_df[is_pathway]
    pathway_pathway_mapping.rename(columns={'child': 'source', 'parent': 'target'})
    
    gene_pathway_mapping = network_df[~is_pathway]
    gene_pathway_mapping.rename(columns={'child': 'input', 'parent': 'translation'})

    # convert to lists of tuples
    pathway_pathway_mapping = list(pathway_pathway_mapping.itertuples(index=False, name=None))
    gene_pathway_mapping = list(gene_pathway_mapping.itertuples(index=False, name=None))

    # Find all cell types (.h5ad-files)
    cell_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.h5ad')]
    print(f"Found {len(cell_files)} cell types to process.\n")

    for file_name in cell_files:
        cell_name = file_name.replace('.h5ad', '')
        print(f">>> Processing: {cell_name}")
        
        try:
            # Read AnnData
            adata = sc.read_h5ad(os.path.join(INPUT_DIR, file_name), backed='r')
            # Extract all relevant genes
            gene_list = adata.var_names.tolist()
            
            # Create pathway network object
            pn = PathwayNetwork(gene_list, 
                                pathway_pathway_mapping,
                                gene_pathway_mapping
                                )
            
            # Generate matrixes NOTE: (4 layers, change later?)
            matrices = pn.get_connectivity_matrices(n_layers=4)
            
            # Save every matrix with the name of the cell type
            for i, m in enumerate(matrices):
                out_path = os.path.join(OUTPUT_DIR, f"{cell_name}_layer_{i}_mask.csv")
                m.to_csv(out_path)
                print(f"      Saved: {out_path} (Shape: {m.shape})")
                
        except Exception as e:
            print(f"      [ERROR] could not process {cell_name}: {e}")
            
    print("\n=== ALL CELL TYPES PROCESSED ===")

if __name__ == "__main__":
    main()