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
from collections import defaultdict
import argparse
from pathlib import Path

# Internal helper functions

def _filter_relevant_gp_conns(
        relevant_genes: set[str], 
        gene_pathway_conn: list[tuple[str, str]]
        ) -> list[tuple[str, str]]:
    '''
    Extracts relevant gene-pathway connections:
    keeps entry only if the gene exists in the clinical data set
    '''
    return [tup for tup in gene_pathway_conn if tup[0] in relevant_genes]

def _filter_relevant_pp_cons(
        pathway_pathway_conn: list[tuple[str, str]],
        gene_pathway_conn: list[tuple[str, str]]
        ) -> list:
    '''
    Returns relevant pathway-pathway connections, i.e. pathways that are 
    downstream of the genes that exist in the clinical data set.
    '''

    def add_pathways(idx_list: list, relevant_pathways: list[tuple[str, str]]):
        # ends when relevant_pathways is empty
        if not relevant_pathways: return idx_list
        
        # concatenate
        updated_idx_list = idx_list + relevant_pathways

        # filter for child pathways that are connected to a gene that 
        # exists in the clinical dataset
        subsetted = [tup for tup in pathway_pathway_conn 
                    if tup[0] in relevant_pathways]

        # list of parent pathways, intermediate conversion to set
        new_target = list({p[1] for p in subsetted})
        
        # recursive until list of relevant pathways is empty
        return add_pathways(updated_idx_list, new_target)
    
    # list of relevant pathways from gene-pathway connections 
    # (input only has genes that exist in the data set)
    relevant_pathways = list({m[1] for m in gene_pathway_conn})
    
    # filter pathway-pathway connections
    idx_list = add_pathways([], relevant_pathways)

    # list of relevant pathway-pathway connections
    res = [tup for tup in pathway_pathway_conn if tup[0] in idx_list]
    return res

def _get_flattened_gp_hierarchy(
        pathway_pathway_conn: list[tuple[str, str]], 
        gene_pathway_conn: list[tuple[str, str]]
        ) -> pd.DataFrame:
    '''
    Returns a data frame with columns input (gene) and connections (pathway ID), 
    which represents a flattened hierarchy of all genes and pathways that are 
    (perhaps indirectly) connected
    '''
    
    # directed graph
    graph = nx.DiGraph()
    # add all pathway-pathway connections as edges
    graph.add_edges_from(pathway_pathway_conn)

    # map genes to immediate pathways for faster lookup
    gene_to_pathways = defaultdict(list)
    for gene, pathway in gene_pathway_conn:
        gene_to_pathways[gene].append(pathway)
    
    components = {"input": [], "connections": []}
    
    # gene set
    unique_genes = {m[0] for m in gene_pathway_conn}
    nrGenes = len(unique_genes)

    print(f'Mapping {nrGenes} genes to Reactome-paths...')
    
    for i, gene in enumerate(unique_genes):
        if i % 5000 == 0 and i > 0: print(f"...processed {i}/{nrGenes} genes")

        # list parent pathways of this particular gene 
        parent_pathways = gene_to_pathways[gene]
        
        for pathway_id in parent_pathways:
            if graph.has_node(pathway_id):
                # get all ancestors/descendant reachable in the graph
                reachable_nodes = (
                    nx.single_source_shortest_path(graph, pathway_id)
                    .keys())
                    
                for conn in reachable_nodes:
                    components["input"].append(gene)
                    components["connections"].append(conn)
    
    # drop all duplicate rows (same gene & pathway)
    # returns flattened hierarchy with gene and pathway columns
    return pd.DataFrame(components).drop_duplicates()

def _get_layer_adjacency_matrix(layer: dict) -> pd.DataFrame:
    '''
    Returns per layer adjacency matrix with;
    rows: genes
    columns: pathways
    '''
    # row elements
    pathways = list(layer.keys())

    # column elements
    all_inputs = list(itertools.chain.from_iterable(layer.values()))
    unique_inputs = list(np.unique(all_inputs))
    
    # creating an empty grid
    df = pd.DataFrame(index=pathways, columns=unique_inputs)
    
    # set cell to 1 if connection exists
    for pathway, inputs in layer.items():
        df.loc[pathway, inputs] = 1

    # set cell to 0 if no connection exists, 
    # transpose to correct shape
    return df.infer_objects(copy=False).fillna(0).T

def _add_skip_connections(
        sub_graph: nx.DiGraph, 
        node: str, 
        n_layers: int
        ) -> nx.DiGraph:
    
    '''Adds skip connections until depth n_layers is reached'''
    source = node
    
    for n in range(n_layers):
        # just make a copy with the id of prev node + suffix
        target = f"{node}_copy{n + 1}"
        # adds skip connection
        sub_graph.add_edge(source, target)
        source = target
    return sub_graph

def _get_normalized_subgraph(
        pathway_graph: nx.DiGraph, 
        n_layers: int = 4
        ) -> nx.DiGraph:
    """
    Ensure that every path in the network has the depth n_layers.
    Returns a layer normalized subgraph.
    """

    # cuts of any paths more than n_layers deep
    sub_graph = nx.ego_graph(pathway_graph, "output_node", radius=n_layers)
    
    # pathways that link directly to genes
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    
    for node in terminal_nodes:
        # distance between output and terminal node
        dist = len(nx.shortest_path(sub_graph, source="output_node", target=node))
        
        # pad with skip connections if depth is < n_layers
        if dist <= n_layers:
            _add_skip_connections(sub_graph, node, n_layers - dist + 1)
    
    return sub_graph

def _get_nodes_at_level(
        normalized_subgraph: nx.DiGraph, 
        n_layers: int
        ) -> list[str]:
    '''
    Returns a list of nodes for one specific level of the network.
    '''
    # nodes max n steps away from output node
    nodes = set(nx.ego_graph(normalized_subgraph, "output_node", radius=n_layers))
    
    # remove all lower level nodes, leaving only the ones exactly on layer n
    if n_layers >= 1:
        nodes -= set(nx.ego_graph(normalized_subgraph, "output_node", radius=n_layers - 1))
    return list(nodes)

def _get_network_layers(
        normalized_subgraph: nx.DiGraph, 
        n_layers: int
        ) -> list[dict]:
    '''
    Returns a list of dictionaries, one dict for each level of the network.
    Each dictionary contains the following;
    key: node
    value: list of the node's successors
    '''
    
    layers = []
    for n in range(n_layers):
        nodes = _get_nodes_at_level(normalized_subgraph, n)

        l_map = {}

        for node in nodes:
            # clean suffix from current node
            clean_node = re.sub(r"_copy.*", "", node)

            # find successors and clean suffix
            clean_succesors = []
            for successor in normalized_subgraph.successors(node):
                clean_s = re.sub(r"_copy.*", "", successor)
                clean_succesors.append(clean_s)
            
            l_map[clean_node] = clean_succesors
        
        # for each level, append a map of nodes an successors
        layers.append(l_map)
    
    return layers

# Huvudklass:
class PathwayNetwork:
    def __init__(
        self, 
        relevant_genes: set[str], 
        pathway_pathway_conn: list[tuple[str, str]], 
        gene_pathway_conn: list[tuple[str, str]]
        ) -> None:

        self.relevant_genes = relevant_genes
        
        # Filters relevant gene-pathway conns (gene exists in clinical data set)
        self.gene_pathway_conn = _filter_relevant_gp_conns(
            self.relevant_genes, gene_pathway_conn)
        
        # Filters to only keep pathway-pathway conns that are 
        # downstream of genes that exist in the clinical data set
        self.pathway_pathway_conn = _filter_relevant_pp_cons(
            pathway_pathway_conn, self.gene_pathway_conn)

        # Creates a flattened hierearchy in the form of a data frame 
        # with all genes and pathways that are connected
        self.flattened_gp_hierarchy = _get_flattened_gp_hierarchy(
            self.pathway_pathway_conn, self.gene_pathway_conn)
        
        # filters for the genes that are actually connected to a 
        # pathway network, removes duplicates, sorts alphabetical for consistency
        self.input_genes = sorted(set(self.flattened_gp_hierarchy["input"]))
        
        # directed graph
        G = nx.DiGraph()
        # add all pathway-pathway connections as edges
        G.add_edges_from(self.pathway_pathway_conn)
        
        # reverses graph so that we get the direction parent->child
        self.pathway_graph = G.reverse()
        
        # find root nodes (with in-degree 0), represent high level 'broad' networks
        root_nodes = [n for n, d in self.pathway_graph.in_degree() 
                      if d == 0 and n != "output_node"]
        
        # adds output node as connection to each root node,
        # because each path should lead to the final output
        for node in root_nodes:
            self.pathway_graph.add_edge("output_node", node)

    def get_connectivity_matrices(
            self, 
            n_layers: int
            ) -> list[pd.DataFrame]:
        
        normalized_subgraph = _get_normalized_subgraph(self.pathway_graph, n_layers)
        network_layers = _get_network_layers(normalized_subgraph, n_layers)
        
        original_pathways = set(self.pathway_graph.nodes())
        normalized_pathways = set(normalized_subgraph.nodes())
        pruned_pathways = original_pathways - normalized_pathways
        print(f"Pathways pruned (unreachable or too deep): {len(pruned_pathways)}")

        # TODO: this is the logic that we need to change if 
        # we want to map genes to intermediate pathways
        terminal_nodes = [n for n, d in normalized_subgraph.out_degree() if d == 0]
        
        # group genes for speed of compute
        df_conn = pd.DataFrame(self.gene_pathway_conn, columns=["input", "connections"])
        grouped_genes = df_conn.groupby("connections")["input"].unique()

        # Check if any genes are mapped to intermediate pathways
        all_annotated_pathways = set(df_conn["connections"])
        terminal_pathways = {re.sub(r"_copy.*", "", n) for n in terminal_nodes}

        non_terminal_annotated = all_annotated_pathways - terminal_pathways
        print(f"Genes mapped to intermediate pathways: {len(non_terminal_annotated)}")
        
        # TODO: this is the logic that we need to change if 
        # we want to map genes to intermediate pathways
        term_map = {}
        for n in terminal_nodes:
            # remove copy suffix from pathway
            clean_pathway = re.sub(r"_copy.*", "", n)
            
            # connect sub-pathways to genes
            term_map[clean_pathway] = grouped_genes.get(
                clean_pathway, np.array([])).tolist()
        
        network_layers.append(term_map)

        matrices, curr_in = [], self.input_genes

        # process network layers in reverse, input -> output
        for i, layer_connections in enumerate(network_layers[::-1]):
            print(f"      ...calculating matrix for layer {i}")
            adj_mat = _get_layer_adjacency_matrix(layer_connections)
            
            # the network expects gene input in alphabetical order
            if i == 0: 
                curr_in = sorted(adj_mat.index); 
                self.input_genes = curr_in
            
            # ensures only genes/pathways present in graph and current data are kept
            merged = pd.DataFrame(index=curr_in).merge(
                adj_mat, right_index=True, left_index=True, how="inner")
            
            # force rows and columns into alph order for consistency in network
            merged = merged.reindex(sorted(merged.columns), axis=1).sort_index()
            matrices.append(merged)
            
            # outputs of current layer is input for the next
            curr_in = list(adj_mat.columns)
        return matrices



def main(
        INPUT_DIR: Path, 
        OUTPUT_DIR: Path, 
        CONNECTIVITY_FILE: Path, 
        several_datasets: bool, 
        mask_label: str, 
        n_layers: int
        ):
    print("=== STARTING FULL MASK GENERATION ===")

    # Read the network connectivities
    # includes both gene to pathway and pathway to pathway
    print("Reading connectivities")
    network_df = (pd.read_csv(CONNECTIVITY_FILE)
                 .dropna(subset=['child', 'parent'])
                 .astype(str))
    
    # True if the child element is a pathway (pathway-pathway entries)
    is_pathway = network_df['child'].str.contains('R-HSA-')

    pathway_pathway_mapping = network_df[is_pathway]
    pathway_pathway_mapping.rename(columns={'child': 'source', 'parent': 'target'})
    
    gene_pathway_mapping = network_df[~is_pathway]
    gene_pathway_mapping.rename(columns={'child': 'input', 'parent': 'translation'})

    # convert to lists of tuples
    pathway_pathway_mapping = list(pathway_pathway_mapping.itertuples(index=False, name=None))
    gene_pathway_mapping = list(gene_pathway_mapping.itertuples(index=False, name=None))

    def _do_mask_creation(file: str, save_str: str, data_info: str) -> None:
        cell_name = file.removesuffix('.h5ad')
        print(f">>> Processing: {cell_name}")
        
        try:
            # Read AnnData
            f = INPUT_DIR / file
            adata = sc.read_h5ad(f, backed='r')
            # Extract all relevant genes, convert to set for faster lookup
            gene_set = set(adata.var_names.tolist())
            
            # Create pathway network object
            pn = PathwayNetwork(gene_set, 
                                pathway_pathway_mapping,
                                gene_pathway_mapping
                                )

            missing_genes = gene_set - set(pn.input_genes)
            print(f"Nr genes without pathway annotations: {len(missing_genes)}")
            
            # Generate matrixes
            matrices = pn.get_connectivity_matrices(n_layers=n_layers)
            
            for i, m in enumerate(matrices):
                save_name = save_str if save_str != '' else cell_name 
                out_path = OUTPUT_DIR / data_info / f"{save_name}_layer_{i}_mask.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                m.to_csv(out_path)
                print(f"      Saved: {out_path} (Shape: {m.shape}) to {out_path}")
                
        except Exception as e:
            print(f"      [ERROR] could not process {cell_name}: {e}")
    
    
    # Find all cell types (.h5ad-files)
    cell_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.h5ad')]
    print(f"Found {len(cell_files)} cell types. \n")

    # get the directory name (has info on vars used to create processed files)
    data_info = INPUT_DIR.parts[-1]
    if several_datasets:
        for file_name in cell_files:
            _do_mask_creation(file_name, '', data_info)
            print("\n=== ALL CELL TYPES PROCESSED ===")
    else:
        cell_files_stripped = [c.removesuffix('.h5ad') for c in cell_files]
        save_str = "_".join(cell_files_stripped)
        # we select which cell type to make the mask from (some may have fewer than n_top HVGs)
        to_mask_from = f'{mask_label}.h5ad'
        _do_mask_creation(to_mask_from, save_str, data_info)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Pre-process anndata objects and save as .h5ad"
    )

    parser.add_argument(
        "filepath_in", 
        type=Path,
        help="Filepath to read from"
    )

    parser.add_argument(
        "filepath_out", 
        type=Path,
        help="Filepath to output masks to"
    )

    parser.add_argument(
        "connectivity_file", 
        type=Path,
        help="File with gene-pathway and pathway-pathway connections"
    )

    # Optional argument
    parser.add_argument(
        "--several_datasets", 
        type=bool,
        default=False,
        help="If True prepares mask matrixes for all files in a folder (if gene set is different for files / cell types)"
    )

    # Optional argument
    parser.add_argument(
        "--mask_label", 
        type=str,
        default='exc3',
        help="label for cell type to mask from"
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="nr of layers (excluding output layer)"
    )

    args = parser.parse_args()

    filepath_in = args.filepath_in
    filepath_out = args.filepath_out
    connectivity_file = args.connectivity_file
    several_datasets = args.several_datasets
    mask_label = args.mask_label
    n_layers = args.n_layers

    main(filepath_in, filepath_out, connectivity_file, several_datasets, mask_label, n_layers)