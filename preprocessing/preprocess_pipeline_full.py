from preprocess import *
import argparse
import os
from pathlib import Path

def pipeline(
        to_include: list, 
        n_top_genes: int, 
        min_genes: int, 
        min_cells:int, 
        draw_umaps: bool
        ) -> None:
    
    ### To read from and write to current users folders
    user = os.environ.get('USER') or os.environ.get('USERNAME')
    base_path = Path("/data/users") / user / "kand/data/"
    filepath = str(base_path)

    # from int to readable labels
    included_labels = get_labels(to_include)

    # create empty dict
    datasets = get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    read_files(datasets, filepath)

    # filter bad cells
    filter_cells(datasets, min_genes=200)

    # filter lowly expressed genes
    filter_genes(datasets, min_cells=200)

    # normalize for HVG selection
    prep_for_hvg_sel(datasets)

    # write hvgs per cell type to txt file
    extract_per_cell_type_hvgs(datasets, filepath, n_top_genes)

    # find and filter common hvgs, from txt files
    filter_shared_hvgs(datasets, filepath, n_top_genes)

    # sum counts per subject and high res cell type
    pseudobulk(datasets)
    
    normalize(datasets)

    # add disease status
    add_metadata(datasets, filepath)

    save_files(datasets, filepath, 'pseudo')

    # visualize how the preprocessing has improved (?) 
    # separation of cells (slow!!)
    if draw_umaps: draw_umaps(datasets)

    print('Pipeline completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Pre-process anndata objects and save as .h5ad"
    )

    # Positional argument: accepts one or more integers
    parser.add_argument(
        "to_include",
        type=int,        
        nargs='+',       
        help="indices to include: \n0=astro \n1=exc1 \n2=exc2 \n3=exc3 \n4=immune \n5=inhi \n6=oligo \n7=opcs \n8=vasc",
    )

    # Optional argument n_top_genes
    parser.add_argument(
        "--n_top_genes", 
        type=int,
        default=2000,
        help="Number of highly variable genes to keep (default: 2000)"
    )

    # Optional argument
    parser.add_argument(
        "--gene_in_min_cells", 
        type=int,
        default=200,
        help="Filter out genes that are expressed in less than n cells."
    )

    # Optional argument
    parser.add_argument(
        "--cell_with_min_genes", 
        type=int,
        default=200,
        help="Filter out cells that have less than n gene expressions."
    )

    # Optional argument
    parser.add_argument(
        "--draw_umaps", 
        type=bool,
        default=False,
        help="Visualize cell sparation. Defualt is False"
    )

    args = parser.parse_args()

    to_include = args.to_include
    n_top = args.n_top_genes
    min_cells = args.gene_in_min_cells
    min_genes = args.cell_with_min_genes
    draw_umaps = args.draw_umaps

    pipeline(to_include, n_top_genes=n_top, min_genes=min_genes, min_cells=min_cells, draw_umaps=draw_umaps)