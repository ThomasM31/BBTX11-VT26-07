from preprocess import *
import argparse
import os
from pathlib import Path

def pipeline(
        to_include: list, 
        n_top_genes: int, 
        min_genes: int, 
        min_cells:int,
        min_common_hvgs:int,
        common_hvg_inc_value:int,  
        draw_umaps: bool
        ) -> None:
    
    ### To read from and write to current users folders
    user = os.environ.get('USER') or os.environ.get('USERNAME')
    base_path = Path("/data/users") / user / "kand/data/"
    
    # To read from and write to the shared folder
    #base_path = Path("/data/shared/alzgene26/data")
    
    filepath = str(base_path)

    # from int to readable labels
    included_labels = get_labels(to_include)

    # create empty dict
    datasets = get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    read_files(datasets, filepath)

    #-------FILTERING PREP FOR PER GENE FILTERING-------
    
    for label, adata in datasets.items():
        f = os.path.join(filepath, f'processed_data/expr_counts/{label}.csv')
        if not Path(f).exists():
            # writes gene expression count per dataset to .csv
            d = {label:adata}
            write_gene_expr_count(d, filepath)

    f = os.path.join(filepath, f'genes_to_keep_{min_cells}.csv')
    if not Path(f).exists():
        # sum gene expression counts for all datasets, 
        # list all genes that are expressed in more than min_cells in .csv
        sum_gene_expr_counts(datasets, filepath, min_cells)

    #-------PERFORM FILTERING-------
    # filter bad cells
    filter_cells(datasets, min_genes=200)

    # filter lowly expressed genes
    filter_genes(datasets, filepath, min_cells)

    #-------FIND AND FILTER HVGs-------
    
    # make one .txt file for each cell type with 
    # genes sorted from most to least variable
    # this method does not require normalized data
    for label, adata in datasets.items():
        f = os.path.join(filepath, f'processed_data/hvg_lists/{label}.txt')
        if not Path(f).exists():
            d = {label:adata}
            extract_hvgs_full_list(d, filepath)

    # find common hvgs from txt files
    included = "_".join(datasets.keys())
    hvg_file = f'{included}_common_{min_common_hvgs}.txt'
    f = os.path.join(f'{filepath}/hvg_common', hvg_file)
    
    if not Path(f).exists():
        find_common_hvgs(datasets, filepath, n_top_genes, min_common_hvgs, common_hvg_inc_value)

    # filter by common hvgs
    # requires a file with all included cell types in the file name, 
    # and the min nr of common HVGs to include
    # e.g. astro_immune_common_1000.txt
    filter_common_hvgs(datasets, filepath, hvg_file)

    #-------PSEUDOBULK AND NORMALIZE-------

    # sum counts per subject and high res cell type
    pseudobulk(datasets)
    
    #  normalize per pseudobulk sample
    normalize(datasets)

    #-------ADD METADATA-------

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
        "--min_common_hvgs", 
        type=int,
        default=1000,
        help="Target for minimum nr of common hvgs"
    )

    # Optional argument
    parser.add_argument(
        "--common_hvg_inc_value", 
        type=int,
        default=3000,
        help="Amount to increment n_top_genes if nr common genes is under target."
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
    min_common_hvgs = args.min_common_hvgs
    common_hvg_inc_value = args.common_hvg_inc_value

    pipeline(
        to_include, 
        n_top_genes=n_top, 
        min_genes=min_genes, 
        min_cells=min_cells, 
        min_common_hvgs=min_common_hvgs, 
        common_hvg_inc_value=common_hvg_inc_value, 
        draw_umaps=draw_umaps
        )