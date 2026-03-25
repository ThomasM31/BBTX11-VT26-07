import preprocess as pre
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
        draw_umaps: bool,
        shared_dir_mode: bool
        ) -> None:
    
    if shared_dir_mode:
        # To read from and write to the shared folder
        base_path = Path("/data/shared/alzgene26/data")
    else:
        ### To read from and write to current users folders
        #user = os.environ.get('USER') or os.environ.get('USERNAME')
        #base_path = Path("/data/users") / user / "kand/data/"
        pass
    
    processed_data = "processed_data"
    run_vars = f'mg_{min_genes}_mc_{min_cells}_mhvg{min_common_hvgs}' 

    conv_data_path          = base_path / "conv_data"
    gene_expr_count_path    = base_path / processed_data / "expr_counts"
    genes_keep_path         = base_path / processed_data / "filter_genes"
    hvg_lists_path          = base_path / processed_data / "hvg_lists"
    hvg_common_path         = base_path / processed_data / "hvg_common"
    figures_path            = base_path / "figures"
    metadata_path           = base_path / "supplementary_data"
    completed_path          = base_path / processed_data / "completed" / run_vars

    # create folders if they do not exist
    gene_expr_count_path.mkdir(parents=True, exist_ok=True)
    genes_keep_path.mkdir(parents=True, exist_ok=True)
    hvg_lists_path.mkdir(parents=True, exist_ok=True)
    hvg_common_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    completed_path.mkdir(parents=True, exist_ok=True)
    
    conv_data_path           = str(conv_data_path)
    gene_expr_count_path    = str(gene_expr_count_path)
    genes_keep_path         = str(genes_keep_path)
    hvg_lists_path          = str(hvg_lists_path)
    hvg_common_path         = str(hvg_common_path)
    figures_path            = str(figures_path)
    metadata_path           = str(metadata_path)
    completed_path          = str(completed_path)

    filepath = str(base_path)

    # from int to readable labels
    included_labels = pre.get_labels(to_include)

    # create empty dict
    datasets = pre.get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    pre.read_files(datasets, conv_data_path)

    #-------FILTERING PREP FOR PER GENE FILTERING-------
    
    for label, adata in datasets.items():
        f = os.path.join(gene_expr_count_path, f'{label}.csv')
        if not Path(f).exists():
            # writes gene expression count per dataset to .csv
            d = {label:adata}
            pre.write_gene_expr_count(d, gene_expr_count_path)

    f = os.path.join(genes_keep_path, f'genes_to_keep_{min_cells}.csv')
    if not Path(f).exists():
        # sum gene expression counts for all datasets, 
        # list all genes that are expressed in more than min_cells in .csv
        pre.sum_gene_expr_counts(
            datasets, 
            gene_expr_count_path, 
            genes_keep_path, 
            min_cells)

    #-------PERFORM FILTERING-------
    # filter bad cells
    pre.filter_cells(datasets, min_genes=200)

    # filter lowly expressed genes
    pre.filter_genes(datasets, genes_keep_path, min_cells)

    #-------FIND AND FILTER HVGs-------
    
    # make one .txt file for each cell type with 
    # genes sorted from most to least variable
    # this method does not require normalized data
    for label, adata in datasets.items():
        f = os.path.join(hvg_lists_path, f'{label}.txt')
        if not Path(f).exists():
            d = {label:adata}
            pre.extract_hvgs_full_list(d, hvg_lists_path)

    # find common hvgs from txt files
    included = "_".join(datasets.keys())
    hvg_file = f'{included}_common_{min_common_hvgs}.txt'
    f = os.path.join(hvg_common_path, hvg_file)
    
    if not Path(f).exists():
        pre.find_common_hvgs(
            datasets, 
            hvg_lists_path, 
            hvg_common_path, 
            n_top_genes, 
            min_common_hvgs, 
            common_hvg_inc_value
            )

    # filter by common hvgs
    # requires a file with all included cell types in the file name, 
    # and the min nr of common HVGs to include
    # e.g. astro_immune_common_1000.txt
    pre.filter_common_hvgs(datasets, hvg_common_path, hvg_file)

    #-------PSEUDOBULK AND NORMALIZE-------

    # sum counts per subject and high res cell type
    pre.pseudobulk(datasets)
    
    #  normalize per pseudobulk sample
    pre.normalize(datasets)

    #-------ADD METADATA-------

    # add disease status
    pre.add_metadata(datasets, metadata_path)

    pre.save_files(datasets, completed_path, 'completed')

    # visualize how the preprocessing has improved (?) 
    # separation of cells (slow and uses a lot of memory!!)
    # for this one it is better to load each data set separately
    if draw_umaps: pre.draw_umaps(datasets=datasets, filepath=figures_path)

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

    # Optional argument
    parser.add_argument(
        "--shared_dir_mode", 
        type=bool,
        default=True,
        help="Directory to work in: shared = True, user = False. Default is True."
    )

    args = parser.parse_args()

    to_include = args.to_include
    n_top = args.n_top_genes
    min_cells = args.gene_in_min_cells
    min_genes = args.cell_with_min_genes
    draw_umaps = args.draw_umaps
    min_common_hvgs = args.min_common_hvgs
    common_hvg_inc_value = args.common_hvg_inc_value
    shared_dir_mode = args.shared_dir_mode

    pipeline(
        to_include, 
        n_top_genes=n_top, 
        min_genes=min_genes, 
        min_cells=min_cells, 
        min_common_hvgs=min_common_hvgs, 
        common_hvg_inc_value=common_hvg_inc_value, 
        draw_umaps=draw_umaps,
        shared_dir_mode = shared_dir_mode
        )