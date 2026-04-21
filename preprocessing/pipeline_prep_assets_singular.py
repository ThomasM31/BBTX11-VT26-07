import preprocess as pre
import argparse
import os
from pathlib import Path
import pipeline_paths as ppaths

'''
FIRST PIPELINE TO PREPARE DATA FOR PREPROCESSING.
DOES TASKS THAT CAN BE DONE FOR EACH DATASET SEPARATELY (RUN 1 AT A TIME).
'''

def pipeline(
        to_include: int, 
        n_top_genes: int, 
        min_genes: int, 
        min_cells:int,
        nr_common_hvgs:int,
        common_hvg_inc_value:int,
        shared_dir_mode: bool
        ) -> None:
    
    pp = ppaths.PipelinePaths(shared_dir_mode)

    #-------START PROCESSING-------

    # from int to readable labels
    included_labels = pre.get_labels([to_include])

    # create empty dict
    datasets = pre.get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    pre.read_files(datasets, pp.conv_data_path)

    #-------FILTERING PREP FOR PER GENE FILTERING-------
    for label, adata in datasets.items():
        f = pp.gene_expr_count_path / f'{label}.csv'
        # writes gene expression count per dataset to .csv
        d = {label:adata}
        pre.write_gene_expr_count(d, pp.gene_expr_count_path)     

    # save genes that exist in reactome to file
    source_file = 'ReactomePathways.gmt'
    save_file = 'reactome_genes.txt'
    pre.save_reactome_genes(pp.pathway_data_path, source_file, save_file)
    
    # make one .txt file for each cell type with 
    # genes sorted from most to least variable
    # this method does not require normalized data
    for label, adata in datasets.items():
        f = pp.hvg_lists_path / f'{label}.txt'
        d = {label:adata}
        pre.extract_hvgs_full_list(d, pp.hvg_lists_path)
    
    print('Pipeline completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Pre-process anndata objects and save as .h5ad"
    )

    # Positional argument: accepts one or more integers
    parser.add_argument(
        "to_include",
        type=int,     
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
        "--nr_common_hvgs", 
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
    nr_common_hvgs = args.nr_common_hvgs
    common_hvg_inc_value = args.common_hvg_inc_value
    shared_dir_mode = args.shared_dir_mode

    pipeline(
        to_include, 
        n_top_genes=n_top, 
        min_genes=min_genes, 
        min_cells=min_cells, 
        nr_common_hvgs=nr_common_hvgs, 
        common_hvg_inc_value=common_hvg_inc_value, 
        shared_dir_mode = shared_dir_mode
        )