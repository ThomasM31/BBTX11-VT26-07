import preprocess as pre
import argparse
import os
from pathlib import Path
import pipeline_paths as ppaths

'''
SECOND PIPELINE TO PREPARE DATA FOR PREPROCESSING.
DOES TASKS THAT MUST BE DONE FOR SEVERAL DATASETS AT ONCE.
'''

def pipeline(
        to_include: list, 
        min_cells:int,
        shared_dir_mode: bool
        ) -> None:
    
    pp = ppaths.PipelinePaths(shared_dir_mode)
        
    #-------START PROCESSING-------
    
    # from int to readable labels
    included_labels = pre.get_labels(to_include)

    # create empty dict
    datasets = pre.get_datasets(included_labels)

    # note: we don't need to read the anndata files for this pipeline

    # sum gene expression counts for all datasets, 
    # list all genes that are expressed in more than min_cells in .csv
    f = pp.genes_keep_path / f'genes_to_keep_{min_cells}.csv'
    pre.sum_gene_expr_counts(
        included_labels, 
        pp.gene_expr_count_path, 
        f, 
        min_cells)    

    # We need to have access to the genes that we actually want to keep
    # (highly expressed and in Reactome) so that we can limit our HVGs
    # so we don't get genes in our list of common HVGs that we don't want to use.
    genes_to_keep = []
    # get genes with expression over threshold from file
    genes_to_keep.extend(
        pre.get_expressed_genes(datasets, pp.genes_keep_path, min_cells)
        )

    # get genes that exist in reactome from file
    save_file = Path('reactome_genes.txt')
    genes_to_keep.extend(pre.get_reactome_genes(datasets, pp.pathway_data_path, save_file))
    genes_to_keep = set(genes_to_keep)
    
    # Find n_top common hvgs from txt files, save to file
    pre.find_common_hvgs(
        datasets, 
        pp.hvg_lists_path, 
        pp.hvg_common_path
        )

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
        "--shared_dir_mode", 
        type=bool,
        default=True,
        help="Directory to work in: shared = True, user = False. Default is True."
    )

    args = parser.parse_args()

    to_include = args.to_include
    min_cells = args.gene_in_min_cells
    shared_dir_mode = args.shared_dir_mode

    pipeline(
        to_include, 
        min_cells=min_cells,
        shared_dir_mode = shared_dir_mode
        )