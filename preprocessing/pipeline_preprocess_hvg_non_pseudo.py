import preprocess as pre
import argparse
import os
from pathlib import Path
import pipeline_paths as ppaths

'''
THE ACTUAL PREPROCESSING PIPELINE. USES DATA CREATED BY THE FIRST TWO PREP PIPELINES.
RUN EACH DATASET SEPARATELY.
'''

def pipeline(
        to_include: list, 
        n_top_genes: int, 
        min_genes: int, 
        min_cells:int,
        nr_common_hvgs:int,
        common_hvg_inc_value:int,
        shared_dir_mode: bool
        ) -> None:
    
    run_vars = f'mg_{min_genes}_mc_{min_cells}_mhvg{nr_common_hvgs}' 
    pp = ppaths.PipelinePaths(shared_dir_mode, run_vars)
        
    #-------START PROCESSING-------
    
    # from int to readable labels
    included_labels = pre.get_labels(to_include)

    # create empty dict
    datasets = pre.get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    pre.read_files(datasets, pp.conv_data_path)

    #-------PERFORM FILTERING-------
    
    # The cell filtering doesn't actually remove anything from our dataset using
    # the default values, therefore commented out

    # remove cells with less than 200 expressed genes
    #pre.filter_cells_by_min_genes(datasets, min_genes=200)
    
    # remove cells with mito gene expression over threshold
    #pre.filter_cells_by_mitochondrial_content(datasets)

    genes_to_keep = []
    
    # get genes with expression over threshold from file
    genes_to_keep.append(pre.get_expressed_genes(
        datasets, pp.genes_keep_path, min_cells)
        )

    # get genes that exist in reactome from file
    save_file = Path('reactome_genes.txt')
    genes_to_keep.append(pre.get_reactome_genes(datasets, pp.pathway_data_path, save_file))

    # filter genes
    pre.filter_genes(datasets, genes_to_keep)
    
    #-------FILTER HVGs-------
    
    # prepare filename to read common hvgs
    all_labels = pre.get_labels(list(range(0,9)))
    hvg_common_filename =  f'{"_".join(all_labels)}_common_{nr_common_hvgs}.txt'
    
    # filter by common hvgs
    pre.filter_common_hvgs(datasets, pp.hvg_common_path, hvg_common_filename)

    #-------PSEUDOBULK AND NORMALIZE-------

    # sum counts per subject and high res cell type
    #pre.pseudobulk(datasets)
    
    #  normalize per pseudobulk sample
    #pre.normalize(datasets)

    #-------ADD METADATA-------

    # add disease status
    pre.add_metadata_non_pseudo(datasets, pp.metadata_path)


    #-------MOVE PROCESSED DATA TO MAIN LAYER AND SAVE-------
    
    # move pseudobulk data to main layer, discard everything else
    # this will make the files a lot smaller
    #pre.move_pseudo_main(datasets)
    
    pre.move_hvgs_main(datasets)
    
    print(f'finished:')
    for label, adata in datasets.items():
        print(adata)

    pre.save_files(datasets, pp.compl_hvg_non_pseudo, 'completed')
    
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