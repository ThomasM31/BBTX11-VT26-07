import preprocess as pre
import argparse
import os
from pathlib import Path
import importlib.util
import anndata as ad

# import module for consistent paths
path = Path(__file__).resolve().parent.parent / "pipeline_paths.py"
spec = importlib.util.spec_from_file_location("ppaths", path)
ppaths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppaths)

# import module for consistent paths
path = Path(__file__).resolve().parent.parent / "pipeline_paths_generalize.py"
spec = importlib.util.spec_from_file_location("gpaths", path)
gpaths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpaths)

'''
THE ACTUAL PREPROCESSING PIPELINE. USES DATA CREATED BY THE FIRST TWO PREP PIPELINES.
RUN EACH DATASET SEPARATELY.
'''

def pipeline(
        n_top_genes: int, 
        min_genes: int, 
        min_cells:int,
        nr_common_hvgs:int,
        common_hvg_inc_value:int,
        shared_dir_mode: bool
        ) -> None:
    
    run_vars = f'mg_{min_genes}_mc_{min_cells}_mhvg{nr_common_hvgs}' 
    
    # path for train/test data
    ttp = ppaths.PipelinePaths(shared_dir_mode, full_pipeline_run_vars=run_vars)
    # path for generalizability data
    gp = gpaths.PipelinePaths(shared_dir_mode, full_pipeline_run_vars=run_vars)
        
    #-------START PROCESSING-------

    included_labels = ['all']
    
    adata = ad.read_h5ad(gp.conv_data_path / 'GSE157827_merged.h5ad')
    
    # rename to be same as training data set
    adata.obs = adata.obs.rename(columns={'group': 'subject'})
    datasets = {'all': adata}

    # get the order of the original genes (for verification after processing)
    genes_ordered = datasets[included_labels[0]].var_names.tolist()

    #-------PERFORM FILTERING-------
    
    # remove cells with less than 200 expressed genes
    pre.filter_cells_by_min_genes(datasets, min_genes=200)
    
    # remove cells with mito gene expression over threshold
    pre.filter_cells_by_mitochondrial_content(datasets)

    # remove genes not in the final mask (same genes for all cell types, just pick one)
    astro = ad.read_h5ad(ttp.compl_full_pipe_path / 'astro.h5ad')

    astro_genes = astro.var_names.tolist()

    adata = adata[:, astro_genes]
    datasets['all'] = adata
    print(adata)

    #-------PSEUDOBULK AND NORMALIZE-------

    # sum counts per subject and high res cell type
    pre.pseudobulk(datasets, 'mean')
    
    #  normalize per pseudobulk sample
    pre.normalize(datasets)

    #-------MOVE PROCESSED DATA TO MAIN LAYER AND SAVE-------
    
    # move pseudobulk data to main layer, discard everything else
    # this will make the files a lot smaller
    pre.move_to_main(datasets, 'pseudo')

    #-------ADD METADATA-------

    # add disease status
    adata = datasets['all']
    adata.obs['AD_status'] = adata.obs['subject'].str.startswith('AD').astype(int)
    datasets['all'] = adata

    print(adata)

    pre.save_files(datasets, gp.compl_full_pipe_path, 'completed')

    print('Pipeline completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Pre-process anndata objects and save as .h5ad"
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

    n_top = args.n_top_genes
    min_cells = args.gene_in_min_cells
    min_genes = args.cell_with_min_genes
    nr_common_hvgs = args.nr_common_hvgs
    common_hvg_inc_value = args.common_hvg_inc_value
    shared_dir_mode = args.shared_dir_mode

    pipeline(
        n_top_genes=n_top, 
        min_genes=min_genes, 
        min_cells=min_cells, 
        nr_common_hvgs=nr_common_hvgs, 
        common_hvg_inc_value=common_hvg_inc_value,
        shared_dir_mode = shared_dir_mode
        )