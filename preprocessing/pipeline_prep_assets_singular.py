import preprocess as pre
import argparse
import os
from pathlib import Path
import pipeline_paths as ppaths
import pandas as pd

'''
FIRST PIPELINE TO PREPARE DATA FOR PREPROCESSING.
DOES TASKS THAT CAN BE DONE FOR EACH DATASET SEPARATELY (RUN 1 AT A TIME).
'''

def pipeline(
        to_include: int,
        train_size: float,
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
        # writes gene expression count per dataset to .csv
        #pre.write_gene_expr_count({label:adata}, pp.gene_expr_count_path)     
        pass

    # save genes that exist in reactome to file
    source_file = 'ReactomePathways.gmt'
    save_file = 'reactome_genes.txt'
    pre.save_reactome_genes(pp.pathway_data_path, source_file, save_file)
    
    # do train-test split (for HVG selection)
    print('Selecting training subjects.')
    path = pp.metadata_path / 'individual_metadata_deidentified.tsv'
    metadata = pd.read_csv(path, sep='\t')
    md_sel = metadata[['subject']]
    # use the same random state for all cell types so we get same subjects for all
    sample = md_sel.sample(frac=train_size, random_state=92)

    # keep track of which subjects are to be used as train subjects
    train_subjects_save_path = pp.metadata_path / 'train_subjects.tsv'
    print(f'Writing {sample.size} train subjects to {train_subjects_save_path}')
    sample.to_csv(train_subjects_save_path)
    
    # make one .txt file for each cell type with 
    # genes sorted from most to least variable 
    # (only computed for train subjects to avoid data leakage)
    # this method does not require normalized data
    print('Ordering genes by variability')
    for label, adata in datasets.items():
        train_adata = adata[adata.obs['subject'].isin(sample['subject'])].copy()
        pre.extract_hvgs_full_list({label:train_adata}, pp.hvg_lists_path)
    
    print('Pipeline completed')

def range_limited_float_type(arg):
    MIN_VAL = 0.0
    MAX_VAL = 1.0
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError(f"Must be a float")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError(f"Argument must be in [{MIN_VAL},{MAX_VAL}]")
    return f

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

    parser.add_argument(
        "--train_size",
        help="float with min:0.0, max:1.0, default=0.8",
        type=range_limited_float_type,
        default=0.8
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
    train_size = args.train_size
    shared_dir_mode = args.shared_dir_mode

    pipeline(
        to_include, 
        train_size=train_size,
        shared_dir_mode = shared_dir_mode
        )