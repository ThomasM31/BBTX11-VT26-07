from preprocess import *
import argparse
import os
from pathlib import Path

def pipeline(
        hvg_file: str,
        to_include: list, 
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
    # this time from the 'hvg' folder
    read_hvg_adata(datasets, filepath)

    # filter by common hvgs
    filter_common_hvgs(datasets, filepath, hvg_file)

    # sum counts per subject and high res cell type
    pseudobulk(datasets)
    
    #  normalize per pseudobulk sample
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

    parser.add_argument(
        "hvg_file",
        type=str,
        metavar="FILE",
        help="Path to the file containing common Highly Variable Genes (HVGs)."
    )

    # Positional argument: accepts one or more integers
    parser.add_argument(
        "to_include",
        type=int,        
        nargs='+',       
        help="indices to include: \n0=astro \n1=exc1 \n2=exc2 \n3=exc3 \n4=immune \n5=inhi \n6=oligo \n7=opcs \n8=vasc",
    )

    # Optional argument
    parser.add_argument(
        "--draw_umaps", 
        type=bool,
        default=False,
        help="Visualize cell sparation. Defualt is False"
    )

    args = parser.parse_args()

    hvg_file = args.hvg_file
    to_include = args.to_include
    draw_umaps = args.draw_umaps

    pipeline(
        hvg_file,
        to_include,
        draw_umaps=draw_umaps
        )