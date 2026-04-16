import preprocess as pre
import argparse
import os
from pathlib import Path

def pipeline(
        to_include: list,
        proc_dir: str
        ) -> None:
    
    base_path = Path("/data/shared/alzgene26/data")
    proc_dir = Path(proc_dir)
    processed_data = "processed_data"

    paths = {}
    paths["conv_data_path"] = base_path / "conv_data"
    paths["figures_path"]   = base_path / "figures"
    paths["completed_path"] = base_path / processed_data / "completed" / proc_dir
    paths["test_data_path"] = base_path / processed_data / "test_data"

    # create folders if they do not exist
    for path_name, path  in paths.items():
        path.mkdir(parents=True, exist_ok=True)

    # from int to readable labels
    included_labels = pre.get_labels(to_include)

    # create empty dict
    datasets_orig = pre.get_datasets(included_labels)
    datasets_pseudo = pre.get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    pre.read_files(datasets_orig, paths["conv_data_path"])

    pre.read_intermediate(datasets_pseudo, paths["completed_path"])

    if len(datasets_orig.keys()) != len(datasets_pseudo.keys()):
        print("Not same number of datasets to compare. Exiting.")
        return
    
    #-------DRAW UMAPS-------

    # visualize how the preprocessing has improved (?) 
    # separation of cells (slow and uses a lot of memory!!)
    # for this one it is better to load each data set separately
    pre.draw_umaps(datasets_orig, datasets_pseudo, paths["figures_path"])

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

    parser.add_argument(
        "proc_dir",
        type=str,               
        help="name of directory with processed data",
    )

    args = parser.parse_args()

    to_include = args.to_include
    proc_dir = args.proc_dir

    pipeline(
        to_include,
        proc_dir
        )