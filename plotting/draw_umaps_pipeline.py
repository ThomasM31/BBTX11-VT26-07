import umaps
import argparse
from pathlib import Path
import sys

parent = Path(__file__).resolve().parent.parent
sys.path.append(str(parent))

from preprocessing import pipeline_paths as ppaths
from preprocessing import preprocess as pre


def pipeline(
        to_include: list,
        proc_dir: Path,
        shared_dir_mode: bool
        ) -> None:
    
    pp = ppaths.PipelinePaths(shared_dir_mode, proc_dir.parts[-1])

    # from int to readable labels
    included_labels = pre.get_labels(to_include)

    # create empty dict
    datasets_orig = pre.get_datasets(included_labels)
    datasets_pseudo = pre.get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    pre.read_files(datasets_orig, pp.conv_data_path)

    pre.read_intermediate(datasets_pseudo, pp.compl_base / proc_dir)

    if len(datasets_orig.keys()) != len(datasets_pseudo.keys()):
        print("Not same number of datasets to compare. Exiting.")
        return
    
    #-------DRAW UMAPS-------

    # visualize how the preprocessing has improved (?) 
    # separation of cells (slow and uses a lot of memory!!)
    # for this one it is better to load each data set separately
    umaps.draw_umaps(datasets_orig, datasets_pseudo, pp.figures_path)

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
        type=Path,               
        help="name of directory with processed data (everything after 'completed')",
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
    proc_dir = args.proc_dir
    shared_dir_mode = args.shared_dir_mode

    pipeline(
        to_include,
        proc_dir,
        shared_dir_mode
        )