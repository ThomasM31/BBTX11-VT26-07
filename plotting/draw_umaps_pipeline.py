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
    
    pp = ppaths.PipelinePaths(shared_dir_mode)
    save_path = pp.figures_path / proc_dir
    save_path.mkdir(parents=True, exist_ok=True)

    # from int to readable labels
    included_labels = pre.get_labels(to_include)
    label = included_labels[0]

    # create empty dict
    datasets_processed = pre.get_datasets(included_labels)

    # read h5ad files, add to datasets dict
    pre.read_intermediate(datasets_processed, pp.compl_base / proc_dir)
    proc_dataset = datasets_processed[label]

    print(proc_dataset)

    color_by = 'cell_type_high_resolution'
    umaps.draw_umap(proc_dataset, label, save_path, 'processed', color_by)

    color_by = 'AD_status'
    umaps.draw_umap(proc_dataset, label, save_path, 'processed', 'AD_status')

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