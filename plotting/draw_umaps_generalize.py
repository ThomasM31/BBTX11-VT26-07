import umaps
import argparse
from pathlib import Path
import sys

from pathlib import Path
import importlib.util
import anndata as ad

def pipeline(
        to_include: list,
        proc_dir: Path,
        shared_dir_mode: bool
        ) -> None:
    
    path = Path(__file__).resolve().parent.parent / "pipeline_paths_generalize.py"
    spec = importlib.util.spec_from_file_location("ppaths", path)
    ppaths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppaths)
   
    
    pp = ppaths.PipelinePaths(shared_dir_mode)
    save_path = pp.figures_path / proc_dir
    save_path.mkdir(parents=True, exist_ok=True)

    label = 'all'
    proc_dataset = ad.read_h5ad(pp.compl_base / proc_dir / f'{label}.h5ad')
    print(proc_dataset)
    umaps.draw_umap(proc_dataset, label, save_path, 'processed', 'AD_status')
    
    label = 'GSE157827_merged'
    orig_dataset = ad.read_h5ad(pp.conv_data_path / f'{label}.h5ad')
    print(orig_dataset)
    label = 'all'
    umaps.draw_umap(orig_dataset, label, save_path, 'original', 'AD_status')

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