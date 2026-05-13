import umaps
import argparse
from pathlib import Path
import sys

from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
import scanpy as sc

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from preprocessing import preprocess as pre

def draw_umap(adata, 
              label: str, 
              filepath: Path,
              stage: str, 
              color_by:str) -> None:
    print(f'Drawing umaps for {label} ({stage}, {color_by}).')

    f = filepath / f'{label}_{stage}_{color_by}'
    title = f'{label.capitalize()} ({stage}) colored by "{color_by}".'

    sc.tl.pca(adata, svd_solver='arpack')
    n_pcs = 30 if adata.n_obs > 30 else adata.n_obs - 1
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.pl.umap(adata, 
                title=title,
                color=[color_by], 
                show=False,
                legend_loc="upper left").figure.savefig(f)
    

def draw_combined_umaps(orig_adata, proc_adata, label, save_path):
    # Konvertera AD_status till läsbara kategorier för båda objekten
    for data in [orig_adata, proc_adata]:
        if 'AD_status' in data.obs.columns:
            # Säkerställ att det är strängar/kategorier för diskret färgskala
            data.obs['AD_status'] = data.obs['AD_status'].astype(str).map({
                '1.0': 'AD',
                '1': 'AD',
                '0.0': 'Ej AD',
                '0': 'Ej AD'
            }).astype('category')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    tasks = [
        (orig_adata, 'Oprocessad', 'cell_type_high_resolution', axes[0, 0]),
        (orig_adata, 'Oprocessad', 'AD_status', axes[0, 1]),
        (proc_adata, 'Processad', 'cell_type_high_resolution', axes[1, 0]),
        (proc_adata, 'Processad', 'AD_status', axes[1, 1])
    ]

    for adata, stage, color, ax in tasks:
        # Beräkningar
        sc.tl.pca(adata, svd_solver='arpack')
        n_pcs = 30 if adata.n_obs > 30 else adata.n_obs - 1
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs=n_pcs)
        sc.tl.umap(adata)

        # Snyggare titlar
        clean_label = 'AD-status' if color == 'AD_status' else 'Celltyp'
        
        sc.pl.umap(adata, 
                   color=color, 
                   ax=ax, 
                   show=False, 
                   title=f"{stage}: {clean_label}")

    plt.tight_layout()
    fig.savefig(save_path / f"{label}_combined_umaps.png", bbox_inches='tight')
    plt.close(fig)

def pipeline(
        to_include: list,
        proc_dir: Path,
        shared_dir_mode: bool,
        train_test_mode: bool
        ) -> None:
    
    path = Path(__file__).resolve().parent.parent / "pipeline_paths.py"
    spec = importlib.util.spec_from_file_location("ppaths", path)
    ppaths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppaths)   
    
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

    datasets_orig = pre.get_datasets(included_labels)
    # read h5ad files, add to datasets dict
    pre.read_files(datasets_orig, pp.conv_data_path)
    orig_dataset = datasets_orig[label]

    if 'AD_status' in proc_dataset.obs.columns:
        mapping = proc_dataset.obs[['subject', 'AD_status']].drop_duplicates().set_index('subject')['AD_status']
        orig_dataset.obs['AD_status'] = orig_dataset.obs['subject'].map(mapping)

    draw_combined_umaps(orig_dataset, proc_dataset, label, save_path)
    #umaps.draw_umap(orig_dataset, label, save_path, 'original', 'cell_type_high_resolution')
    #umaps.draw_umap(orig_dataset, label, save_path, 'original', 'AD_status')
    #umaps.draw_umap(proc_dataset, label, save_path, 'processed', 'cell_type_high_resolution')
    #umaps.draw_umap(proc_dataset, label, save_path, 'processed', 'AD_status')

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

    # Optional argument
    parser.add_argument(
        "--train_test_mode", 
        type=bool,
        default=True,
        help="To run on data from train test pipeline (True, default) or generalizability data (False)"
    )

    args = parser.parse_args()

    to_include = args.to_include
    proc_dir = args.proc_dir
    shared_dir_mode = args.shared_dir_mode
    train_test_mode = args.train_test_mode

    pipeline(
        to_include,
        proc_dir,
        shared_dir_mode,
        train_test_mode
        )