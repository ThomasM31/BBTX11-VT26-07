import sys
import scanpy as sc
import pandas as pd
import pandas as pd, numpy as np
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path += [str(ROOT), str(ROOT / 'preprocessing'), str(ROOT / 'Pathways')]

import preprocess as pre
from MaskCreator import PathwayNetwork
import data_handling as dh

def load_pathway_conns(connectivity_file):
    """Lifted out of MaskCreator.main so it can be reused."""
    df = pd.read_csv(connectivity_file).dropna(subset=['child', 'parent']).astype(str)
    is_pw = df['child'].str.contains('R-HSA-')
    return (list(df[is_pw].itertuples(index=False, name=None)),    # pathway-pathway
            list(df[~is_pw].itertuples(index=False, name=None)))   # gene-pathway

def build_masks(genes, conns, n_layers=4) -> dict:
    pn = PathwayNetwork(set(genes), conns[0], conns[1])
    return {f"df{i}": m for i, m in enumerate(pn.get_connectivity_matrices(n_layers))}

def preprocess_split(train_cells, eval_cells: dict, nr_hvgs, conns, n_layers=4):
    """Fit everything on train_cells; apply to every set in eval_cells."""
    hvgs = pre.select_hvgs(train_cells, nr_hvgs)                 # fit: train only
    sets = {'train': train_cells, **eval_cells}
    for name, a in sets.items():
        a = pre.aggregate_to_pseudobulk(pre.filter_from_genes(a, hvgs).copy(), bulk_by='mean')
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        sets[name] = a
    sets['train'], scaler = pre.standardize(sets['train'])       # fit: train only
    for name in eval_cells:
        sets[name] = pre.standardize(sets[name], scaler)

    masks = build_masks(sets['train'].var_names, conns, n_layers)
    sets = dh.rollup_to_patient_level(sets)
    sets = dh.pad_align_data(dh.subset_genes(sets, masks['df0']), masks['df0'])
    return sets, masks, {'hvgs': list(hvgs), 'scaler': scaler}

def run_CV_nested_HVG_sel(
    adata,
    device,
    eval_metric: str,
    conns,
    n_hvgs : int = 1000,
    n_layers : int = 4,
    batch_size: int = 32,
    activation_fn = nn.Tanh(),
    k_outer: int = 5,
    k_inner: int = 5,
    epochs: int = 150,
    n_trials: int = 30,
    save_path = None, 
    data_info = '', 
    time = ''
    ) -> tuple[np.ndarray, list[dict]]:

    metric = dh.get_metric(eval_metric)
    subj = adata.obs[['subject', 'label']].drop_duplicates().reset_index(drop=True)

    outer = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=0)
    outer_score, chosen = [], []

    for fold, (tr, va) in enumerate(outer.split(subj, subj['label'])):
        tr_cells = adata[adata.obs['subject'].isin(subj['subject'].iloc[tr])]
        va_cells = adata[adata.obs['subject'].isin(subj['subject'].iloc[va])]

        data, masks, artifacts = preprocess_split(
            tr_cells, 
            {'val':va_cells}, 
            n_hvgs, 
            conns, 
            n_layers)

        _, in_features, layers_list, tensor_masks = dh.compute_features(masks, device)

        # INNER: tune hyperparams + epoch on outer_train only
        best = dh.hyperparameter_tuning_optuna(
            data['train'], in_features, layers_list, tensor_masks, device, 
            metric=metric, batch_size=batch_size, activation_fn=activation_fn,
            k=k_inner, epochs=epochs, n_trials=n_trials
        )

        # refit a fresh model on all of outer train
        model, crit, opt, sched = dh.create_model(
            in_features, layers_list, tensor_masks, device,
            lr=best['lr'], weight_decay=best['weight_decay'],
            dropout=best['dropout'], activation_fn=activation_fn,
            epochs=best['tune_total_epochs']
        )

        dh.train_fixed_epochs(
            model, 
            dh.make_annloader(data['train'], batch_size, shuffle=True),
            crit, opt, sched, device, n_epochs=best['epochs'],
            save_path=save_path, data_info=f'{data_info}_outer{fold}', time=time)

        # evaluate once on the untouched outer fold
        _, _, score = dh.evaluate_model(
            model, 
            dh.make_annloader(data['val'], batch_size, shuffle=False), 
            device, metric=metric)
        
        outer_score.append(score)
        chosen.append({**best, 'n_genes': in_features, 'layers': layers_list})
        print(f'Outer fold {fold+1}: score {eval_metric} = {score:.4f}, params={best}, genes={in_features}, layers={layers_list}')

    print(f"Nested CV Score ({eval_metric}): {np.mean(outer_score):.4f} +/- {np.std(outer_score):.4f}")
    return np.array(outer_score), chosen