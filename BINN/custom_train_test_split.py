import os
from pathlib import Path
import argparse

import scanpy as sc
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

import anndata as ad
from anndata.experimental import AnnCollection
ad.settings.allow_write_nullable_strings = True

def read_files(to_include: list[int], filepath: str) -> dict:
    ### To read from current user's folders
    #user = os.environ.get('USER') or os.environ.get('USERNAME')
    #base_path = Path("/data/users") / user / "kand/data/processed_data/"
    #filepath = str(base_path)

    labels = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']

    included_labels = [labels[i] for i in to_include]
    print(f'Labels to include: {included_labels}')

    # labels as keys, adata objects as values, e.g.
    # {'immune': adata_obj, 'oligo': adata_obj}
    datasets = {
        label: None 
        for label in included_labels
    }

    for label in included_labels:
        print(f'Reading {label}')
        f = label + '.h5ad'
        f = os.path.join(filepath,f)
        datasets[label] = ad.read_h5ad(f)

    return datasets

def create_encoded_collection(datasets: dict) -> AnnCollection:
    # Encoding by general cell type
    cell_type_low_res_encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)

    # Encoding by fine-grained cell type
    cell_type_high_res_encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)

    # Encode disease status
    status_encoder = LabelEncoder()

    ### create converter
    converters = {
        'obs': {
            # One-hot encode the covariates
            'cell_type_low_res': lambda x: cell_type_low_res_encoder.transform(x.to_numpy()[:, None]),
            'cell_type_high_res': lambda x: cell_type_high_res_encoder.transform(x.to_numpy()[:, None]),

            # Label encode the target (AD_status)
            'AD_status': status_encoder.transform
        }
    }

    ### We can join several data sets together in a collection
    # This is the way to do it with a large dataset
    collection = AnnCollection(
        datasets,
        join_vars='outer', # keep all records
        label='cell_type_low_res',
        convert=converters, # do encoding when creating the collection
        indices_strict=False # in case there exist cells with same ID (unlikely here?)
    )

    # for brevity
    collection.obs.rename(columns={'cell_type_high_resolution': 'cell_type_high_res'}, inplace=True)

    ### fit encodings
    cell_type_low_res_encoder.fit(collection.obs["cell_type_low_res"].to_numpy()[:,None])
    cell_type_high_res_encoder.fit(collection.obs["cell_type_high_res"].to_numpy()[:,None])
    status_encoder.fit(collection.obs["AD_status"])

    return collection

def custom_train_test_split(collection: AnnCollection, train_size: float):
    # This ensures that all rows with the same 'subject' stay together
    # Prevents the model from learning the personal signature of the individual subjects,
    # instead of markers for disease
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=34)

    # Perform the split
    # We pass the 'AD_status' to the groups context to maintain balance
    train_idx, test_idx = next(gss.split(
        X=collection.obs, 
        y=collection.obs['AD_status'], 
        groups=collection.obs['subject']
    ))

    # Create training and testing objects
    train_adata = collection[train_idx]
    test_adata = collection[test_idx]

    # Verify the results
    print(f"Train Subjects: {train_adata.obs['subject'].nunique()}")
    print(f"Test Subjects: {test_adata.obs['subject'].nunique()}")

    return train_adata, test_adata


def pipeline(to_include: list[int], filepath: str, train_size: float):
    '''Prepare train-test split from preprocessed .h5ad files'''
    datasets = read_files(to_include, filepath)
    
    print('Creating encoded collection')
    collection = create_encoded_collection(datasets)

    print('Do train-test split')
    train_adata, test_adata = custom_train_test_split(collection, train_size)
    
    print('Completed')
    return train_adata, test_adata

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
        description = "Create train-test split from preprocessed .h5ad data"
    )

    parser.add_argument(
        "train_size",
        help="float with min:0.0, max:1.0, default=0.8",
        type=range_limited_float_type,
        default=0.8
    )

    parser.add_argument(
        "to_include",
        help="indices to include: \n0=astro \n1=exc1 \n2=exc2 \n3=exc3 \n4=immune \n5=inhi \n6=oligo \n7=opcs \n8=vasc",
        nargs='+',
        type=int
    )

    parser.add_argument(
        "filepath",
        help="directory where anndata files are located",
        type=str
    )
    
    args = parser.parse_args()

    to_include = [int(arg) for arg in args.to_include]
    filepath = args.filepath
    train_adata, test_adata = pipeline(to_include, filepath, args.train_size)
    