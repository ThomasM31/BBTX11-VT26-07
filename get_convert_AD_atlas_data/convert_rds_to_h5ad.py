import anndata2ri
from rpy2.robjects import r
import os
import scanpy as sc
import anndata
import argparse
import os

# import
r('library(Seurat)')

## Script to convert a .rds file to a python readable anndata format, saved as a .hd5a file
## NOTE: R and the package Seurat need to be installed on your machine to run this script
## This can be done using a conda env

def rds_to_hd5a(path_to_rds, new_file):
    with anndata2ri.converter.context():
        # open file
        r(f'seurat_obj <- readRDS("{path_to_rds}")')
        
        # conversion step
        adata = r('as.SingleCellExperiment(seurat_obj)')

        # prevent crash due to modern datatypes
        anndata.settings.allow_write_nullable_strings = True
        
        # save
        adata.write(new_file, compression="gzip")

        # ensure that the file was properly saved
        test_load = sc.read_h5ad(new_file)
        print(f"Successfully saved {test_load.n_obs} cells")
        return test_load


def run(arg):
    user = os.environ.get('USER') or os.environ.get('USERNAME')
    base_path = Path("/data/users") / user / "kand/data/raw_data/"
    path = str(base_path)

    ftype_old = ".rds"
    ftype_new = ".h5ad"

    f_load = os.path.join(path, arg)
    f_load += ftype_old

    f_save = os.path.join(path, arg)
    f_save += ftype_new
    
    adata = rds_to_hd5a(f_load, f_save)

if __name__ == '__main__':
    ## Argument from bash script

    parser = argparse.ArgumentParser(description="Convert .rds to .hd5a")
    parser.add_argument("filename", help="The base name of the rds file (no extension)")
    args = parser.parse_args()

    print(f"Processing: {args.filename}")

    run(args.filename)