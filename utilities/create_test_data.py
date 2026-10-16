import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import anndata as ad
from pathlib import Path

def fake_gene():
    gene = []
    for i in range(5):
        i = np.random.randint(65,91)
        gene.append(chr(i))        
    return "".join(gene)

# list of genes
gene_names = pd.read_csv("./gene_names.csv")
fake_df = pd.DataFrame({'gene_ids': [fake_gene() for _ in range(100)]})
gene_names = pd.concat([gene_names, fake_df], ignore_index=True)

genes = gene_names.iloc[:, 0].tolist()

# list of subjects
metadata = pd.read_csv("../simulated/data/supplementary_data/individual_metadata_deidentified.tsv", sep="\t")
subject_ids = metadata.iloc[:, 0].dropna().unique().tolist()

n_cells = 10000
n_genes = len(genes)

# Updated metadata schema to include cognitive diagnosis, Braak stage, and CERAD score
obs_columns = [
    "subject",
    "msex",
    "age_death",
    "pmi",
    "race",
    "Pathologic_diagnosis_of_AD",
    "cogdx",
    "braaksc",
    "ceradsc"
]

# Pre-generate persistent metadata per subject ID
subject_meta = {
    s: {
        "msex": np.random.choice([0, 1]),
        "age_death": np.random.randint(65, 95),
        "pmi": round(np.random.uniform(2.5, 12.0), 2),
        "race": np.random.choice([1, 2]),
        "Pathologic_diagnosis_of_AD": np.random.choice(["yes", "no"]),
        "cogdx": np.random.choice([1, 2, 4]),
        "braaksc": np.random.choice([0, 1, 2, 3, 4, 5, 6]),
        "ceradsc": np.random.choice([1, 2, 3, 4])
    }
    for s in subject_ids
}

def create_sample_obs(prefix, subjects_pool):
    selected_subjects = np.random.choice(subjects_pool, size=n_cells)
    
    data = []
    for s in selected_subjects:
        meta = subject_meta[s].copy()
        meta["subject"] = s
        data.append(meta)
        
    df = pd.DataFrame(data, index=[f"{prefix}_cell_{i}" for i in range(n_cells)])
    return df[obs_columns]

# Helper function to generate count matrices with simulated HVGs
def generate_counts_with_hvgs(n_cells, n_genes, hvg_fraction=0.10, base_lam=1.5):
    n_hvgs = int(n_genes * hvg_fraction)
    
    base_expression = np.random.gamma(shape=2.0, scale=base_lam / 2.0, size=n_genes)
    hvg_indices = np.random.choice(n_genes, size=n_hvgs, replace=False)
    
    counts = np.zeros((n_cells, n_genes), dtype=np.int32)
    
    non_hvg_indices = np.setdiff1d(np.arange(n_genes), hvg_indices)
    for g in non_hvg_indices:
        counts[:, g] = np.random.poisson(lam=base_expression[g], size=n_cells)
        
    for g in hvg_indices:
        cell_rates = np.random.gamma(shape=0.5, scale=base_expression[g] / 0.5, size=n_cells)
        counts[:, g] = np.random.poisson(lam=cell_rates)
        
    return csr_matrix(counts)

# Generate sparse count matrices with HVGs included
X1 = generate_counts_with_hvgs(n_cells, n_genes, hvg_fraction=0.10, base_lam=1.5)
X2 = generate_counts_with_hvgs(n_cells, n_genes, hvg_fraction=0.10, base_lam=2.0)

# Build AnnData objects
var = pd.DataFrame(index=genes)
obs1 = create_sample_obs("batch1", subject_ids)
obs2 = create_sample_obs("batch2", subject_ids)

adata1 = ad.AnnData(X=X1, obs=obs1, var=var)
adata2 = ad.AnnData(X=X2, obs=obs2, var=var)

print(adata1)

p = Path("../simulated/data/raw_data/")
adata1.write_h5ad(p / "test1.h5ad")
adata2.write_h5ad(p / "test2.h5ad")

print(f'Simulated test data saved to {p}')