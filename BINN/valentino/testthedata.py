import anndata as ad
filepath='/data/shared/alzgene26/data/processed_data/completed/mg_200_mc_200_mhvg1000/vasc.h5ad'


data=ad.read_h5ad(filepath)

print(data.uns['pseudo'].obs['AD_status'])


import anndata as ad

filepath = '/data/shared/alzgene26/data/processed_data/completed/mg_200_mc_200_mhvg1000/astro.h5ad'
adata = ad.read_h5ad(filepath)

pseudo_obs = adata.uns['pseudo'].obs
print("pseudo_obs columns:", pseudo_obs.columns.tolist())
print("'AD_status' in columns:", 'AD_status' in pseudo_obs.columns)

# Check for hidden characters or case issues
for col in pseudo_obs.columns:
    print(repr(col))  # repr shows hidden characters/spaces