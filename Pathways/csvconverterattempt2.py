import pandas as pd

# Läs in GMT-filen (Gener -> Pathways)
print("Läser in GMT-filen...")
edges = []
with open('/data/shared/alzgene26/PathwaysLA/ReactomePathways.gmt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        pathway_id = parts[1]
        genes = parts[2:]
        for gene in genes:
            edges.append({'child': gene, 'parent': pathway_id})

df_gene_pathway = pd.DataFrame(edges)

# Läs in Relations-filen (Pathway -> Parent Pathway)
print("Läser in hierarkin...")
df_pathway_relation = pd.read_csv('/data/shared/alzgene26/PathwaysLA/ReactomePathwaysRelation.txt', sep='\t', header=None)
df_pathway_relation.columns = ['parent', 'child']
# Vi vänder på dem så de matchar 'child' -> 'parent' formatet
df_pathway_relation = df_pathway_relation[['child', 'parent']]

# Slå ihop allt till en stor konnektivitetslista
print("Bygger den slutgiltiga matrisen...")
full_network = pd.concat([df_gene_pathway, df_pathway_relation], ignore_index=True)

# Save:
Save_Network = False
if Save_Network == True:
    full_network.to_csv('/data/shared/alzgene26/PathwaysLA/binn_connectivity.csv', index=False)
    print(f"Klart! Filen 'binn_connectivity.csv' har skapats med {len(full_network)} kopplingar.")
else:
    print(f"Klart! Ingen fil sparades, ty Save_Network var satt till False.")