import os
import csv
import scanpy as sc

# Inställningar
input_path = "/data/shared/alzgene26/PathwaysLA/binn_connectivity.csv"
output_path = "/data/shared/alzgene26/PathwaysLA/binn_cutted.csv"
file_temp = "/data/shared/alzgene26/data/conv_data/OPCs.h5ad"

# 1. Ladda initiala gener och rensa eventuella mellanslag
if os.path.exists(file_temp):
    big = sc.read_h5ad(file_temp, backed="r")
    # Vi lägger till .strip() för säkerhets skull
    initial_genes = set(str(g).strip() for g in big.var_names[:])
else:
    print("Filen hittades inte")
    initial_genes = set()
    
# 2. Bygg adj_list med .strip() på alla noder
adj_list = {}
with open(input_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 2: continue
        source, target = row[0].strip(), row[1].strip()
        if source not in adj_list:
            adj_list[source] = []
        adj_list[source].append(target)
        
# Rensa målfilen
open(output_path, "w").close()

# Starta traversering
layer = initial_genes.copy()
infeasible = initial_genes.copy()

print(f"Startar med {len(layer)} gener...")

while layer:
    temp_layer = set()
    edges_to_write = []
    
    # Processa nuvarande lager
    while layer:
        current_node = layer.pop()
        if current_node in adj_list:
            for target_node in adj_list[current_node]:
                if target_node not in infeasible:
                    temp_layer.add(target_node)
                    edges_to_write.append((current_node, target_node))
    
    # Om vi hittade nya kanter, skriv ner dem
    if edges_to_write:
        with open(output_path, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(edges_to_write) # Mycket snabbare än writerow i loop
        
        print(f"Hittade {len(edges_to_write)} nya kopplingar. Nästa lager har {len(temp_layer)} noder.")
    
    # Uppdatera inför nästa varv
    infeasible.update(temp_layer)
    layer = temp_layer