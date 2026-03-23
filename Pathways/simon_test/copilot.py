import os
import csv
import scanpy as sc

# Inställningar
input_path = "/data/shared/alzgene26/PathwaysLA/binn_connectivity.csv"
output_path = "/data/shared/alzgene26/PathwaysLA/binn_cutted.csv"
file_temp = "/data/shared/alzgene26/data/conv_data/OPCs.h5ad"

# 1. Ladda initiala gener
if os.path.exists(file_temp):
    big = sc.read_h5ad(file_temp, backed="r")
    initial_genes = set(big.var_names[:])
else:
    print("Filen hittades inte")
    initial_genes = set()
# 2. Bygg adjacency list
adj_list = {}
with open(input_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        source, target = row[0], row[1]
        adj_list.setdefault(source, []).append(target)

# 3. BFS med nivåer
print("Startar nivå-baserad BFS...")

levels = {}  # nod -> nivå
queue = []

# Initiera nivå 1
for g in initial_genes:
    levels[g] = 1
    queue.append(g)

found_edges = []

while queue:
    current = queue.pop(0)
    current_level = levels[current]

    if current in adj_list:
        for target in adj_list[current]:

            # Om target redan har en nivå, hoppa över (förhindrar nästling)
            if target in levels:
                continue

            # Tilldela target nästa nivå
            levels[target] = current_level + 1

            # Spara kanten
            found_edges.append((current, target))

            # Lägg target i kön
            queue.append(target)

# 4. Skriv resultat
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(found_edges)

print("gener")
print(initial_genes)
print(f"Klar! Hittade {len(found_edges)} nivå-baserade kopplingar.")
print(f"Antal nivåer: {max(levels.values())}")