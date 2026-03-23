import os
import csv
import scanpy as sc

# Inställningar
input_path = "/data/shared/alzgene26/PathwaysLA/binn_connectivity.csv"
output_path = "/data/shared/alzgene26/PathwaysLA/binn_cutted.csv"
file_temp = "/data/shared/alzgene26/data/conv_data/OPCs.h5ad"

# 1. Ladda initiala gener från scanpy
if os.path.exists(file_temp):
    big = sc.read_h5ad(file_temp, backed="r")
    initial_genes = set(big.var_names[:30])
else:
    print("Filen hittades inte")
    initial_genes = set()
    
# 2. Bygg en graf (Adjacency List) för snabb uppslagning
# Vi mappar: käll-nod -> lista av mål-noder
adj_list = {}
with open(input_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        source, target = row[0], row[1]
        if source not in adj_list:
            adj_list[source] = []
        adj_list[source].append(target)

# 3. Hitta alla kopplingar med BFS (Breadth-First Search)
found_edges = []
visited_edges = set() # För att undvika duplikanter
queue = list(initial_genes) # Startnoder

print("Startar traversering...")

while queue:
    current_node = queue.pop(0)
    
    # Om noden har utgående kopplingar
    if current_node in adj_list:
        for target_node in adj_list[current_node]:
            edge = (current_node, target_node)
            
            # Lägg bara till om vi inte sett denna exakta koppling förut
            if edge not in visited_edges:
                visited_edges.add(edge)
                found_edges.append(edge)
                print("+1")
                
                # Lägg till mål-noden i kön för att utforska nästa nivå
                queue.append(target_node)

# 4. Skriv resultatet till filen en enda gång
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(found_edges)


print("gener")    
print(initial_genes)
print(f"Klar! Hittade {len(found_edges)} unika kopplingar.")