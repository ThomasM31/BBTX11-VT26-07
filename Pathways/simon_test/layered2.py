import os
import csv
import scanpy as sc

# Inställningar
input_path = "/data/shared/alzgene26/PathwaysLA/binn_connectivity.csv"
output_path = "/data/shared/alzgene26/PathwaysLA/binn_cutted.csv"
file_temp = "/data/shared/alzgene26/data/conv_data/OPCs.h5ad"

# 1. Ladda initiala gener (Layer 0)
initial_genes = set()
if os.path.exists(file_temp):
    big = sc.read_h5ad(file_temp, backed="r")
    initial_genes = set(str(g).strip() for g in big.var_names[:30])
else:
    print("Filen hittades inte")

# 2. Läs in hela nätverket
adj_list = {}
with open(input_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 2: continue
        src, tgt = row[0].strip(), row[1].strip()
        if src not in adj_list: adj_list[src] = []
        adj_list[src].append(tgt)

# 3. Beräkna maximalt djup för varje nod (Longest Path)
# node_depth lagrar det största avståndet från en initial gen
node_depth = {gene: 0 for gene in initial_genes}
queue = list(initial_genes)

print("Beräknar hierarki...")

idx = 0
while idx < len(queue):
    current = queue[idx]
    idx += 1
    
    current_depth = node_depth[current]
    
    if current in adj_list:
        for target in adj_list[current]:
            # Om vi hittar en väg som är längre än den tidigare kända, 
            # uppdatera djupet och lägg till i kön för att uppdatera dess barn
            if target not in node_depth or node_depth[target] < current_depth + 1:
                node_depth[target] = current_depth + 1
                queue.append(target)
                
    # Säkerhetsspärr för att undvika oändliga loopar vid cirkulära beroenden
    if idx > 1000000: 
        print("Varning: Möjlig cirkulär referens detekterad. Avbryter expansion.")
        break

# 4. Extrahera kanter baserat på det slutgiltiga djupet
# Vi vill bara behålla kanter (u -> v) där depth(v) > depth(u)
final_edges = []
for src, targets in adj_list.items():
    if src in node_depth:
        for tgt in targets:
            if tgt in node_depth and node_depth[tgt] > node_depth[src]:
                final_edges.append((src, tgt))

# 5. Statistik och spara
depth_counts = {}
for d in node_depth.values():
    depth_counts[d] = depth_counts.get(d, 0) + 1

print("\nNätverksstruktur:")
for d in sorted(depth_counts.keys()):
    print(f"Lager {d}: {depth_counts[d]} noder")

with open(output_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(final_edges)

print(f"\nKlart! Totalt {len(final_edges)} hierarkiska kopplingar sparade.")