import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. Ladda data
df = pd.read_csv("/data/shared/alzgene26/PathwaysLA/binn_cutted.csv", header=None)
G = nx.from_pandas_edgelist(df, source=0, target=1, create_using=nx.DiGraph())

# 2. Beräkna lager (vilket steg från startgenen)
# Vi mappar varje nod till dess minsta avstånd från en "rot-nod"
# Om vi inte vet rötterna, tar vi noder som inte har några inkommande kanter
roots = [n for n, d in G.in_degree() if d == 0]
levels = {}

for root in roots:
    for node, dist in nx.single_source_shortest_path_length(G, root).items():
        levels[node] = max(levels.get(node, 0), dist)

# Lägg till nivån som ett attribut på noden
for node, level in levels.items():
    G.nodes[node]['layer'] = level

# 3. Skapa layouten
# multipartite_layout använder 'layer'-attributet för att ställa upp dem i kolumner
pos = nx.multipartite_layout(G, subset_key="layer")

# 4. Rita
plt.figure(figsize=(16, 10))

# Rita kanter med lite transparens så de inte tar över
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=True)

# Rita noder
nx.draw_networkx_nodes(G, pos, 
                       node_color=[levels.get(n, 0) for n in G.nodes()], 
                       cmap=plt.cm.viridis, 
                       node_size=600)

# Rita etiketter
nx.draw_networkx_labels(G, pos, font_size=7)

plt.title("Hierarkisk layout (Vänster till Höger)")
plt.axis('off')
plt.savefig("/data/shared/alzgene26/PathwaysLA/graf_layered.png", dpi=300)
print("Klart! Filen 'graf_layered.png' är skapad utan PyGraphviz.")