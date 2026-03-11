import pandas as pd
from pyvis.network import Network

file = "/data/shared/alzgene26/PathwaysLA/binn_cutted.csv"

# 1. Load CSV without headers
# header=None tells pandas to treat the first row as data, not labels
df = pd.read_csv(file, header=None) 

# 2. Initialize the network
# Added cdn_resources='remote' to fix that Jupyter/Chrome warning you saw
net = Network(directed=True, notebook=True, filter_menu=True, cdn_resources='remote')

# 3. Add edges using column indexes
for _, row in df.iterrows():
    # row[0] is the first column, row[1] is the second
    source = str(row[0])
    target = str(row[1])
    
    net.add_node(source, label=source)
    net.add_node(target, label=target)
    net.add_edge(source, target)

# 4. Save and open
net.show("graf.html")