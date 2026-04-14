import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

nodes = ["PhD", "Career Outside Science",  "Early Career Researcher", "Research Staff",
         "Permanent Research Staff",  "Professor",  "Non-Academic Research"]
nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
edges = [
    (0, 1, 53), (0, 2, 47), (2, 6, 17), (2, 3, 30), (3, 1, 22.5), (3, 4, 3.5), (3, 6, 4.), (4, 5, 0.45)
]

value_dim = hv.Dimension('Percentage', unit='%')
careers = hv.Sankey((edges, nodes), ['From', 'To'], vdims=value_dim)

sankey_obj = careers.opts(
    opts.Sankey(labels='label', label_position='right', width=900, height=300, cmap='Set1',
                edge_color=value_dim('To').str(), node_color=value_dim('index').str()))

careers.save(sankey_obj, 'sample.html', backend='bokeh')