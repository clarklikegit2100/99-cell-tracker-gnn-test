import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

# Load your .pt file
data = torch.load('pytorch_geometric_data.pt', map_location='cpu')

# If the file contains a list, use the first element
if isinstance(data, list):
    data = data[0]

# Convert to NetworkX graph
G = to_networkx(data, to_undirected=True)

# Visualize
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_size=50, node_color='skyblue', edge_color='gray')
plt.title("Graph Visualization from PyTorch Geometric Data")
plt.show()
