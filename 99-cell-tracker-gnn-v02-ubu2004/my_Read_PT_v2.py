import torch
import matplotlib.pyplot as plt
import networkx as nx
from imageio import save
from torch_geometric.utils import to_networkx

# === Step 1: Load PyG graph data ===
pt_file=r"pt-read/01_RES_inference_U373/pytorch_geometric_data.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# modified by jz 20250710
#data = torch.load(pt_file, map_location=device, weights_only=False)

data = torch.load(pt_file, map_location=device)

print("Graph loaded.")
print(f"Node feature shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")

# === Step 2: Load edge weights ===
pt_weights_file = r'pt-read/01_RES_inference_U373/raw_output.pt'
edge_weights = torch.load(pt_weights_file, map_location='cpu').detach().cpu().numpy()
print("Edge weights loaded.")
print(f"Edge weights shape: {edge_weights.shape}")

# === Step 3: Build edge list with original PyG edge_index order ===
edges = [(int(s), int(t)) for s, t in data.edge_index.t().tolist()]

# Save to text file
output_file = r'pt-read\01_RES_inference\edges.txt'
with open(output_file, 'w') as f:
    for src, tgt in edges:
        f.write(f"{src} {tgt}\n")

print(f"✅ Saved {len(edges)} edges to {output_file}")

print(f"Total edges: {len(edges)}")

# === Step 4: Create NetworkX graph from edge list ===
G = nx.Graph()
G.add_edges_from(edges)

# === Step 5: Visualize with edge colors ===
pos = nx.spring_layout(G, seed=42)  # or use nx.kamada_kawai_layout(G)

plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue')
nx.draw_networkx_edges(
    G, pos,
    edgelist=edges,
    width=1.5,
    edge_color=edge_weights,
    edge_cmap=plt.cm.plasma,
)
plt.title("Graph with Edge Weights from raw_output.pt")
plt.axis('off')
plt.show()
