import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx


pt_file=r"pt-read\pytorch_geometric_data.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.load(pt_file, map_location=device, weights_only=False)

print(type(data))

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

print("Node feature shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)


# # 加载边权重（raw_output.pt）
# edge_weights = torch.load(r'pt-read\raw_output.pt', map_location='cpu')
# edge_weights = edge_weights.detach().cpu().numpy()
#
# # 生成默认布局（spring layout）
# pos = nx.spring_layout(G, seed=42)
#
# # 可视化图
# plt.figure(figsize=(8, 8))
# edges = list(G.edges)
# print(f"Total number of edges: {len(edges)}")
# print("First 20 edges:")
# for edge in edges[:20]:
#     print(edge)
#
#
# # 注意：NetworkX 边是 tuple 顺序，PyG 的 edge_index 顺序可能不同（默认认为一致）
# nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue')
# nx.draw_networkx_edges(
#     G,
#     pos,
#     edgelist=edges,
#     width=1.5,
#     edge_color=edge_weights,
#     edge_cmap=plt.cm.plasma  # 可选 colormap
# )
# plt.title("Edge weights visualization from raw_output.pt")
# plt.axis('off')
# plt.show()


# import torch
#
# pt_file = r'pt-read\raw_output.pt'
# data = torch.load(pt_file, map_location='cpu', weights_only=False)
#
# print("Type:", type(data))
# print("Keys (if dict):", data.keys() if isinstance(data, dict) else None)
# print("raw_output.pt type:", type(data))
# print("Shape:", data.shape if hasattr(data, 'shape') else "No shape")
# print("Sample content:", data[:5] if hasattr(data, '__getitem__') else "Not indexable")
#
# embedding = torch.load(r'pt-read\raw_output.pt')  # shape: [760, 2] 假设是2D
#
# # 可视化时用坐标布局表示嵌入
# pos = {i: embedding[i].tolist() for i in range(embedding.shape[0])}
#
# plt.figure(figsize=(8, 8))
# nx.draw(G, pos, node_size=30, node_color='orange')
# plt.title("Graph Layout by Embedding (raw_output.pt)")
# plt.show()


# 构造 networkx 图
G = nx.Graph()
G.add_edges_from(edges)

# 加载边权重
edge_weights = torch.load(r'pt-read\raw_output.pt').numpy()

# 可视化
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue')
nx.draw_networkx_edges(
    G, pos,
    edgelist=edges,
    width=1.5,
    edge_color=edge_weights,
    edge_cmap=plt.cm.plasma
)
plt.title("Edge weights from raw_output.pt (aligned to edge_index order)")
plt.axis('off')
plt.show()