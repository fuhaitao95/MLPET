import torch
from dgl.data import QM9EdgeDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


# Function to convert DGLGraph to PyG Data object


# InMemoryDataset for handling the converted PyG Data objects
class QM9PygDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(QM9PygDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # 加载预处理后的数据

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graph_dataset = QM9EdgeDataset()
        pyg_graphs = [self.dgl_to_pyg(graph) for graph in graph_dataset]
        data, slices = self.collate(pyg_graphs)
        torch.save((data, slices), self.processed_paths[0])

    def dgl_to_pyg(self, dgl_graph):
        edge_index = torch.stack(dgl_graph[0].edges())  # (src, dst)
        x = dgl_graph[0].ndata['attr']  # Node features
        edge_attr = dgl_graph[0].edata['edge_attr']  # Edge features
        # 假设存在一个用于边预测的标签，可以是 0 或 1，表示边是否存在
        edge_label = self.generate_edge_labels(dgl_graph[0])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label, y=dgl_graph[1])

    def generate_edge_labels(self, dgl_graph):
        # 根据 DGL 图生成边的标签 (是否存在边的二分类标签)
        # 假设你可以根据一些逻辑生成边的标签, 这里我们使用 1 表示所有边存在
        num_edges = dgl_graph.num_edges()
        edge_label = torch.ones(num_edges, dtype=torch.float)  # 用 1 表示边存在，可以根据需要调整
        return edge_label
# # Assume you have a list of DGLGraph objects
# dgl_graphs = [...]  # Replace with your list of DGLGraph
#
# # Create dataset instance
# dataset = QM9PygDataset(root='pretrain_dataset')
#
# # Create DataLoader for batch processing
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# # Iterate through the DataLoader
# for batch in train_loader:
#     print(batch)  # PyG Data objects
