import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import full_atom_feature_dims, BondEncoder, AtomEncoder
from torch_geometric.data.data import BaseData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch.nn.functional as F


# region 基础编码器 卷积层
class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_channel, out_channel)
        self.bn = torch.nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x


class EmbedEncoder(torch.nn.Module):
    def __init__(self, embed_feature_dims_list, emb_dim):
        super(EmbedEncoder, self).__init__()

        self.embedding_list = torch.nn.ModuleList()
        if embed_feature_dims_list:
            for i, dim in enumerate(embed_feature_dims_list):
                emb = torch.nn.Embedding(dim, emb_dim)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.embedding_list.append(emb)
        else:
            for i, dim in enumerate(full_atom_feature_dims):
                emb = torch.nn.Embedding(dim, emb_dim)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.embedding_list.append(emb)

    # batch atomfeature 200个batch 56个原子 39个维度[200,56,39]
    # 39的维度信息在ChemUtils的GetAtomFeatures中
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.embedding_list[i](x[:, i])

        return x_embedding


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, bond_feature_size, emb_dim, encode_edge=True):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if bond_feature_size:
            self.bond_encoder = LinearEncoder(bond_feature_size, emb_dim)
        else:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)

        self.encode_edge = encode_edge

    def forward(self, x, edge_index, edge_attr):
        if self.encode_edge and edge_attr is not None:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = edge_attr
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, bond_feature_size, emb_dim, encode_edge=True):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if bond_feature_size:
            self.bond_encoder = LinearEncoder(bond_feature_size, emb_dim)
        else:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        self.encode_edge = encode_edge

        self.GRU_update = torch.nn.GRUCell(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if self.encode_edge and edge_attr is not None and edge_attr.numel() > 0:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = edge_attr

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr.numel() == 0:
            return norm.view(-1, 1) * F.relu(x_j)
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out, x):
        aggr_out = self.GRU_update(aggr_out, x)
        # torch.nn.GRUCell(args.hid_dim, args.hid_dim)
        return aggr_out


# endregion

class GNN_encoder(torch.nn.Module):
    """
    atom_feature_size (int): 如果为空则使用默认的
    Output:
        node representations
    """

    def __init__(self, num_layer, atom_feature_size, bond_feature_size, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False, gnn_type='gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if atom_feature_size:
            self.atom_encoder = LinearEncoder(atom_feature_size, emb_dim)
        else:
            self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(bond_feature_size, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(bond_feature_size, emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        # 处理第二层数据
        if isinstance(batched_data, BaseData):
            x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        elif len(batched_data) == 3:
            x, edge_index, edge_attr = batched_data[0], batched_data[1], batched_data[2]

        # [x, edge_index, edge_attr] = batched_data
        ### computing input node embedding
        temp_h = [self.atom_encoder(x)]
        h_list = temp_h
        for layer in range(self.num_layer):
            # 在卷积序列中的消息传播中
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        assert torch.isfinite(node_representation).all(), "node_representation has NaNs or Infs"

        return node_representation


class AtomFusionEncoder(nn.Module):
    def __init__(self, emb_dim, fusion_type='gate'):
        """
        Args:
            atom_input_dim (int): 原子索引特征维度（如 atom_x）
            with_atom_input_dim (int): 附加特征维度（如 atom_feature）
            emb_dim (int): 输出特征维度（与GNN输入保持一致）
            fusion_type (str): 'concat', 'sum', or 'gate'
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.atom_encoder = AtomEncoder(emb_dim)

        if fusion_type == 'concat':
            self.proj = nn.Linear(2 * emb_dim, emb_dim)
        elif fusion_type == 'gate':
            self.gate = nn.Linear(2 * emb_dim, 1)

    def forward(self, atom_x, atom_feature):
        """
        Args:
            atom_x: [N] long tensor, atom indices
            atom_feature: [N, F] float tensor, additional atom features
        Returns:
            fused_rep: [N, emb_dim]
        """
        x2 = self.atom_encoder(atom_feature)  # [N, emb_dim]

        if self.fusion_type == 'concat':
            out = torch.cat([atom_x, x2], dim=-1)  # [N, 2*emb_dim]
            return self.proj(out)  # [N, emb_dim]
        elif self.fusion_type == 'sum':
            return atom_x + x2
        elif self.fusion_type == 'gate':
            gate_input = torch.cat([atom_x, x2], dim=-1)  # [N, 2*emb_dim]
            alpha = torch.sigmoid(self.gate(gate_input))  # [N, 1]
            return alpha * atom_x + (1 - alpha) * x2
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")


class WithGNN_encoder(torch.nn.Module):
    """
    atom_feature_size (int): 如果为空则使用默认的
    Output:
        node representations
    """

    def __init__(self, num_layer, bond_feature_size, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(WithGNN_encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.fused_encoder = AtomFusionEncoder(
            emb_dim=emb_dim,
            fusion_type='concat'  # 或 'sum'、'gate'
        )

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GCNConv(bond_feature_size, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        # 处理第二层数据
        if isinstance(batched_data, BaseData):
            atom_x, atom_feature, edge_index, edge_attr = batched_data.h, batched_data.x, batched_data.edge_index, batched_data.edge_attr
        elif len(batched_data) == 4:
            atom_x, atom_feature, edge_index, edge_attr = batched_data[0], batched_data[1], batched_data[2], \
            batched_data[3]

        # [x, edge_index, edge_attr] = batched_data
        ### computing input node embedding
        h0 = self.fused_encoder(atom_x.squeeze(-1), atom_feature)  # [N, emb_dim]
        h_list = [h0]

        for layer in range(self.num_layer):
            # 在卷积序列中的消息传播中
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        assert torch.isfinite(node_representation).all(), "node_representation has NaNs or Infs"

        return node_representation

# class SMILES_Predictor(torch.nn.Module):
#     def __init__(self, embed_size, output_size=19):
#         super(SMILES_Predictor, self).__init__()
#
#         self.fc = nn.Linear(embed_size, output_size)  # 预测19个目标值
#
#     def forward(self, x):  # x = [batch_size, max_node_num, dim]
#         output = self.fc(x.mean(dim=1))  # (batch_size, 19)
#         return output
#
#
# class Edge_Predictor(torch.nn.Module):
#     def __init__(self, emb_dim):
#         super(Edge_Predictor, self).__init__()
#         self.edge_predictor = torch.nn.Linear(emb_dim * 2, 1)  # 用于预测边的存在与否
#
#     def forward(self, x, edge_index):  # x = [batch_size, max_node_num, dim]
#
#         # 获取边的两个节点的嵌入
#         src_node_emb = x[edge_index[0]]
#         tgt_node_emb = x[edge_index[1]]
#
#         edge_emb = torch.cat([src_node_emb, tgt_node_emb], dim=-1)
#
#         # 对节点对嵌入进行拼接，并预测边的存在
#         edge_logits = self.edge_predictor(edge_emb)
#         return edge_logits
