import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set, AttentionalAggregation

from Utils.motif_utils import flatten_2d_list_with_batch, flatten_2d_list_with_batch_motifeat
from model.graph_model import GNN_encoder, WithGNN_encoder


# 该gnn用于训练基序的提取并与用基序提取算法训练过的数据集进行训练
# 不确定是否用于预训练
class Motify_SMILES_Predictor(nn.Module):

    def __init__(self, num_tasks, atom_feature_size=None, bond_feature_size=None, num_layer=5, emb_dim=300,
                 gnn_type='gcn', residual=True, drop_ratio=0.5, JK="last", graph_pooling="max",
                 task_type="classification", evaluator=None, lr=0.001):
        '''
            num_tasks (int): number of labels to be predicted
            atom_feature_size (int): 指定原子特征的长度，默认不设置
            bond_feature_size (int): 指定原子特征的长度，默认不设置
            emb_dim (int): 中间卷积层的特征维度
            gnn_type (str): 默认编码器的卷积类型
        '''

        super(Motify_SMILES_Predictor, self).__init__()

        # 属性初始化
        self.isencoderloading = False
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        # region pytorch-lightning 参数
        self.task_type = task_type
        self.evaluator = evaluator
        self.lr = lr

        self.y_true_val = []
        self.y_pred_val = []
        self.y_true_test = []
        self.y_pred_test = []
        # endregion
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        # todo 确实是否需要节点gnn_encoder 目前做对比实验用
        # self.gnn_encoder = GNN_encoder(self.num_layer, atom_feature_size, bond_feature_size, self.emb_dim, JK=self.JK,
        #                                drop_ratio=self.drop_ratio,
        #                                residual=residual,
        #                                gnn_type=gnn_type)
        # todo 用于训练基序内部的特征 如果用x则是atom_feature_size 如果是h_node 则是emb_dim
        # self.motif_classifier = Motify_Classifier(gnn_type=gnn_type, num_tasks=self.num_tasks,
        #                                           num_layer=self.num_layer,
        #                                           emb_dim=self.emb_dim,
        #                                           drop_ratio=self.drop_ratio, graph_pooling=self.graph_pooling)
        # self.motif_encoder = GNN_encoder(self.num_layer, None, bond_feature_size, self.emb_dim, JK=self.JK,
        #                          drop_ratio=self.drop_ratio,
        #                          residual=residual,
        #                          gnn_type=gnn_type)
        self.motif_encoder = WithGNN_encoder(
            self.num_layer,
            None,
            self.emb_dim,
            JK='last',
            drop_ratio=self.drop_ratio,
            residual=True,
        )

        # for param in self.motif_encoder.parameters():
        #     param.requires_grad = False

        # todo 用于子图训练的gnn_encoder
        self.graph_encoder = GNN_encoder(self.num_layer, self.emb_dim, bond_feature_size, self.emb_dim, JK=self.JK,
                                         drop_ratio=self.drop_ratio,
                                         residual=residual,
                                         gnn_type=gnn_type)

        # todo 对子图进行池化操作
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
            self.motif_pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
            self.motif_pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
            self.motif_pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
            self.motif_pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
            self.motif_pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # for param in self.motif_pool.parameters():
        #     param.requires_grad = False

        # 输出结果的维度处理
        if graph_pooling == "set2set":  # todo 可以时时其他的
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, logits, atom_mask, batched_data):
        B, N, D = logits.size()
        # device = atom_feat.device
        #
        # 将掩码展开为一维：只保留真实原子对应的行
        flat_mask = atom_mask.view(-1)  # [B*N]
        flat_atom_feat = logits.reshape(-1, D)  # [B*N, D]
        masked_atom_feat = flat_atom_feat[flat_mask]  # [M, D]，M为真实原子数量
        #         masked_atom_feat = self.pool(masked_atom_feat, batched_data.batch)  # [B, D]

        #         # 残差块 + 预测头
        #         h_final = self.graph_pred_linear(masked_atom_feat)
        # 将基序内部的motif特征训练转为batch形式可能是使训练有效的方法
        # 针对原子在整图中的特征进行训练
        # h_node = self.gnn_encoder(batched_data)  # todo 理论上应该不用训练 看具体效果 但是对比实验会用上所以不要删
        # 将分子水平基序训练制作成批次形式

        # 基序特征收集
        # 如果顺利的话 在这里对edge_cliques做一层处理
        motif_list, motif_batch = flatten_2d_list_with_batch_motifeat(batched_data.motif_list, masked_atom_feat,
                                                                      batched_data.cliques, batched_data,
                                                                      batched_data.x.device)  # todo 这个地方可以用h_node 也可以用x
        # motif_list, motif_batch = flatten_2d_list_with_batch(batched_data.motif_list,
        #                                                      batched_data.x.device)  # todo 这个地方可以用h_node 也可以用x
        batched_motifs = Batch.from_data_list(motif_list)

        batched_motifsgraph = Batch.from_data_list(batched_data.g_motif_graph)
        if self.isencoderloading:
            # h_motif_graph = self.motif_classifier(batched_motifs)
            pass
        else:
            # 如果这个地方使用x作为基序特征 则基序的编码器需要换成atom编码器
            # todo x和h_node没有明显区别，先用x训练 这个moitf_encoder用作预训练特征器
            # h_motif = self.motif_encoder([masked_atom_feat, batched_motifs.edge_index, batched_motifs.edge_attr])
            # h_motif = self.motif_encoder(batched_motifs)
            h_motif = self.motif_encoder(
                [batched_motifs.motif_atom_feat, batched_motifs.x, batched_motifs.edge_index, batched_motifs.edge_attr])
            # 从[n_atoms, emb_dim] -> [n_batch, emb_dim]

            h_motif_graph = self.motif_pool(h_motif, batched_motifs.batch)

        # 图水平特征
        # 对上层基序图进行一次gnn_encoder [n_motif, emb_dim]
        # todo graph_encoder
        h_graph_node = self.graph_encoder(
            [h_motif_graph, batched_motifsgraph.edge_index, batched_motifsgraph.edge_attr])

        h_graph = self.pool(h_graph_node, batched_motifsgraph.batch)  # todo 不确定这里是否要进行pool操作 可能需要将基序内节点的特征整合成一维图特征
        h_final = self.graph_pred_linear(h_graph)
        return h_final


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        return residual + out


class SMILES_Graph_Predictor(nn.Module):
    def __init__(self, args, num_tasks, graph_pooling="attention"):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        self.num_tasks = num_tasks

        # GNN encoder 用于处理筛选出的有效原子特征
        self.gnn = WithGNN_encoder(
            args.predictor_layer_num,
            None,
            self.embed_dim,
            JK='last',
            drop_ratio=self.dropout,
            residual=True,
        )

        # 残差模块
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(self.embed_dim, self.dropout)
            for _ in range(2)
        ])

        # 图级表示池化层
        self.graph_pooling = graph_pooling

        # 池化
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(args.embed_dim, 2 * args.embed_dim),
                                            torch.nn.LayerNorm(2 * args.embed_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * args.embed_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(args.embed_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # 输出预测头
        self.pred_linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.num_tasks)
        )

    def forward(self, logits, atom_feature, atom_mask, batch):
        B, N, D = logits.size()
        # device = atom_feat.device
        #
        # 将掩码展开为一维：只保留真实原子对应的行
        flat_mask = atom_mask.view(-1)  # [B*N]
        flat_atom_feat = logits.reshape(-1, D)  # [B*N, D]
        masked_atom_feat = flat_atom_feat[flat_mask]  # [M, D]，M为真实原子数量

        # GNN 编码器
        x = self.gnn([masked_atom_feat, atom_feature, batch.edge_index, batch.edge_attr])  # [M, D]

        # 图池化
        x = self.pool(x, batch.batch)  # [B, D]

        # 残差块 + 预测头
        x = self.res_blocks(x)
        x = self.pred_linear(x)
        return x


# 原子属性预测
class SMILES_Predictor(torch.nn.Module):
    def __init__(self, args, num_tasks, graph_pooling="attention"):
        super(SMILES_Predictor, self).__init__()

        self.graph_pooling = graph_pooling

        # 池化
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(args.embed_dim, 2 * args.embed_dim),
                                            torch.nn.LayerNorm(2 * args.embed_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * args.embed_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(args.embed_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.n_layers = args.predictor_layer_num

        # 构建 n 层 MLP
        layers = [nn.Linear(args.embed_dim, 2 * args.embed_dim)]
        for _ in range(self.n_layers):
            layers.append(nn.Linear(2 * args.embed_dim, 2 * args.embed_dim))
            layers.append(nn.LayerNorm(2 * args.embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=args.dropout))

        self.mlp = nn.Sequential(*layers)

        self.pred_linear = torch.nn.Linear(2 * args.embed_dim, num_tasks)
        # Dropout for interaction features
        self.Dropout = nn.Dropout(p=args.dropout)

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(args.embed_dim, 2 * args.embed_dim),
        #     torch.nn.LayerNorm(2 * args.embed_dim),
        #     torch.nn.ReLU(),
        #     nn.Dropout(p=args.dropout),  # 添加 Dropout
        #     torch.nn.Linear(2 * args.embed_dim, num_tasks))

    def forward(self, x):  # x = [batch_size, max_node_num, dim]
        output_embed = self.Dropout(x)  # Dropout after concatenation

        # 假设输入 x 的形状为 [batch_size, num_node, dim]
        batch_size, num_node, dim = output_embed.shape  # 获取维度信息
        # 转换 x 的形状为 [batch_size * num_node, dim]
        output_embed_pooled_input = output_embed.reshape(-1, dim)  # [batch_size * num_node, dim]
        # 生成 batch_index，用于标记每个节点属于哪个 batch
        batch_index = torch.arange(batch_size).repeat_interleave(num_node).to(
            output_embed_pooled_input.device)  # [batch_size * num_node]
        output_embed = self.pool(output_embed_pooled_input, batch_index)

        output_embed = self.mlp(output_embed)
        output_embed = self.pred_linear(output_embed)

        return output_embed


# 原子类别预测
class Atom_Predictor(torch.nn.Module):
    def __init__(self, args, num_tasks):
        super(Atom_Predictor, self).__init__()
        self.atom_predictor = torch.nn.Linear(args.embed_dim, num_tasks)  # 用于预测atom的分类
        self.dropout_interaction = nn.Dropout(p=args.dropout)

    def forward(self, x, atom_mask):  # x = [batch_size, max_node_num, dim]
        # 提取相应的 x 切片并根据 atom_mask 进行过滤
        # 使用索引选取目标原子
        batch_indices = torch.arange(x.size(0))  # 批次索引，形状为 [batch]
        x_selected = x[batch_indices, atom_mask, :]  # 形状为 [batch, 1, dim]
        x_selected = self.dropout_interaction(x_selected)
        # 将列表转换为一个新的张量，形状为 [batch_size * node_num, dim]
        # 对节点对嵌入进行拼接，并预测边的存在
        atom_logits = self.atom_predictor(x_selected)

        return atom_logits


# 边预测
class Edge_Predictor(torch.nn.Module):
    def __init__(self, args):
        super(Edge_Predictor, self).__init__()
        self.edge_predictor = torch.nn.Linear(args.embed_dim, 1)  # 用于预测边的存在与否
        self.dropout_interaction = nn.Dropout(p=args.dropout)

    def forward(self, x, atom_mask, edge_index, neg_edge_index):  # x = [batch_size, max_node_num, dim]

        # 正负样本合并
        edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)

        # 遍历每个图，提取相应的 x 切片
        # 创建一个空列表以存储每个图的 x
        x_list = []
        # 遍历每个图，提取相应的 x 切片并根据 atom_mask 进行过滤
        for i in range(x.size(0)):  # 遍历 batch_size
            # 使用 atom_mask 筛选出当前图的原子节点特征
            masked_x = x[i][atom_mask[i]]  # 只获取 atom_mask 为 True 的节点特征
            x_list.append(masked_x)  # 添加到列表中

        # 将列表转换为一个新的张量，形状为 [batch_size, node_num, dim]
        x_new = torch.cat(x_list, dim=0)
        # 获取边的两个节点的嵌入
        # src_node_emb = x_new[edge_index[0]]
        # tgt_node_emb = x_new[edge_index[1]]

        # edge_emb = torch.cat([src_node_emb, tgt_node_emb], dim=-1)
        edge_emb = x_new[edge_index[0]] * x_new[edge_index[1]]
        edge_emb = self.dropout_interaction(edge_emb)
        # 对节点对嵌入进行拼接，并预测边的存在
        edge_logits = self.edge_predictor(edge_emb)

        return edge_logits


# 原子类别预测
class TDDistance_Predictor(torch.nn.Module):
    def __init__(self, args):
        super(TDDistance_Predictor, self).__init__()
        self.dist_predictor = torch.nn.Linear(args.embed_dim, 1)  # 用于预测 距离
        self.dropout_interaction = nn.Dropout(p=args.dropout)

    def forward(self, x, atom_mask, edge_index):  # x = [batch_size, max_node_num, dim]
        # 遍历每个图，提取相应的 x 切片
        # 创建一个空列表以存储每个图的 x
        x_list = []
        # 遍历每个图，提取相应的 x 切片并根据 atom_mask 进行过滤
        for i in range(x.size(0)):  # 遍历 batch_size
            # 使用 atom_mask 筛选出当前图的原子节点特征
            masked_x = x[i][atom_mask[i]]  # 只获取 atom_mask 为 True 的节点特征
            x_list.append(masked_x)  # 添加到列表中

        # 将列表转换为一个新的张量，形状为 [batch_size, node_num, dim]
        x_new = torch.cat(x_list, dim=0)

        edge_emb = x_new[edge_index[0]] * x_new[edge_index[1]]
        edge_emb = self.dropout_interaction(edge_emb)
        # 直接将特征映射到 距离邻接矩阵
        dist_matrix = self.dist_predictor(edge_emb)  # positions = [n_atoms, n_atoms]

        return dist_matrix


# ddi预测
class DDI_Predictor(torch.nn.Module):
    def __init__(self, args, num_tasks, graph_pooling="attention"):
        super(DDI_Predictor, self).__init__()

        self.graph_pooling = graph_pooling

        # 池化
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(args.embed_dim * 2, 2 * args.embed_dim),
                                            torch.nn.LayerNorm(2 * args.embed_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * args.embed_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(args.embed_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.pred_linear = torch.nn.Linear(args.embed_dim * 2, num_tasks)
        # Dropout for interaction features
        self.dropout_interaction = nn.Dropout(p=args.dropout)

        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.embed_dim * 2, 4 * args.embed_dim),
            torch.nn.LayerNorm(4 * args.embed_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=args.dropout),  # 添加 Dropout
            torch.nn.Linear(4 * args.embed_dim, num_tasks))

    def forward(self, x, batch_size):  # x = [batch_size * 2, max_node_num, dim]

        # x_flattened = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # batch_index = torch.repeat_interleave(torch.arange(x.shape[0]), x.shape[1])
        # x = self.pool(x)
        # x = x.mean(dim=1)
        drug1 = x[:batch_size]
        drug2 = x[batch_size:]

        # output_embed = (drug1 * drug2)
        output_embed = torch.cat([drug1, drug2], dim=2)
        output_embed = self.dropout_interaction(output_embed)  # Dropout after concatenation

        # 假设输入 x 的形状为 [batch_size, num_node, dim]
        batch_size, num_node, dim = output_embed.shape  # 获取维度信息
        # 转换 x 的形状为 [batch_size * num_node, dim]
        output_embed_pooled_input = output_embed.reshape(-1, dim)  # [batch_size * num_node, dim]
        # 生成 batch_index，用于标记每个节点属于哪个 batch
        batch_index = torch.arange(batch_size).repeat_interleave(num_node).to(
            output_embed_pooled_input.device)  # [batch_size * num_node]
        output_embed = self.pool(output_embed_pooled_input, batch_index)

        output_embed = self.interaction_mlp(output_embed)

        # output_embed = torch.cat([drug1, drug2], dim=1)
        # output = self.pred_linear(output_embed)
        return output_embed
