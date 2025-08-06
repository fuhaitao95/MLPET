import torch
from rdkit.Chem import BRICS
from torch_geometric.data import Data


# 将一个二维数组转换为一维列表并包含batch列表
def flatten_2d_list_with_batch(input_list, device):
    flattened_list = []
    batch_tensor = torch.tensor([], device=device, dtype=torch.long)
    for i, sublist in enumerate(input_list):
        for item in sublist:
            flattened_list.append(item)
            batch_tensor = torch.cat((batch_tensor, torch.tensor([i], device=device, dtype=torch.long)))  # 将批次信息添加到集合中
    return flattened_list, batch_tensor


def flatten_2d_list_with_batch_motifeat(input_list, masked_atom_feat, cliques, batch, device):
    """
    将 masked_atom_feat 拆分为每个图，然后根据 cliques 获取每个基序的原子特征，
    并将其挂载到对应 item（Data 对象）上，字段为 item.motif_atom_feat。

    返回：
        - flattened_list: 所有 item（已挂上 motif 特征）打平成一维
        - batch_tensor: 与 flattened_list 对应的图索引 tensor（[num_items]）
    """
    from torch_geometric.data import Batch

    batch_indices = batch.batch.tolist()  # [N_atoms]，表示每个原子属于哪个图
    batch_size = batch_indices[-1] + 1

    # step1: 将 masked_atom_feat 拆成每图的特征
    split_feats = [[] for _ in range(batch_size)]
    for idx, graph_idx in enumerate(batch_indices):
        split_feats[graph_idx].append(masked_atom_feat[idx])

    split_feats = [
        torch.stack(f, dim=0) if len(f) > 0 else torch.empty(0, masked_atom_feat.size(-1))
        for f in split_feats
    ]

    # step2: 遍历每图，对每个 item 赋予其基序特征
    flattened_list = []
    batch_tensor = torch.tensor([], device=device, dtype=torch.long)

    for i, sublist in enumerate(input_list):
        graph_feats = split_feats[i]  # Tensor[num_atoms, D]
        graph_cliques = cliques[i]  # List[List[int]]

        motif_feats = []
        for motif_idx in graph_cliques:
            motif_tensor = graph_feats[motif_idx]
            motif_feats.append(motif_tensor)

        # 逐个 item 赋值
        for j, item in enumerate(sublist):
            if j >= len(motif_feats):
                raise IndexError(f"Index {j} out of motif_feats range for graph {i}")
            item.motif_atom_feat = motif_feats[j]
            flattened_list.append(item)
            batch_tensor = torch.cat(
                (batch_tensor, torch.tensor([i], device=device, dtype=torch.long))
            )

    return flattened_list, batch_tensor


def motif_only_construct(cliques, x, edge_index, edge_attr, y, device):
    g_motifs = []
    for clique in cliques:
        # 构建基序Data对象，方便内部Batch使用
        g_motif = Data()
        g_motif.num_nodes = len(clique)
        g_motif.y = y
        edge_clique = [[], []]
        attr_clique = []
        for i, a_atom in enumerate(clique):  # 状态不好写的，估计可以优化
            for j, b_atom in enumerate(clique):
                if i == j:
                    continue
                else:
                    for i_edge, row_col in enumerate(zip(edge_index[0], edge_index[1])):
                        row, col = row_col
                        if a_atom == row and b_atom == col:  # 判断成功后往里添加基序索引的边 改为添加i j索引代替原子编号
                            edge_clique[0].append(i)
                            edge_clique[1].append(j)
                            attr_clique.append(edge_attr[i_edge])
                            break

        g_motif.edge_index = torch.tensor(edge_clique).to(device=device, dtype=torch.long)
        if len(attr_clique) > 0:
            g_motif.edge_attr = torch.stack(attr_clique).to(device=device, dtype=torch.long)
        else:
            g_motif.edge_attr = torch.tensor(attr_clique).to(device=device, dtype=torch.long)
        g_motif.x = x[clique]
        g_motifs.append(g_motif)

    return g_motifs


def motifgraph_featconstruct(cliques, x, edge_index, edge_attr, device):  # cliques:[[1,2,3],[4,5,6]]
    # 由于基序中存在重复的节点， 所以图的信息要遍历获取
    # todo 考虑一下极端情况，cliques为null
    edge_motif = [[], []]
    attr_motif = []
    for i_cliques in range(len(cliques)):
        for j_cliques in range(len(cliques)):
            if i_cliques == j_cliques:
                continue
            else:
                # 将其转换为基序索引 注意基序中的节点有可能同时出现在两个基序中 需要修改edge结构
                for a_atom in cliques[i_cliques]:
                    for b_atom in cliques[j_cliques]:
                        for i, row_col in enumerate(zip(edge_index[0], edge_index[1])):
                            row, col = row_col
                            if a_atom == row and b_atom == col:  # 判断成功后往里添加基序索引的边
                                edge_motif[0].append(i_cliques)
                                edge_motif[1].append(j_cliques)
                                attr_motif.append(edge_attr[i])
                                break

    edge_motif = torch.tensor(edge_motif).to(device=device, dtype=torch.long)
    if len(attr_motif) > 0:
        attr_motif = torch.stack(attr_motif).to(device=device, dtype=torch.long)
    else:
        attr_motif = torch.tensor([]).to(device=device, dtype=torch.long)
    # 基序图数据
    g_motif_graph = Data()
    g_motif_graph.num_nodes = len(cliques)
    g_motif_graph.x = torch.tensor([]).to(device=device, dtype=torch.long)
    g_motif_graph.edge_attr = attr_motif
    g_motif_graph.edge_index = edge_motif

    edge_cliques = []
    attr_cliques = []
    num_edge_cliques = []
    g_motifs = []
    for clique in cliques:
        # 构建基序Data对象，方便内部Batch使用
        g_motif = Data()
        g_motif.num_nodes = len(clique)

        edge_clique = [[], []]
        attr_clique = []
        for i, a_atom in enumerate(clique):  # 状态不好写的，估计可以优化
            for j, b_atom in enumerate(clique):
                if i == j:
                    continue
                else:
                    for i_edge, row_col in enumerate(zip(edge_index[0], edge_index[1])):
                        row, col = row_col
                        if a_atom == row and b_atom == col:  # 判断成功后往里添加基序索引的边 改为添加i j索引代替原子编号
                            edge_clique[0].append(i)
                            edge_clique[1].append(j)
                            attr_clique.append(edge_attr[i_edge])
                            break

        g_motif.edge_index = torch.tensor(edge_clique).to(device=device, dtype=torch.long)
        if len(attr_clique) > 0:
            g_motif.edge_attr = torch.stack(attr_clique).to(device=device, dtype=torch.long)
        else:
            g_motif.edge_attr = torch.tensor(attr_clique).to(device=device, dtype=torch.long)
        g_motif.x = x[clique]
        g_motifs.append(g_motif)

        num_edge_cliques.append(len(attr_clique))
        edge_cliques.append(torch.tensor(edge_clique).to(device=device, dtype=torch.long))
        if len(attr_clique) > 0:
            attr_cliques.append(torch.stack(attr_clique))
        else:
            attr_cliques.append(torch.tensor(attr_clique))

    edge_index_cliques_tensor = torch.tensor([[], []]).to(device=device, dtype=torch.long)
    attr_cliques_tensor = torch.tensor([]).to(device=device, dtype=torch.long)
    if len(edge_cliques) > 0:
        edge_index_cliques_tensor = torch.cat(edge_cliques, dim=1).to(device=device, dtype=torch.long)
        attr_cliques_tensor = torch.cat(attr_cliques, dim=0).to(device=device, dtype=torch.long)

    return edge_motif, attr_motif, edge_index_cliques_tensor, attr_cliques_tensor, num_edge_cliques, g_motifs, g_motif_graph


# todo 数据可视化方案
def motif_decomp(mol):  # 传入Data对象方便debug获取信息
    graph_mol = mol
    # 选择使用mol对象来处理数据
    n_atoms = graph_mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []

    for bond in graph_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(graph_mol))  # turbo list [((24,23),('3','16')), ... ] 第一个值为边 第二个值为该边的原子类型

    if len(res) != 0:
        for bond in res:  # 清除基序与基序直接的连接 取其第一个值
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:  # 判断两条边是否相交 如果相交就合并
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]  # 清除合并后的空集合
    cliques = [c for c in cliques if n_atoms > len(c) > 0]
    # todo 预处理的提取算法
    # 处理ring基序
    # num_cli = len(cliques)
    # ssr_mol = Chem.GetSymmSSSR(graph_mol)
    # for i in range(num_cli):
    #     c = cliques[i]
    #     cmol = get_clique_mol(graph_mol, c)
    #     ssr = Chem.GetSymmSSSR(cmol)
    #     if len(ssr) > 1:
    #         for ring in ssr_mol:
    #             if len(set(list(ring)) & set(c)) == len(list(ring)):
    #                 cliques.append(list(ring))
    #                 cliques[i] = list(set(cliques[i]) - set(list(ring)))
    #                 # todo 这里有bug拆分后的集合里无法连接的节点
    #                 print(cliques[i])

    cliques = [c for c in cliques if n_atoms > len(c) > 0]
    if len(cliques) == 0:
        return [[c for c in range(n_atoms)]]
    return cliques


if __name__ == '__main__':
    print()
