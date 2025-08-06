import os
from typing import Set

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, rdchem, AllChem
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from Prompt_model.token_SMILES import smiles_toks

# 用户自定义字符集（支持的原子字符）
smiles_vocab = {
    # 多字符元素
    'Cl': 'L', 'Br': 'R', 'Si': 'T', 'Na': 'Z', 'Ca': 'K', 'Se': 'e', 'Li': 'i',
    'Mg': 'g', 'Al': 'A', 'Fe': 'f', 'Zn': 'z', 'Cu': 'U', 'Mn': 'm',
    'Hg': 'h', 'Pb': 'p', 'Sn': 'n', 'As': 'a', 'Co': 'c', 'Cr': 'r',
    'Ni': 'N', 'Ti': 't', 'Ba': 'b', 'Sr': 's', 'Pd': 'd', 'Pt': 'P',
    'Sb': 'S', 'Bi': 'B', 'Ag': 'G', 'Au': 'u', 'Cd': 'C', 'Ge': 'j',
    'Te': 'E', 'Zr': 'x', 'Nb': 'y', 'Mo': 'M', 'Ru': 'q', 'Rh': 'Q',
    'C': 'C', 'c': 'c', 'N': 'N', 'n': 'n', 'O': 'O', 'o': 'o', 'F': 'F', 'S': 'S', 's': 's', 'P': 'P',
    'H': 'H', 'B': 'B', 'I': 'I', 'Y': 'Y', 'D': 'D', 'G': 'G', 'L': 'L', 'M': 'M', 'V': 'V',
    # 可根据需要扩展
}


def convert_smiles_to_single_char(smiles: str) -> str:
    """
    将原始 SMILES 表达式中的多字符原子（如 Cl, Br）替换为单字符映射（如 L, R）
    其他字符保持不变。
    """
    # 先按长度优先替换（防止 "C" 被提前替换掉）
    sorted_vocab = sorted(smiles_vocab.items(), key=lambda x: -len(x[0]))

    for key, val in sorted_vocab:
        smiles = smiles.replace(key, val)

    return smiles


def atomic_nums_to_smiles_tokens(x: torch.Tensor, batch: torch.Tensor) -> list:
    """
    将 x 中的原子编号还原为单字符的 SMILES-like 字符串（按 batch 分组）
    - 避免多字符原子符号，使用用户定义的字符代替（如 Cl → L, Br → R）
    """
    from rdkit.Chem import GetPeriodicTable
    ptable = GetPeriodicTable()

    possible_atomic_nums = list(range(1, 119)) + ['misc']
    atomic_nums = x[:, 0].tolist()

    def get_custom_symbol(index):
        val = possible_atomic_nums[index]
        if isinstance(val, int):
            sym = ptable.GetElementSymbol(val)
            if smiles_vocab.get(sym, 'X') == 'X':
                print(sym)

            return smiles_vocab.get(sym, 'X')  # 不识别则用 'X' 占位
        else:
            return 'X'

    # 转换为原子字符列表
    atomic_symbols = [get_custom_symbol(idx) for idx in atomic_nums]

    # 聚合每个分子的原子符号
    mol_count = batch.max().item() + 1
    smiles_tokens = ['' for _ in range(mol_count)]
    for i, mol_idx in enumerate(batch.tolist()):
        smiles_tokens[mol_idx] += atomic_symbols[i]

    return smiles_tokens





def get_max_position_num(smiles_list):
    return max(len(smiles) for smiles in smiles_list)


def extract_unique_chars(smiles_list: list) -> Set[str]:
    """
    从一组SMILES字符串中提取所有唯一字符。

    Args:
        smiles_list (list): 包含SMILES字符串的列表

    Returns:
        Set[str]: SMILES字符串中出现的所有唯一字符集合
    """
    unique_chars = set()
    for smiles in smiles_list:
        unique_chars.update(smiles)
    return unique_chars


def list_to(datalist, device):
    for i in range(len(datalist)):
        datalist[i] = datalist[i].to(device)
    return datalist


def smiles_to_indices(smiles, smiles_vocab, max_len):
    """
    将 SMILES 字符串转化为字符索引序列
    """
    char_to_index = {char: idx for idx, char in enumerate(smiles_vocab)}
    indices = [char_to_index[char] for char in smiles if char in char_to_index]

    # Padding 或截断到 max_len 长度
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))  # 使用 0 进行 Padding
    else:
        indices = indices[:max_len]  # 截断到 max_len

    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # 返回索引序列


def build_sparse_distance_matrix(dist_3d, num_atoms_in_batch):
    """
    从 dist_3d 构建稀疏的距离矩阵。

    Args:
        dist_3d: 形状为 [num_entries, 3] 的张量，每个元素为 [i, j, distance]
        num_atoms_in_batch: 批次中原子的总数

    Returns:
        distance_matrix: 稀疏的距离矩阵，形状为 [num_atoms_in_batch, num_atoms_in_batch]
    """
    indices = dist_3d[:, :2].long().t()  # [2, num_entries]
    values = dist_3d[:, 2]  # [num_entries]

    distance_matrix = torch.sparse_coo_tensor(indices, values, size=(num_atoms_in_batch, num_atoms_in_batch))

    return distance_matrix


def get_atom_char_indices(smiles):
    """
    获取 SMILES 字符串中原子字符及其索引。

    Args:
        smiles (str): 输入的 SMILES 字符串。
        smiles_toks (dict): SMILES 的字符集定义，包括原子和其他符号。

    Returns:
        atom_indices (list): 包含原子字符及其索引的列表，格式为 [(index, atom), ...]。
    """
    # 定义原子字符集
    atom_chars = set(smiles_toks['toks'][:smiles_toks['atom_num']])  # 前 `atom_num` 个是原子

    # 遍历 SMILES，查找原子字符及其索引
    atom_indices = []
    i = 0
    while i < len(smiles):
        # 如果当前字符是原子字符
        if smiles[i] in atom_chars:  # 检测单字符原子
            atom_indices.append((i, smiles[i]))
        i += 1  # 跳过非原子字符

    return atom_indices


def smiles_to_3D_coordinates(mol):
    """
    使用 RDKit 生成 SMILES 的 3D 坐标，如果失败则返回默认坐标。

    Args:
        mol (mol): 输入的 mol。
        default_coords (numpy.ndarray or None): 如果生成失败，返回的默认 3D 坐标。
                                                如果为 None，生成零矩阵。
        random_seed (int): 随机种子，用于 3D 构象生成。
        max_attempts (int): 最大尝试次数，用于 3D 构象生成。

    Returns:
        numpy.ndarray: 生成的 3D 坐标矩阵，形状为 [num_atoms, 3]。
    """
    # 使用 RDKit 解析 SMILES 字符串并添加氢原子

    # 获取原子数量（在添加氢原子之前）
    natom = mol.GetNumAtoms()
    try:
        mol = Chem.AddHs(mol)

        # 设置 ETKDG 参数
        params = AllChem.ETKDGv3()

        # 尝试生成 3D 构象
        success = AllChem.EmbedMolecule(mol, params)

        if success == 0:  # 构象生成成功
            # 优化 3D 坐标
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()

            # 提取非氢原子的坐标
            non_h_coords = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() > 1:  # 非氢原子
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    non_h_coords.append([pos.x, pos.y, pos.z])
            coords = torch.tensor(non_h_coords, dtype=torch.float)  # [n_atoms, 3]
            return coords, 1
        else:
            raise ValueError("Failed to generate 3D coordinates.")

    except Exception as e:
        # print(f"Error generating 3D coordinates for SMILES '{smiles}': {e}")

        # 返回形状匹配的默认值
        coords = torch.zeros([natom, 3], dtype=torch.float)
        return coords, 0


# One-hot编码函数
def one_hot_encode_smiles(smiles, smiles_vocab, max_len):
    # 创建字符到索引的映射
    char_to_index = {char: idx for idx, char in enumerate(smiles_vocab)}
    # 保证每个 SMILES 编码后的长度一致 (max_len)
    one_hot_matrix = np.zeros((max_len, len(smiles_vocab)))

    # 遍历 SMILES 并编码（考虑 padding）
    for i, char in enumerate(smiles):
        if i >= max_len:  # 截断过长的 SMILES
            break
        if char in char_to_index:
            one_hot_matrix[i, char_to_index[char]] = 1
    return one_hot_matrix


def load_sdf(sdf_path, drug_id):
    """ Load SDF file and return RDKit molecule object. """
    if os.path.exists(sdf_path):
        mol_supplier = Chem.SDMolSupplier(sdf_path)
        mol = next(mol_supplier)  # Assume there's only one molecule per SDF file
        if mol is not None:
            return mol
    return None


def sdf_to_smiles(sdf_file):
    """
    将 SDF 文件中的化合物结构转换为 SMILES。

    Args:
        sdf_file (str): SDF 文件的路径。

    Returns:
        smiles_list (list): 包含 SMILES 字符串的列表。
    """
    smiles_list = []

    # 读取 SDF 文件中的化合物
    suppl = SDMolSupplier(sdf_file)

    # 遍历每个分子并转换为 SMILES
    if len(suppl) > 1:
        print(sdf_file)
    for mol in suppl:
        if mol is not None:  # 确保分子加载成功
            smiles = Chem.MolToSmiles(mol)
            return smiles, mol


# 手动定义常见元素的电负性（保留常见的元素，可以根据需要扩展）
electronegativity_table = {
    1: 2.20,  # 氢
    6: 2.55,  # 碳
    7: 3.04,  # 氮
    8: 3.44,  # 氧
    9: 3.98,  # 氟
    16: 2.58,  # 硫
    17: 3.16,  # 氯
    # 可以继续添加其他元素
}


def get_electronegativity(atomic_num):
    """
    根据原子序数获取电负性，如果没有定义，返回 0.
    """
    return electronegativity_table.get(atomic_num, 0.0)


def atom_features(atom):
    """
    提取原子的 11 个特征，基于 QM9 数据集中的原子特征定义。

    Args:
        atom: RDKit Mol 对象中的一个原子。

    Returns:
        torch.Tensor: 包含原子特征的张量。
    """
    return torch.tensor([
        atom.GetAtomicNum(),  # 原子的原子序数
        atom.GetMass() * 0.01,  # 原子质量标准化 (乘以 0.01)
        atom.GetDegree(),  # 原子连接的键的数量
        atom.GetTotalValence(),  # 原子的总化合价
        atom.GetHybridization().real,  # 原子的杂化状态
        atom.GetIsAromatic(),  # 原子是否是芳香族
        atom.GetFormalCharge(),  # 原子的形式电荷
        atom.GetTotalNumHs(),  # 原子上连接的氢原子数量
        Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()),  # 原子的范德华半径
        atom.GetAtomicNum(),  # 原子的原子序数
        get_electronegativity(atom.GetAtomicNum())  # 原子的电负性
    ], dtype=torch.float)


def bond_features(bond):
    """
    提取键特征。你可以根据需要自定义。

    Args:
        bond: RDKit Mol 对象中的一个键。

    Returns:
        torch.Tensor: 包含键特征的张量。
    """
    if bond is None:
        bond_type = 0
    else:
        bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == rdchem.BondType.SINGLE,  # 是否为单键
        bond_type == rdchem.BondType.DOUBLE,  # 是否为双键
        bond_type == rdchem.BondType.TRIPLE,  # 是否为三键
        bond_type == rdchem.BondType.AROMATIC  # 是否为芳香键
    ], dtype=torch.float)


def mol_to_pyg(mol):
    """
    将 RDKit Mol 对象转换为 PyTorch Geometric 的 Data 对象。

    Args:
        mol: RDKit Mol 对象。

    Returns:
        Data: 包含分子的 PyG 图结构和节点、边特征。
    """
    # 1. 提取原子特征，创建节点特征矩阵 x
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))
    x = torch.stack(atom_features_list)  # 节点特征矩阵 (num_atoms, num_features)

    # 2. 提取边信息，创建边索引和边特征矩阵 edge_index, edge_attr
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()  # 起始原子索引
        j = bond.GetEndAtomIdx()  # 终止原子索引
        edge_index.append([i, j])  # 添加边
        edge_index.append([j, i])  # PyG 的图是无向图，所以双向都需要添加
        edge_attr.append(bond_features(bond))  # 添加键特征
        edge_attr.append(bond_features(bond))  # 双向边特征

    if len(edge_attr) == 0:
        edge_index.append([0, 0])
        edge_attr.append(bond_features(None))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 边索引 (2, num_edges)
    edge_attr = torch.stack(edge_attr)  # 边特征矩阵 (num_edges, num_edge_features)

    # 3. 创建 PyG 的 Data 对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def smiles_to_edgeindex(mol):
    # 获取原子的数量
    num_atoms = mol.GetNumAtoms()

    # 初始化边索引列表
    edge_index = []

    # 遍历分子中的所有键，提取边的信息
    for bond in mol.GetBonds():
        # 获取键的起始原子和结束原子
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # 添加边（双向）到 edge_index
        edge_index.append([start_idx, end_idx])
        edge_index.append([end_idx, start_idx])  # 如果是无向图，添加反向边

    # 转换为 PyTorch 张量
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 找到孤立节点（即没有出现在 edge_index_tensor 中的节点）
    connected_nodes = torch.unique(edge_index_tensor)
    isolated_nodes = torch.tensor([i for i in range(num_atoms) if i not in connected_nodes], dtype=torch.long)

    # 对孤立节点添加自环边
    if isolated_nodes.numel() > 0:
        self_loops = torch.stack([isolated_nodes, isolated_nodes])
        edge_index_tensor = torch.cat([edge_index_tensor, self_loops], dim=1)

    return edge_index_tensor


def add_virtual_edges(data):
    if data.edge_index.size(1) == 0:  # 检查是否没有任何边
        # 添加自连接的虚拟边
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return data
