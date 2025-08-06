import os
import random

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm

from Utils.featureUtil import smiles_to_edgeindex, smiles_to_3D_coordinates, get_atom_char_indices


class CombinedQM9Dataset(Dataset):
    def __init__(self, smiles_dataset, graph_dataset):
        super().__init__()
        if len(smiles_dataset) != len(graph_dataset):
            print("SMILES and Graph datasets must have the same length.")
        # assert len(smiles_dataset) == len(graph_dataset), "SMILES and Graph datasets must have the same length."
        self.smiles_dataset = smiles_dataset
        self.graph_dataset = graph_dataset
        self.length = min(len(smiles_dataset), len(graph_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        smiles_data = self.smiles_dataset[idx]  # 从 SMILES 数据集中获取数据
        graph_data = self.graph_dataset[idx]  # 从 Graph 数据集中获取数据
        return smiles_data, graph_data  # 返回两个数据集的对应项


# 11698 失败 133885总共
class QM9SMILESDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, csv_file='qm9.csv', max_len=29):
        """
        InMemoryDataset for loading SMILES and molecular properties from a CSV file.
        Args:
            root (str): Root directory where the dataset should be stored.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in a sample and returns a transformed version before saving to disk.
            csv_file (str): Path to the CSV file.
            max_len (int): Maximum length of SMILES sequence. 经过计算QM9是 29
        """
        self.csv_file = csv_file
        self.max_len = max_len
        self.num_tasks = 19
        self.task_type = "regression"

        super(QM9SMILESDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # 加载预处理后的数据

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # 读取 CSV 文件
        df = pd.read_csv(os.path.join(self.raw_dir, self.csv_file))

        # SMILES 和分子属性
        smiles_list = df['smiles'].tolist()
        targets = df[['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                      'zpve', 'u0', 'u298', 'h298', 'g298', 'cv',
                      'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']].values

        data_list = []

        for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Processing SMILES"):
            # 使用 RDKit 解析 SMILES 字符串
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            y = torch.tensor(targets[idx], dtype=torch.float).unsqueeze(0)  # 确保 y 维度为 (1, 19)
            num_atoms = mol.GetNumAtoms()

            # 原子字符预测任务
            # 随机选择一个原子索引
            atom_char_indices = get_atom_char_indices(smiles)
            atom_mask_index = random.choice(atom_char_indices)  # 随机选择一个原子
            masked_smiles = ''.join([c if i != atom_mask_index[0] else '*' for i, c in enumerate(smiles)])

            # 边预测任务
            edge_index = smiles_to_edgeindex(mol)

            # 空间距离预测任务
            pos, success = smiles_to_3D_coordinates(mol)
            dist_3d = []
            dist_3d_v = []
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    distance = torch.norm(pos[i] - pos[j], p=2)
                    dist_3d.append([i, j])
                    dist_3d_v.append([distance])
            dist_3d = torch.tensor(dist_3d, dtype=torch.long).t().contiguous()
            dist_3d_v = torch.tensor(dist_3d_v, dtype=torch.float)
            if dist_3d.numel() == 0:  # 检查 dist_3d 是否为空
                dist_3d_mask = torch.tensor([], dtype=torch.long)  # 创建空 mask
            else:
                dist_3d_mask = torch.full((dist_3d.size(1),), success, dtype=torch.long)

            # 通过 torch_geometric.data.Data 来存储每个样本
            data = Data(smiles=smiles, y=y,
                        # 原子字符预测任务
                        atom_mask_idx=atom_mask_index[0], masked_smiles=masked_smiles, masked_char=atom_mask_index[1],
                        # 边预测任务
                        edge_index=edge_index, num_nodes=num_atoms,
                        # 空间距离预测任务
                        mol_3d_edge_index=dist_3d, mol_3d_dist=dist_3d_v, mol_dist_mask=dist_3d_mask)

            # 生成负样本 改到动态生成 并使用随机种子固定
            data = train_test_split_edges(data)
            if hasattr(data, 'train_neg_adj_mask'):
                del data.train_neg_adj_mask
            # data.num_nodes = num_atoms
            data_list.append(data)

        # 保存预处理后的数据
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
