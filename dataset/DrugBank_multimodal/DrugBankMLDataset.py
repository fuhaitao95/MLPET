import sqlite3
import os
from typing import Set

import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
from tqdm import tqdm


from Prompt_model.converter import PromptConverter
from Prompt_model.token_SMILES import Alphabet
from Utils.evalUtil import compute_class_weights, compute_class_weights4ddi
from Utils.featureUtil import extract_unique_chars, get_max_position_num


class EventDDIDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EventDDIDataset, self).__init__(root, transform, pre_transform)

        self.task_type = "classification"

        self.num_tasks = 65

        # 加载已处理数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['event.db']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        pass

    def process(self):
        # 连接SQLite数据库
        conn = sqlite3.connect(os.path.join(self.raw_dir, 'event.db'))
        cursor = conn.cursor()

        # 读取drug表
        cursor.execute("SELECT * FROM drug")
        drugs = cursor.fetchall()
        drug_to_index = {drug[1]: idx for idx, drug in enumerate(drugs)}  # 使用id作为药物索引

        # 读取event表
        cursor.execute("SELECT * FROM event")
        events = cursor.fetchall()

        # 读取extraction表
        cursor.execute("SELECT * FROM extraction")
        extractions = cursor.fetchall()

        #
        # # 读取event_number表
        # cursor.execute("SELECT * FROM event_number")
        # event_number = cursor.fetchall()
        # event_number_dict = {row[0]: row[1] for row in event_number}  # interaction -> frequency

        # 读取CSV文件，将数据加载为DataFrame
        csv_data = pd.read_csv(os.path.join(self.raw_dir, 'drug_properties.csv'), usecols=['Drug ID', 'SMILES'])

        data_list = []

        # 创建字典来存储 (mechanism, action) 到唯一标签的映射
        label_map = {}
        label_counter = 0

        # 遍历 extraction 表，将每个 (mechanism, action) 对组合分配一个唯一标签 ID
        extraction_dict = {}
        for ext in extractions:
            index = ext[0]
            mechanism_action = (ext[1], ext[2])  # (mechanism, action)

            # 如果该 (mechanism, action) 对组合还没有标签，则创建一个新标签
            if mechanism_action not in label_map:
                label_map[mechanism_action] = label_counter
                label_counter += 1
            # print("药物作用类型数量："+label_counter)
            # 将 drug_pair 映射到标签 ID
            extraction_dict[index] = label_map[mechanism_action]

        # 遍历每个交互事件
        for event in tqdm(events, desc="Processing events"):
            drug1_id, drug2_id, index = event[1], event[3], event[0]

            if drug1_id not in drug_to_index or drug2_id not in drug_to_index:
                continue  # 跳过没有药物信息的交互
            # 与DeepDDI label对齐
            label = extraction_dict[index]
            drug1_idx = drug_to_index[drug1_id]
            drug2_idx = drug_to_index[drug2_id]

            drug1_data = csv_data[csv_data['Drug ID'] == drug1_id]
            drug2_data = csv_data[csv_data['Drug ID'] == drug2_id]

            if drug1_data.empty or drug2_data.empty:
                print(drug1_id + "&" + drug2_id + " smiles is none")

            drug1_smiles = drug1_data['SMILES'].iloc[0]
            drug2_smiles = drug2_data['SMILES'].iloc[0]

            # 添加边信息 将数据转换为PyTorch Tensor
            edge_index = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long)

            # Create the Data object for this drug pair
            data = Data(
                drug1_smiles=drug1_smiles,  # Drug1 SMILES
                drug2_smiles=drug2_smiles,  # Drug2 SMILES
                edge_index=edge_index,  # Edge between drug1 and drug2
                label=torch.tensor([label], dtype=torch.long)  # Label (DDI interaction type)
            )

            data_list.append(data)

        # 保存预处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Usage:
# dataset = EventDDIDataset(root='../../dataset/DrugBank_multimodal')
# data = dataset[0]
# weigth = compute_class_weights4ddi(dataset, 65)

# Counter(
# {3: 9810, 0: 9496, 2: 5646, 1: 2386, 8: 1312, 33: 1132, 35: 1102, 11: 1086,
# 22: 695, 7: 551, 9: 362, 20: 318, 4: 245, 10: 245, 41: 188, 13: 165, 38: 163,
# 18: 159, 6: 158, 21: 154, 14: 126, 43: 102, 19: 100, 56: 95, 46: 92, 16: 81,
# 36: 77, 37: 75, 30: 70, 12: 67, 25: 64, 23: 62, 15: 59, 17: 58, 26: 57, 28: 55,
# 39: 54, 31: 51, 5: 49, 53: 48, 24: 44, 48: 44, 42: 40, 54: 40, 64: 40, 32: 34,
# 40: 34, 49: 21, 34: 20, 59: 15, 45: 13, 29: 12, 52: 10, 57: 10, 47: 9, 50: 9,
# 62: 9, 27: 7, 58: 7, 63: 6, 44: 5, 51: 5, 55: 5, 60: 5, 61: 5})

# print(data)
# smiles_list = dataset.drug1_smiles + dataset.drug2_smiles
# # 提取SMILES字符集
# unique_smiles_chars = extract_unique_chars(smiles_list)
# print("SMILES字符串中的所有唯一字符：", unique_smiles_chars)
#
# max_position_num = get_max_position_num(smiles_list)
# print("max_position_num:", max_position_num)
