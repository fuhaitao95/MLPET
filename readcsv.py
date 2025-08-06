# import pandas as pd
# import numpy as np
# import ast
#
# # 1. 读取 CSV 文件
# df = pd.read_csv("filtered_inferred_ddi.csv")
#
# # 2. 逐行解析 pred_prob 字段为 list 并收集到一个大列表
# all_pred_probs = []
# for x in df['pred_prob']:
#     prob_list = ast.literal_eval(x)        # 将字符串 "[0.1, 0.3, ...]" 变成 list
#     all_pred_probs.append(prob_list)       # 加入大列表
#
# # ✅ 可选：转为 numpy 数组
# all_pred_probs_array = np.array(all_pred_probs)
#
# # ✅ 输出确认
# print("共解析出 %d 个样本，每个样本有 %d 类预测概率" % (len(all_pred_probs_array), len(all_pred_probs_array[0])))
# print("示例第一个样本的 pred_prob：", all_pred_probs_array[0])

# import os
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import SDMolSupplier
#
# # ========== 第一部分：从 drug_properties.csv 加载 Drug ID - SMILES ==========
#
# def load_csv_smiles_mapping(csv_path):
#     df = pd.read_csv(csv_path, usecols=['Drug ID', 'SMILES'])
#     mapping = {}
#     for _, row in df.iterrows():
#         drug_id = row['Drug ID']
#         smiles = row['SMILES']
#         if pd.notna(smiles):
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is not None:
#                 cano_smiles = Chem.MolToSmiles(mol, canonical=True)
#                 mapping[drug_id] = cano_smiles
#     return mapping
#
#
# # ========== 第二部分：从 sdf 文件夹中提取 Drug ID - SMILES ==========
#
# def load_sdf_smiles_mapping(sdf_dir):
#     mapping = {}
#     for filename in os.listdir(sdf_dir):
#         if filename.endswith(".sdf"):
#             drug_id = filename.replace(".sdf", "")
#             sdf_path = os.path.join(sdf_dir, filename)
#             suppl = SDMolSupplier(sdf_path)
#             for mol in suppl:
#                 if mol is not None:
#                     smiles = Chem.MolToSmiles(mol)
#                     cano_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
#                     mapping[drug_id] = cano_smiles
#                     break  # 只取第一个分子
#     return mapping
#
#
# # ========== 第三部分：合并两个映射表 ==========
#
# def merge_mappings(map1, map2):
#     merged = dict(map1)  # 先复制一份
#     for k, v in map2.items():
#         if k not in merged:
#             merged[k] = v  # 如果不存在才添加
#     return merged
#
#
# # ========== 执行流程 ==========
#
# drug_properties_path = 'dataset/DrugBank_multimodal/raw/drug_properties.csv'
# sdf_dir = 'dataset/DeepDDI/raw/DrugBank5.0_Approved_drugs'
#
# csv_mapping = load_csv_smiles_mapping(drug_properties_path)
# sdf_mapping = load_sdf_smiles_mapping(sdf_dir)
# merged_mapping = merge_mappings(csv_mapping, sdf_mapping)
#
# # 可选：保存为 CSV
# df_output = pd.DataFrame(list(merged_mapping.items()), columns=["Drug ID", "Canonical SMILES"])
# df_output.to_csv("drug_id_smiles_mapping.csv", index=False)
#
# print(f"总共提取了 {len(merged_mapping)} 个 Drug ID 的 SMILES 映射。")

import pandas as pd
from rdkit import Chem

# # === Step 1: 读取 SMILES → DBID 映射表 ===
# mapping_df = pd.read_csv("drug_id_smiles_mapping.csv")
#
# # 构建 SMILES → DBID 字典（canonical 格式保证一致性）
# smiles_to_dbid = {}
# for _, row in mapping_df.iterrows():
#     dbid = row['Drug ID']
#     smiles = row['Canonical SMILES']
#     if pd.notna(smiles):
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             canonical = Chem.MolToSmiles(mol, canonical=True)
#             smiles_to_dbid[canonical] = dbid
#
# # === Step 2: 读取 DDI 预测结果 ===
# ddi_df = pd.read_csv("filtered_inferred_ddi.csv")
#
# # === Step 3: 替换函数 ===
# def get_dbid_from_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None
#         canonical = Chem.MolToSmiles(mol, canonical=True)
#         return smiles_to_dbid.get(canonical)
#     except:
#         return None
#
# # 替换并记录
# ddi_df['drug1_dbid'] = ddi_df['drug1'].apply(get_dbid_from_smiles)
# ddi_df['drug2_dbid'] = ddi_df['drug2'].apply(get_dbid_from_smiles)
#
# # === Step 4: 过滤无匹配项 ===
# filtered_df = ddi_df.dropna(subset=['drug1_dbid', 'drug2_dbid']).copy()
#
# # 保留并重命名列
# filtered_df = filtered_df[['drug1_dbid', 'drug2_dbid', 'pred_prob', 'pred_label']]
# filtered_df = filtered_df.rename(columns={'drug1_dbid': 'drug1', 'drug2_dbid': 'drug2'})
#
# # === Step 5: 保存结果 ===
# filtered_df.to_csv("filtered_inferred_ddi_dbid.csv", index=False)
# print(f"✅ 处理完成，保存到 filtered_inferred_ddi_dbid.csv，保留条数: {len(filtered_df)}")

import pandas as pd
import ast  # 用于安全解析字符串列表

# 读取CSV
df = pd.read_csv("filtered_inferred_ddi_dbid.csv")

# 提取 pred_prob 中 pred_label 对应的值
def extract_label_prob(row):
    try:
        prob_list = ast.literal_eval(row['pred_prob'])  # 将字符串列表转为Python list
        label_idx = int(row['pred_label'])
        return prob_list[label_idx]
    except Exception as e:
        print(f"Error parsing row {row.name}: {e}")
        return None

# 应用函数并替换 pred_prob 列
df['pred_prob'] = df.apply(extract_label_prob, axis=1)

# # 去除异常行（如None）
# df = df.dropna(subset=['pred_prob'])

# 保存处理后的新CSV（可选）
df.to_csv("filtered_inferred_ddi_dbid_predictions.csv", index=False)

data = [9.840202983468771e-05, 1.8705783077166416e-05, 1.108069000110845e-06, 6.458897701122623e-07, 1.1853755824131618e-12,
        1.7974872202085974e-20, 9.823067016856291e-23, 3.4431762996334214e-10, 4.238140860479689e-08, 7.2420633934200396e-09,
        5.225195903263641e-16, 0.9987161159515381, 9.269466180788646e-18, 2.529585048371974e-12, 1.2371123548314544e-13,
        7.080603177727363e-17, 1.6127373742014228e-11, 5.334489321318124e-19, 1.3509443306247704e-06, 1.8271351675689453e-16,
        2.6187773195622543e-13, 5.876387294519517e-14, 0.001163687091320753, 8.594420296916596e-21, 2.8909599772747774e-16,
        1.2112157631027909e-11, 3.618630937558127e-22, 2.1375838215294818e-20, 8.659316008704362e-21, 6.581717315406831e-18,
        7.60887528114193e-14, 5.585892256939834e-19, 2.812089166135683e-17, 3.395966632021067e-15, 1.5092703475882985e-10,
        6.951344864715393e-15, 2.945406513970455e-15, 1.893816958496115e-19, 8.904213572705126e-22, 1.353612580383795e-23,
        1.5290265601980792e-20, 1.425902377161157e-17, 4.531581798580776e-15, 4.3190195343767446e-12, 4.7682846605458454e-21,
        1.1768361647313632e-18, 1.1708335401985227e-22, 1.4628721428965754e-24, 2.6573567504117563e-19, 1.769789497449914e-19,
        2.570827712978057e-19, 4.599353748414817e-19, 1.12665250069133e-25, 2.014148314638365e-22, 3.4522550885962535e-19,
        1.0816324325150319e-18, 1.494060769233771e-15, 3.528066013769718e-24, 3.467751496219508e-26, 6.77141561006997e-28,
        9.626795015758193e-22, 1.3673344646391964e-30, 1.5441042564469973e-25, 6.105812988694577e-26, 2.7119317218648873e-25]

print(len(data))  # 输出: 65