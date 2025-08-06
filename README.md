
# MLPET: A Multi-Level Prompt-Enhanced Transformer for Unified Molecular Property and Drug-Drug Interaction event Prediction

## Abstract

Accurate prediction of molecular properties and drug–drug interaction (DDI) events is crucial for drug discovery and pharmacological safety assessment. 
However, existing approaches typically focus on single-level molecular representation and lack generalizable strategies for multi-task knowledge transfer. 
In this work, we propose a unified Multi-Level Prompt-Enhanced Transformer framework (MLPET), jointly modeling both multi-level molecular features and inter-drug relational semantics. 
To capture multi-level structural knowledge, we design three complementary self-supervised pretraining tasks: 
(1) atom prediction, which predicts randomly masked atom types to learn local chemical context; 
(2) bond prediction, which infers the existence of chemical bonds between atom pairs to capture topological dependencies; 
and (3) distance prediction, which regresses the 3D spatial distance between atoms to encode molecular geometry. 
In the downstream stage, we introduce a lightweight prompt fusion mechanism that integrates task-specific prompts into a unified vector to guide fine-tuning. 
This enables flexible and efficient knowledge transfer to multiple tasks, including molecular property prediction and multi-class DDI event classification. 
Extensive experiments on MoleculeNet, Ryu, and Deng benchmarks demonstrate that MLPET consistently outperforms state-of-the-art baselines, particularly in low-resource and longtail scenarios. 
Our results highlight the potential of promptguided multi-task pretraining as a generalizable paradigm for molecular representation learning. 


## 简介

本项目实现了一个基于 Prompt 的 SMILES 表达式学习框架，支持多任务联合预训练以及在多种下游任务上的迁移评估，包括：

- 分子三维距离回归（3D Distance Prediction）
- 原子掩码预测（Atom Mask Prediction）
- 边连接预测（Edge Link Prediction）
- 药物-药物相互作用分类（DDI Classification）
- 分子性质预测（Molecular Property Prediction）

---

## 项目结构

```text
.
├── main_pretrain.py       # 预训练：三任务联合训练
├── main_DDI.py            # 多分类 DDI 预测任务
├── main_MPP.py            # 分子性质预测任务（OGB数据集）
├── Prompt_model/          # 编码器、Prompt模块、各类预测器
├── dataset/               # DeepDDI / DrugBank / OGB 数据集加载器
├── pretrain_dataset/      # QM9SMILES 数据集加载器
├── Utils/                 # 工具函数（日志、评估、特征转换等）
└── save/                  # 训练保存路径
```

---

## 快速开始

### 1. 安装环境

```bash
conda create -n mol-prompt python=3.10 -y
conda activate mol-prompt
pip install -r requirements.txt
```

---

### 2. 运行模型

#### 预训练

```bash
python main_pretrain.py \
  --dataset QM9SMILES \
  --device 0 \
  --epochs 100 \
  --batch_size 64 \
  --save save/Molecule \
  --model_save_path save/Molecule/model.pth
```

输出模型包括：

* `model.pth`
* `learnable_prompt.pth`
* `checkpoint_*.pt`
* `log.txt`

---

#### 药物-药物相互作用预测（DDI）

```bash
python main_DDI.py \
  --dataset DeepDDI \
  --device 0 \
  --epochs 100 \
  --batch_size 32 \
  --model_load_path save/Molecule/model.pth \
  --model_save_path save_ds/DDI/model_ds_ddi.pth
```

可选数据集：`Ryu` 或 `Deng`

---

#### 分子属性预测（MPP）

```bash
python main_MPP.py \
  --dataset ogbg-moltox21 \
  --device 0 \
  --epochs 100 \
  --split scaffold \
  --model_load_path save/Molecule/model.pth \
  --model_save_path save_ds/MPP/model_ds_mpp.pth
```

可选数据集包括：

* ogbg-moltox21
* ogbg-molbace
* ogbg-molsider
* ogbg-molbbbp
* ogbg-molclintox
* ogbg-molesol
* ogbg-molfreesolv
* ogbg-mollipo

---

## 评估指标

各任务支持的评估指标：

| 任务类型      | 评估指标                          |
|-----------| ----------------------------- |
| 原子预测      | F1 Score                      |
| 边预测       | ROC-AUC                       |
| 分子距离回归    | RMSE                          |
| DDI 多分类预测 | F1, Accuracy, Precision, AUPR |
| MPP性质预测   | 分类任务（AUC） / 回归任务（RMSE）        |

---

## 🧠 模型结构说明

* **Prompt\_SMILES**：主编码器，结合 SMILES 字符序列与位置嵌入
* **LearnablePrompt**：任务可学习提示向量（支持多个任务）
* **Predictors**：

  * `Atom_Predictor`、`Edge_Predictor`、`TDDistance_Predictor`
  * `DDI_Predictor`、`SMILES_Predictor`

---

