
# MLPET: A Multi-Level Prompt-Enhanced Transformer for Unified Molecular Property and Drug-Drug Interaction event Prediction

## 1 Abstract

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


## 2 Introduction

This project implements a prompt-based learning framework for SMILES sequences, enabling multi-task joint pretraining and transfer evaluation on various downstream tasks, including:

- Distance Prediction
- Atom Prediction Prediction
- Bond Prediction
- DDI Classification
- Molecular Property Prediction

---

## 3 Project Structure

```text
.
├── main_pretrain.py       # Multi-task joint pretraining
├── main_DDI.py            # DDI classification task
├── main_MPP.py            # Molecular property prediction (OGB datasets)
├── Prompt_model/          # Encoders, prompt modules, predictors
├── dataset/               # Loaders for DeepDDI / DrugBank / OGB
├── pretrain_dataset/      # QM9SMILES dataset loader
├── Utils/                 # Utilities (logging, evaluation, feature conversion)
└── save/                  # Directory for saved models and logs
```

---

## 4 Quick Start

### 4.1 Environment Setup

```bash
conda create -n mol-prompt python=3.10 -y
conda activate mol-prompt
pip install -r requirements.txt
```

---

### 4.2 Run Models

#### 4.2.1 Pretraining

```bash
python main_pretrain.py \
  --dataset QM9SMILES \
  --device 0 \
  --epochs 100 \
  --batch_size 64 \
  --save save/Molecule \
  --model_save_path save/Molecule/model.pth
```

Output：

* `model.pth`
* `learnable_prompt.pth`
* `checkpoint_*.pt`
* `log.txt`

---

#### 4.2.2 DDI Event Prediction

```bash
python main_DDI.py \
  --dataset DeepDDI \
  --device 0 \
  --epochs 100 \
  --batch_size 32 \
  --model_load_path save/Molecule/model.pth \
  --model_save_path save_ds/DDI/model_ds_ddi.pth
```

Optional datasets: `Ryu` or `Deng`

---

#### 4.2.3 Molecular Property Prediction

```bash
python main_MPP.py \
  --dataset ogbg-moltox21 \
  --device 0 \
  --epochs 100 \
  --split scaffold \
  --model_load_path save/Molecule/model.pth \
  --model_save_path save_ds/MPP/model_ds_mpp.pth
```

Optional datasets: 

* ogbg-moltox21
* ogbg-molbace
* ogbg-molsider
* ogbg-molbbbp
* ogbg-molclintox
* ogbg-molesol
* ogbg-molfreesolv
* ogbg-mollipo

---

## 5 Evaluation Metrics



| Task      | Metrics                          |
|-----------| ----------------------------- |
| Atom Prediction      | F1 Score                      |
| Bond Prediction       | ROC-AUC                       |
| Distance Regression    | RMSE                          |
| DDI Classification | F1, Accuracy, Precision, AUPR |
| MPP   | AUC (classification) / RMSE (regression)        |

---

## 6 Model Components

* **Prompt\_SMILES**: Main encoder combining SMILES sequences with positional embeddings
* **LearnablePrompt**: LearnablePrompt: Trainable task-specific prompt vectors
* **Predictors**：

  * `Atom_Predictor`、`Edge_Predictor`、`TDDistance_Predictor`
  * `DDI_Predictor`、`SMILES_Predictor`

---

