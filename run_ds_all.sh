#!/bin/bash

# # 两个 DDI 数据集名称
# ddi_datasets=(
#   "Multimodal"
#   "DeepDDI"
# )

# # 遍历模型和数据集
# for sub_name in "${!model_map[@]}"; do
#   model_dir=${model_map[$sub_name]}

#   for dataset in "${ddi_datasets[@]}"; do
#     echo ">>> Running main_DDI.py | Dataset: $dataset | Model: $model_dir"

#     python main_DDI.py \
#       --dataset "$dataset" \
#       --model_load_path "save_ds/${model_dir}/model.pth" \
#       --sub_name "$sub_name"
#   done
# done

# OGB 分子性质预测数据集
mpp_datasets=(
  "ogbg-molbace"
  "ogbg-molbbbp"
  # "ogbg-molsider"
  "ogbg-molclintox"
  "ogbg-moltox21"
  "ogbg-molesol"
  "ogbg-molfreesolv"
  "ogbg-mollipo"  
)

# 每个模型 x 每个数据集运行一次（如需多次可加 for run in $(seq 0 9)）

for dataset in "${mpp_datasets[@]}"; do
    echo ">>> Running main_MPP.py | Dataset: $dataset"

    python main_MPP.py --dataset "$dataset"
done

python main_MPP.py --dataset "ogbg-molsider" --batch_size 16
