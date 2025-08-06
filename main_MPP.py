import argparse
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from ogb.graphproppred import Evaluator
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Prompt_model.Predictor import DDI_Predictor, SMILES_Predictor
from Prompt_model.Prompt import Prompt_SMILES, LearnablePrompt, PromptTuningModule
from Prompt_model.converter import PromptConverter
from Prompt_model.token_SMILES import Alphabet
from Utils.evalUtil import eval_F1_multiclass, eval_rmse
from Utils.logUtil import log_Util
from dataset.moleculeDataset.moleculedatase_smiles import PygGraphPropPredDataset_Smiles
from pretrain_dataset.QM9SMILESDataset import QM9SMILESDataset

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = SmoothL1Loss()

from torch.cuda.amp import autocast, GradScaler
# todo 制作原子属性下游任务

scaler = GradScaler()

def train(model, device, train_loader, optimizer, converter, learnable_prompt, prompt_tuning_module, smiles_predictor, task_type):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        y_true = batch.y.to(device)
        encoded = converter(batch.smiles).to(device)

        combined_prompt = prompt_tuning_module([
            learnable_prompt(i) for i in range(learnable_prompt.num_prompts)
        ])

        with autocast():  # 自动混合精度（减少显存占用 + 更快）
            # logits = model(encoded, learnable_prompt=combined_prompt)['logits']
            logits = model(encoded)['logits']
            y_predict = smiles_predictor(logits)
            is_labeled = batch.y == batch.y

            if "classification" in task_type:
                loss = cls_criterion(y_predict[is_labeled], y_true[is_labeled].float())
            else:
                loss = reg_criterion(y_predict[is_labeled], y_true[is_labeled])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def eval(model, device, loader, converter, learnable_prompt, prompt_tuning_module, smiles_predictor, evaluator):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            # 获取 SMILES 图的输入和目标
            y_true = batch.y.to(device)
            encoded = converter(batch.smiles).to(device)

            # 前向传播
            combined_prompt = prompt_tuning_module([learnable_prompt(i) for i in range(learnable_prompt.num_prompts)])
            # x = model(encoded, learnable_prompt=combined_prompt)['logits']
            x = model(encoded)['logits']

            y_predict = smiles_predictor(x)

            all_preds.append(y_predict.cpu())
            all_targets.append(y_true.cpu())

    # 将所有 SMILES 预测值和目标值堆叠在一起
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    input_dict = {"y_true": all_targets, "y_pred": all_preds}

    return evaluator.eval(input_dict)


def main():
    parser = argparse.ArgumentParser(description='(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--max_position_num', type=int, default=1340,
                        help="QM9 max position 29 + 2 + 1 之后根据下游数据集一起计算 DeepDDI 834 DrugBank_mm 439 Sider 1024")
    parser.add_argument('--layer_num', type=int, default=5, help="33 layers in paper")
    parser.add_argument('--predictor_layer_num', type=int, default=1, help="33 layers in paper")
    parser.add_argument('--attention_head_num', type=int, default=16, help="20 heads in paper")
    parser.add_argument('--embed_dim', type=int, default=256, help="1280 dims in paper")
    parser.add_argument('--ffn_embed_dim', type=int, default=256)
    parser.add_argument("--emb_layer_norm_before", action="store_true", default=False)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--transformer_dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)

    # save model
    parser.add_argument('--dataset', type=str, default="ogbg-molbace"
                        , help='dataset name: ogbg-molbace ogbg-molbbbp ogbg-molsider ogbg-molclintox ogbg-moltox21 '
                               'ogbg-molesol ogbg-molfreesolv ogbg-mollipo')
    parser.add_argument('--split', type=str, default='scaffold', help='scaffold random')
    parser.add_argument('--sub_name', type=str, default='freeze')
    parser.add_argument('--save', type=str, default='save_ds/MPP', help='experiment name')
    parser.add_argument('--model_training_save_path', type=str, default='save_ds/MPP_freeze-D_ogbg-molbace-S_scaffold-DR_0.5-L_5-A_16-DIM_256-B_128-LR_0.001-E_100-20250730-1022',
                        help='the directory used to save models')
    parser.add_argument('--model_save_path', type=str, default='model_ds_mpp.pth',
                        help='the directory used to save models')
    # parser.add_argument('--model_load_path', type=str, default="")
    parser.add_argument('--model_load_path', type=str, default="save/Molecule_pretrain-D_QM9SMILES-L_5-A_16-DIM_256-B_64-LR_0.0001-E_100-20241211-1517/model.pth")
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0, help='sider:3635328891 tox21:3638369060')
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    log_util = log_Util(args)
    args = log_util.save_exp()
    log = logging.getLogger()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 创建数据集
    # dataset = DrugDDIDataset(root="dataset/DeepDDI")
    # dataset = QM9SMILESDataset(root="pretrain_dataset/QM9SMILES")
    dataset = PygGraphPropPredDataset_Smiles(root="dataset/moleculeDataset", name=args.dataset)
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    # 定义模型参数
    # 初始化模型和优化器
    dictionary = Alphabet.build_alphabet()
    converter = PromptConverter(dictionary)
    model = Prompt_SMILES(args, dictionary).to(device)
    learnable_prompt = LearnablePrompt(args, dictionary.num_prompt).to(device)
    if not args.model_load_path == '':
        model_path = Path(args.model_load_path)
        # 检查模型文件是否存在
        if model_path.exists():
            model = torch.load(args.model_load_path, map_location=device).to(device)
            for param in model.parameters():
                param.requires_grad = True
            path = args.model_load_path.replace('model', 'learnable_prompt')
            learnable_prompt = torch.load(f'{path}', map_location=device).to(device)
            log.info(f"model loaded: {args.model_load_path}")

    # Prompt tuning module
    prompt_tuning_module = PromptTuningModule(args.embed_dim, learnable_prompt.num_prompts).to(device)
    smiles_predictor = SMILES_Predictor(args, dataset.num_tasks).to(device)
    # 初始化优化器，包含所有需要优化的模块参数
    optimizer = Adam(
        list(model.parameters()) +
        list(prompt_tuning_module.parameters()) +  # todo 是否固定
        list(smiles_predictor.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # 按 8:2 比例划分数据集
    dataset_size = len(dataset)
    # train_size = 1
    train_size = int(0.8 * dataset_size + 0.5)
    # test_size = 1
    test_size = int(0.1 * dataset_size)
    valid_size = dataset_size - train_size - test_size

    # 使用当前时间戳生成种子
    seed = int(time.time() * 1000) % (2 ** 32)
    if not args.seed == 0:
        seed = args.seed
    log.info(f"Generated seed from time: {seed}")
    generator = torch.Generator().manual_seed(seed)

    if args.split == 'scaffold':
        # 将数据按照设置的训练集验证集测试集编号分类
        split_idx = dataset.get_idx_split()
        train_dataset, test_dataset, valid_dataset = dataset[split_idx["train"]], dataset[split_idx["test"]], dataset[
            split_idx["valid"]]
    else:
        train_dataset, test_dataset, valid_dataset = random_split(dataset, [train_size, test_size, valid_size],
                                                                  generator)

    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # checkpoint 初始化
    # 检查是否存在已保存的模型
    if args.model_training_save_path != '':
        # 获取保存路径下的所有检查点文件
        checkpoints = [f for f in os.listdir(args.model_training_save_path) if
                       f.startswith('checkpoint_') and f.endswith('.pt')]
        if len(checkpoints) > 0:
            # 按照 epoch 数排序，获取最新的检查点
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.pt')[0]), reverse=True)
            latest_checkpoint = checkpoints[0]
            checkpoint_path = os.path.join(args.model_training_save_path, latest_checkpoint)
            log.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            prompt_tuning_module.load_state_dict(checkpoint['prompt_tuning_module_state_dict'])
            smiles_predictor.load_state_dict(checkpoint['smiles_predictor_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # 继续从下一个 epoch 开始
        else:
            start_epoch = 1
    else:
        start_epoch = 1

    train_curve, test_curve, valid_curve = [], [], []

    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"=====Epoch {epoch}=====")

        log.info('Training...')
        avg_loss = train(model, device, train_loader, optimizer, converter, learnable_prompt, prompt_tuning_module,
                         smiles_predictor, dataset.task_type)

        log.info('Evaluating...')
        # train_results = eval(model, device, train_loader, converter, learnable_prompt,
        #                      prompt_tuning_module, smiles_predictor, evaluator)
        test_results = eval(model, device, test_loader, converter, learnable_prompt,
                            prompt_tuning_module, smiles_predictor, evaluator)
        valid_results = eval(model, device, valid_loader, converter, learnable_prompt,
                             prompt_tuning_module, smiles_predictor, evaluator)

        log.info({
            'Average Training Loss': avg_loss,
            # f'Train {dataset.eval_metric}': train_results[dataset.eval_metric],
            f'Test {dataset.eval_metric}': test_results[dataset.eval_metric],
            f'Validation {dataset.eval_metric}': valid_results[dataset.eval_metric]
        })

        # log_util.add_result(0, epoch, avg_loss, train_results, test_results, valid_results)
        log_util.add_resultwithouttrain(0, epoch, avg_loss, test_results, valid_results)

        # 保存训练、验证、测试的性能曲线
        # train_curve.append(train_results[dataset.eval_metric])
        valid_curve.append(valid_results[dataset.eval_metric])
        test_curve.append(test_results[dataset.eval_metric])

        # 定期保存模型检查点
        if args.model_training_save_path != '' and epoch % 50 == 0:
            if not os.path.exists(args.model_training_save_path):
                os.makedirs(args.model_training_save_path)
            path = os.path.join(args.model_training_save_path, f'checkpoint_{epoch}.pt')
            path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'prompt_tuning_module_state_dict': prompt_tuning_module.state_dict(),
                'smiles_predictor_state_dict': smiles_predictor.state_dict(),
            }
            torch.save(checkpoint, path)
            log.info(f"Checkpoint saved to {path}")

    log.info('Finished training!')
    # 选择验证 rmse 最小的模型
    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        # best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        # best_train = min(train_curve)

    log.info('Finished training!')
    log.info('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    log.info('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.model_save_path == '':
        log_util.save_to_csv()
        torch.save(model, args.model_save_path)
        path = args.model_save_path.replace('model', 'prompt_tuning_module')
        torch.save(prompt_tuning_module, path)
        path = args.model_save_path.replace('model', 'smiles_predictor')
        torch.save(smiles_predictor, path)


if __name__ == '__main__':
    main()
