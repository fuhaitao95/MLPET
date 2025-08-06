import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges, negative_sampling
from tqdm import tqdm

import torch.nn.functional as F

from Prompt_model.Predictor import SMILES_Predictor, Edge_Predictor, Atom_Predictor, TDDistance_Predictor
from Prompt_model.Prompt import Prompt_SMILES, LearnablePrompt
from Prompt_model.converter import PromptConverter

from Prompt_model.token_SMILES import Alphabet
from Utils.evalUtil import eval_rmse, eval_acc, eval_rocauc, eval_F1_multiclass, eval_F1, compute_class_weights
from Utils.logUtil import log_Util

from pretrain_dataset.QM9SMILESDataset import QM9SMILESDataset

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


# 设置随机种子函数
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


# Loss function combining both tasks
def combined_loss(td_loss, edge_loss, atom_loss, alpha1=0.4, alpha2=0.3, alpha3=0.3):
    total_loss = alpha1 * td_loss + alpha2 * edge_loss + alpha3 * atom_loss
    return total_loss


# Example training loop
def train(model, device, train_loader, optimizer, converter, learnable_prompt, dist_predictor, edge_predictor,
          atom_predictor, weight_1, weight_2, weight_3, class_weights_tensor):
    model.train()

    for batch in tqdm(train_loader, desc="Training", leave=True):
        optimizer.zero_grad()
        # 获取 SMILES 图的输入和目标 如果 prompt分开训练要在其中加 显式提示信息

        # 前向传播 prompt 训练模式

        # 原子任务
        encoded = converter(batch.masked_smiles, ["<atm>"]).to(device)
        # 获取mask2节点类型 index
        atom_mask_index = batch.atom_mask_idx
        masked_char = converter.encoder(batch.masked_char).squeeze(-1).to(device)
        x = model(encoded, learnable_prompt=learnable_prompt(0))['logits']
        atom_output = atom_predictor(x, atom_mask_index)
        # 计算综合损失
        atom_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(atom_output, masked_char)

        # 边任务
        # 获取边连接信息和边的二分类标签
        edge_index = batch.train_pos_edge_index.to(device)
        neg_edge_index = negative_sampling(
            edge_index=batch.train_pos_edge_index,
            num_nodes=batch.num_nodes,
            num_neg_samples=batch.train_pos_edge_index.size(1)).to(device)

        encoded = converter(batch.smiles, ["<edg>"]).to(device)
        # 获取每个节点类型 分类任务
        atom_mask = [converter.code2atomadj(toks.to(device)) for toks in encoded]
        x = model(encoded, learnable_prompt=learnable_prompt(1))['logits']
        link_logits = edge_predictor(x, atom_mask, edge_index, neg_edge_index)
        link_labels = get_link_labels(batch.train_pos_edge_index, neg_edge_index).to(device)
        edge_loss = F.binary_cross_entropy_with_logits(link_logits[:, 0], link_labels)

        # 3D坐标任务
        indices = batch.mol_3d_edge_index.to(device)
        true_distances = batch.mol_3d_dist.to(device)

        # 将3d的edge_index带权重 转化为矩阵
        pos_mask = batch.mol_dist_mask.squeeze(-1).bool().to(device)
        encoded = converter(batch.smiles, ["<tdp>"]).to(device)
        atom_mask = [converter.code2atomadj(toks.to(device)) for toks in encoded]
        x = model(encoded, learnable_prompt=learnable_prompt(2))['logits']
        dist_logits = dist_predictor(x, atom_mask, indices)

        # 从模型预测的矩阵中提取对应的元素
        # 使用 td_loss 进行 mask
        # 计算 True 的占比
        true_ratio = pos_mask.float().mean().item()
        dist_logits = dist_logits[pos_mask]
        true_distances = true_distances[pos_mask]
        td_loss = nn.MSELoss()(dist_logits, true_distances)

        # 更新权重
        weight_3 *= true_ratio
        weight_1 = weight_2 = (1 - weight_3) / 2

        # 根据双任务进行分析
        # 计算综合损失
        loss = combined_loss(atom_loss, edge_loss, td_loss, weight_1, weight_2, weight_3)

        # 反向传播和优化
        loss.backward()
        optimizer.step()


def eval(model, device, loader, converter, learnable_prompt, dist_predictor, edge_predictor, atom_predictor):
    model.eval()

    all_true_positions = []
    all_predicted_positions = []
    all_atom_preds = []
    all_atom_labels = []
    all_edge_preds = []
    all_edge_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=True):
            # 获取每个节点类型 分类任务

            # 原子任务
            encoded = converter(batch.masked_smiles, ["<atm>"]).to(device)
            # 获取mask2节点类型 index
            atom_mask_index = batch.atom_mask_idx
            masked_char = converter.encoder(batch.masked_char).squeeze(-1).to(device)
            x = model(encoded, learnable_prompt=learnable_prompt.prompts[0])['logits']
            atom_output = atom_predictor(x, atom_mask_index)

            # 边任务
            # 获取边连接信息和边的二分类标签
            encoded = converter(batch.smiles, ["<edg>"]).to(device)
            # 获取每个节点类型 分类任务
            atom_mask = [converter.code2atomadj(toks.to(device)) for toks in encoded]
            x = model(encoded, learnable_prompt=learnable_prompt.prompts[1])['logits']

            for prefix in ['val', 'test']:
                pos_edge_index = batch[f'{prefix}_pos_edge_index']
                neg_edge_index = batch[f'{prefix}_neg_edge_index']
                link_logits = edge_predictor(x, atom_mask, pos_edge_index, neg_edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                # 保存边预测结果和真实边标签
                all_edge_preds.append(link_probs.cpu())
                all_edge_labels.append(link_labels.cpu())

            # 3D坐标任务
            indices = batch.mol_3d_edge_index.to(device)
            true_distances = batch.mol_3d_dist.to(device)

            pos_mask = batch.mol_dist_mask.squeeze(-1).bool().to(device)

            encoded = converter(batch.smiles, ["<tdp>"]).to(device)
            atom_mask = [converter.code2atomadj(toks.to(device)) for toks in encoded]
            x = model(encoded, learnable_prompt=learnable_prompt.prompts[2])['logits']
            dist_logits = dist_predictor(x, atom_mask, indices)

            # td_loss 进行mask
            dist_logits = dist_logits[pos_mask]
            true_distances = true_distances[pos_mask]

            # Store true and predicted positions
            all_true_positions.append(true_distances.cpu())
            all_predicted_positions.append(dist_logits.cpu())

            # 保存边预测结果和真实边标签
            all_atom_preds.append(atom_output.cpu())
            all_atom_labels.append(masked_char.cpu())

    # 将所有 SMILES 预测值和目标值堆叠在一起
    all_true_positions = torch.cat(all_true_positions, dim=0)  # [total_atoms, 3]
    all_predicted_positions = torch.cat(all_predicted_positions, dim=0)  # [total_atoms, 3]

    # 计算 RMSE 评估 空间距离任务
    rmse_result = eval_rmse(all_true_positions, all_predicted_positions)['rmse']

    # 计算 acc 评估边预测任务
    # 将边预测和真实边标签展平并拼接
    all_edge_labels = torch.cat(all_edge_labels, dim=0).numpy()
    all_edge_preds = torch.cat(all_edge_preds, dim=0).numpy()
    edge_result = eval_rocauc(all_edge_labels, all_edge_preds)['rocauc']

    # 将所有 atom 预测值和目标值堆叠在一起
    all_atom_preds = torch.cat(all_atom_preds, dim=0)
    all_atom_labels = torch.cat(all_atom_labels, dim=0)
    # 计算 f1 评估任务
    atom_result = eval_F1_multiclass(all_atom_labels, all_atom_preds)['F1']

    return rmse_result, edge_result, atom_result


def main():
    parser = argparse.ArgumentParser(description='(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--max_position_num', type=int, default=1340,
                        help="QM9 max position 29 + 2 + 1 之后根据下游数据集一起计算 DeepDDI 834 DrugBank_mm 439")
    parser.add_argument('--layer_num', type=int, default=5, help="33 layers in paper")
    parser.add_argument('--attention_head_num', type=int, default=16, help="20 heads in paper")
    parser.add_argument('--embed_dim', type=int, default=256, help="1280 dims in paper")
    parser.add_argument('--ffn_embed_dim', type=int, default=256)
    parser.add_argument("--emb_layer_norm_before", action="store_true", default=False)
    parser.add_argument("--token_dropout", action="store_true", default=True)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001, help="1 × 10−4 in papers")
    parser.add_argument('--weight_0', type=float, default=0.33, help="loss计算权重")
    parser.add_argument('--weight_1', type=float, default=0.33, help="loss计算权重")
    parser.add_argument('--weight_2', type=float, default=0.34, help="loss计算权重")

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--transformer_dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=100, help="270k steps of updates in papers")
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1)

    # save model
    parser.add_argument('--split', type=str, default='random', help='scaffold random')
    parser.add_argument('--dataset', type=str, default="QM9SMILES", help='dataset name')
    parser.add_argument('--sub_name', type=str, default='pretrain')
    parser.add_argument('--save', type=str, default='save/Molecule', help='experiment name')
    parser.add_argument('--model_training_save_path', type=str, default='save',
                        help='the directory used to save models')
    parser.add_argument('--model_save_path', type=str, default='model.pth',
                        help='the directory used to save models')
    parser.add_argument('--model_encoder_load_path', type=str, default='',
                        help='the path of trained encoder')

    # load trained model for test
    parser.add_argument('--model_load_path', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--model_direct_load_path', type=str, default='',
                        help='the path of trained model')
    args = parser.parse_args()

    log_util = log_Util(args)
    args = log_util.save_exp()
    log = logging.getLogger()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 创建数据集
    smiles_dataset = QM9SMILESDataset(root='pretrain_dataset/QM9SMILES')

    # 按 8:2 比例划分数据集
    dataset_size = len(smiles_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = int(0.1 * dataset_size)
    valid_size = dataset_size - train_size - test_size

    train_dataset, test_dataset, valid_dataset = random_split(smiles_dataset, [train_size, test_size, valid_size])
    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型和优化器
    dictionary = Alphabet.build_alphabet()
    model = Prompt_SMILES(args, dictionary).to(device)
    converter = PromptConverter(dictionary)
    learnable_prompt = LearnablePrompt(args, dictionary.num_prompt).to(device)
    dist_predictor = TDDistance_Predictor(args).to(device)
    edge_predictor = Edge_Predictor(args).to(device)
    atom_predictor = Atom_Predictor(args, dictionary.num_standard_atom).to(device)

    class_weights_tensor = compute_class_weights(train_dataset, dictionary.num_standard_atom,converter).to(device)

    # 初始化优化器，包含所有需要优化的模块参数
    optimizer = Adam(
        list(model.parameters()) +
        list(learnable_prompt.parameters()) +
        list(dist_predictor.parameters()) +
        list(edge_predictor.parameters()) +
        list(atom_predictor.parameters()),
        lr=args.lr
    )

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
            learnable_prompt.load_state_dict(checkpoint['learnable_prompt_state_dict'])
            dist_predictor.load_state_dict(checkpoint['dist_predictor_state_dict'])
            edge_predictor.load_state_dict(checkpoint['edge_predictor_state_dict'])
            atom_predictor.load_state_dict(checkpoint['atom_predictor_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # 继续从下一个 epoch 开始
        else:
            start_epoch = 1
    else:
        start_epoch = 1

    train_curve, test_curve, valid_curve = [], [], []

    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"=====Epoch {epoch}=====")

        log.info('Training...')
        train(model, device, train_loader, optimizer, converter, learnable_prompt, dist_predictor, edge_predictor,
              atom_predictor, args.weight_0, args.weight_1, args.weight_2, class_weights_tensor)

        log.info('Evaluating...')
        # train_rmse, train_rocauc, train_f1 = eval(model, device, train_loader, converter, learnable_prompt,
        #                                           dist_predictor,
        #                                           edge_predictor, atom_predictor)
        # test_rmse, test_rocauc, test_f1 = eval(model, device, test_loader, converter, learnable_prompt,
        #                                        dist_predictor,
        #                                        edge_predictor, atom_predictor)
        # valid_rmse, valid_rocauc, valid_f1 = eval(model, device, valid_loader, converter, learnable_prompt,
        #                                           dist_predictor,
        #                                           edge_predictor, atom_predictor)

        # log.info({
        #     'Train RMSE': train_rmse, 'Train ROCAUC': train_rocauc, 'Train f1': train_f1,
        #     'Test RMSE': test_rmse, 'Test ROCAUC': test_rocauc, 'Test f1': test_f1,
        #     'Validation RMSE': valid_rmse, 'Validation ROCAUC': valid_rocauc, 'Validation f1': valid_f1,
        # })

        # 保存训练、验证、测试的性能曲线
        # train_curve.append((train_rmse, train_rocauc, train_f1))
        # test_curve.append((test_rmse, test_rocauc, test_f1))
        # valid_curve.append((valid_rmse, valid_rocauc, valid_f1))

        # 定期保存模型检查点
        if args.model_training_save_path != '' and epoch % 50 == 0:
            if not os.path.exists(args.model_training_save_path):
                os.makedirs(args.model_training_save_path)
            path = os.path.join(args.model_training_save_path, f'checkpoint_1129_{epoch}.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learnable_prompt_state_dict': learnable_prompt.state_dict(),
                'dist_predictor_state_dict': dist_predictor.state_dict(),
                'edge_predictor_state_dict': edge_predictor.state_dict(),
                'atom_predictor_state_dict': atom_predictor.state_dict(),
            }
            torch.save(checkpoint, path)
            log.info(f"Checkpoint saved to {path}")

    # # 选择验证 RMSE 最小的模型
    # best_val_epoch = np.argmin([x[0] for x in valid_curve])
    # best_train = min([x[0] for x in train_curve])
    #
    # log.info('Finished training!')
    # log.info(f'Best train RMSE: {best_train:.4f}')
    # log.info(f'Best validation RMSE: {valid_curve[best_val_epoch][0]:.4f}')
    # log.info(f'Test RMSE: {test_curve[best_val_epoch][0]:.4f}')
    #
    # # 选择验证 ROCAUC 最大的模型
    # best_val_epoch = np.argmax([x[1] for x in valid_curve])
    # best_train = max([x[1] for x in train_curve])
    #
    # log.info('Finished training!')
    # log.info(f'Best train ROCAUC: {best_train:.4f}')
    # log.info(f'Best validation ROCAUC: {valid_curve[best_val_epoch][1]:.4f}')
    # log.info(f'Test ROCAUC: {test_curve[best_val_epoch][1]:.4f}')
    #
    # # 选择验证 f1 最大的模型
    # best_val_epoch = np.argmax([x[2] for x in valid_curve])
    # best_train = max([x[2] for x in train_curve])
    #
    # log.info('Finished training!')
    # log.info(f'Best train F1: {best_train:.4f}')
    # log.info(f'Best validation F1: {valid_curve[best_val_epoch][2]:.4f}')
    # log.info(f'Test F1: {test_curve[best_val_epoch][2]:.4f}')

    if not args.model_save_path == '':
        torch.save(model, args.model_save_path)
        path = args.model_save_path.replace('model', 'learnable_prompt')
        torch.save(learnable_prompt, f'{path}')


if __name__ == '__main__':
    main()
