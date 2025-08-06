import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Prompt_model.Predictor import DDI_Predictor
from Prompt_model.Prompt import Prompt_SMILES, LearnablePrompt, PromptTuningModule
from Prompt_model.converter import PromptConverter
from Prompt_model.token_SMILES import Alphabet
from Utils.datasetUtil import stratified_split, oversample_rare_classes
from Utils.evalUtil import eval_F1_multiclass, compute_class_weights4ddi, eval_acc_multiclass, \
    eval_rocauc_multiclass, eval_aupr_multiclass, eval_precision_multiclass
from Utils.logUtil import log_Util, TqdmToLogger
from dataset.DeepDDI.DeepDDI_dataset import DrugDDIDataset
from dataset.DrugBank_multimodal.DrugBankMLDataset import EventDDIDataset

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


# todo 制作原子属性下游任务

#  training loop
def train(model, device, train_loader, optimizer, converter, learnable_prompt, prompt_tuning_module, ddi_predictor,
          class_weights_tensor):
    model.train()
    log = logging.getLogger()
    # tqdm_out = TqdmToLogger(log)

    total_loss = 0  # 累积损失
    num_batches = 0  # 批次数量
    # all_loss = []
    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        # 获取 SMILES 图的输入和目标
        smiles_input_1 = batch.drug1_smiles
        smiles_input_2 = batch.drug2_smiles
        # 获取边连接信息和边地二分类标签
        y_true = batch.label.to(device)  # 边的标签 (是否存在)

        encoded = converter(smiles_input_1 + smiles_input_2).to(device)

        # 前向传播
        combined_prompt = prompt_tuning_module([learnable_prompt(i) for i in range(learnable_prompt.num_prompts)])
        x = model(encoded, learnable_prompt=combined_prompt)['logits']

        y_predict = ddi_predictor(x, len(batch))

        # 计算综合损失
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # Edge prediction task loss (e.g., binary classification of edges)
        loss = criterion(y_predict, y_true)

        # 累积损失和批次数量
        total_loss += loss.item()
        num_batches += 1
        # all_loss.append(total_loss)
        # 反向传播和优化
        loss.backward()
        optimizer.step()

    # 计算并打印平均损失
    avg_loss = total_loss / num_batches
    return avg_loss


def eval(model, device, loader, converter, learnable_prompt, prompt_tuning_module, ddi_predictor):
    model.eval()

    all_preds = []
    all_targets = []
    log = logging.getLogger()
    # tqdm_out = TqdmToLogger(log)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            # 获取 SMILES 图的输入和目标
            # smiles_input = batch.x.to(device)  # 节点特征
            smiles_input_1 = batch.drug1_smiles
            smiles_input_2 = batch.drug2_smiles
            # 获取边连接信息和边地二分类标签
            y_true = batch.label.to(device)  # 边的标签

            encoded = converter(smiles_input_1 + smiles_input_2).to(device)

            # 前向传播
            combined_prompt = prompt_tuning_module([learnable_prompt(i) for i in range(learnable_prompt.num_prompts)])
            x = model(encoded, learnable_prompt=combined_prompt)['logits']

            y_predict = ddi_predictor(x, len(batch))

            all_preds.append(y_predict.cpu())
            all_targets.append(y_true.cpu())

    # 将所有 SMILES 预测值和目标值堆叠在一起
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # 计算 f1 评估任务
    result = {
        **eval_F1_multiclass(all_targets, all_preds),
        **eval_acc_multiclass(all_targets, all_preds),
        **eval_rocauc_multiclass(all_targets, all_preds),
        **eval_aupr_multiclass(all_targets, all_preds),
        **eval_precision_multiclass(all_targets, all_preds),
    }
    # precision, recall, f1 = result['precision'], result['recall'], result['F1']
    return result


def main():
    parser = argparse.ArgumentParser(description='(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--max_position_num', type=int, default=837,
                        help="QM9 max position 29 + 2 + 1 之后根据下游数据集一起计算 DeepDDI 834 DrugBank_mm 439")
    parser.add_argument('--layer_num', type=int, default=5, help="33 layers in paper")
    parser.add_argument('--predictor_layer_num', type=int, default=3, help="33 layers in paper")
    parser.add_argument('--attention_head_num', type=int, default=16, help="20 heads in paper")
    parser.add_argument('--embed_dim', type=int, default=256, help="1280 dims in paper")
    parser.add_argument('--ffn_embed_dim', type=int, default=256)
    parser.add_argument("--emb_layer_norm_before", action="store_true", default=False)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--transformer_dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)

    # save model
    parser.add_argument('--dataset', type=str, default="DeepDDI", help='dataset name: Multimodal DeepDDI')
    parser.add_argument('--split', type=str, default='random', help='scaffold random')
    parser.add_argument('--sub_name', type=str, default='e1')
    parser.add_argument('--save', type=str, default='save_ds/DDI', help='experiment name')
    parser.add_argument('--model_training_save_path', type=str, default='save_ds/',
                        help='the directory used to save models')
    parser.add_argument('--model_save_path', type=str, default='model_ds_ddi.pth',
                        help='the directory used to save models')
    # parser.add_argument('--model_load_path', type=str, default="")
    parser.add_argument('--model_load_path', type=str, default="save/model.pth")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
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
    if args.dataset == "Deng":
        dataset = EventDDIDataset(root="dataset/DrugBank_multimodal")
    elif args.dataset == "Ryu":
        dataset = DrugDDIDataset(root="dataset/DeepDDI")
    else:
        dataset = EventDDIDataset(root="dataset/DrugBank_multimodal")

    # 定义模型参数
    # 初始化模型和优化器
    dictionary = Alphabet.build_alphabet()
    model = Prompt_SMILES(args, dictionary).to(device)
    converter = PromptConverter(dictionary)
    learnable_prompt = LearnablePrompt(args, dictionary.num_prompt).to(device)
    if not args.model_load_path == '':
        model_path = Path(args.model_load_path)
        # 检查模型文件是否存在
        if model_path.exists():
            model = torch.load(args.model_load_path).to(device)
            for param in model.parameters():
                param.requires_grad = False
            path = args.model_load_path.replace('model', 'learnable_prompt')
            learnable_prompt = torch.load(f'{path}').to(device)

    # Prompt tuning module
    prompt_tuning_module = PromptTuningModule(args.embed_dim, learnable_prompt.num_prompts).to(device)
    ddi_predictor = DDI_Predictor(args, dataset.num_tasks).to(device)
    # 初始化优化器，包含所有需要优化的模块参数
    optimizer = Adam(
        list(model.parameters()) +
        list(prompt_tuning_module.parameters()) +  # todo 是否固定
        list(ddi_predictor.parameters()),
        lr=args.lr
    )

    # 数据集处理
    augmented_dataset = oversample_rare_classes(dataset)
    class_weights_tensor = compute_class_weights4ddi(augmented_dataset, dataset.num_tasks).to(device)

    # 按 8:2 比例划分数据集
    dataset_size = len(augmented_dataset)
    # train_size = 1
    train_size = int(0.8 * dataset_size)
    # test_size = 1
    test_size = int(0.1 * dataset_size)
    valid_size = dataset_size - train_size - test_size

    # 使用当前时间戳生成种子
    seed = int(time.time() * 1000) % (2 ** 32)
    if not args.seed == 0:
        seed = args.seed
    log.info(f"Generated seed from time: {seed}")
    train_dataset, test_dataset, valid_dataset = stratified_split(augmented_dataset,
                                                                  [train_size, test_size, valid_size], seed)

    # _ = compute_class_weights4ddi(train_dataset, dataset.num_tasks).to(device)
    # _ = compute_class_weights4ddi(test_dataset, dataset.num_tasks).to(device)
    # _ = compute_class_weights4ddi(valid_dataset, dataset.num_tasks).to(device)

    # 使用 DataLoader 加载数据  drop_last=True)
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
            ddi_predictor.load_state_dict(checkpoint['ddi_predictor_state_dict'])
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
                         ddi_predictor,
                         class_weights_tensor)

        log.info('Evaluating...')
        # train_results = eval(model, device, train_loader, converter, learnable_prompt,
        #                      prompt_tuning_module, ddi_predictor)
        test_results = eval(model, device, test_loader, converter, learnable_prompt,
                            prompt_tuning_module, ddi_predictor)
        valid_results = eval(model, device, valid_loader, converter, learnable_prompt,
                             prompt_tuning_module, ddi_predictor)

        log.info({
            'Average Training Loss': avg_loss,
            # 'Train precision': train_results['precision'], 'Train recall': train_results['recall'],
            # 'Train f1': train_results['F1'], 'Train acc': train_results['acc'], 'Train rocauc': train_results['rocauc'],
            # 'Train aupr': train_results['aupr'],
            'Test precision': test_results['precision'], 'Test recall': test_results['recall'],
            'Test f1': test_results['F1'], 'Test acc': test_results['acc'], 'Test rocauc': test_results['rocauc'],
            'Test aupr': test_results['aupr'],
            'Validation precision': valid_results['precision'], 'Validation recall': valid_results['recall'],
            'Validation f1': valid_results['F1'], 'Validation acc': valid_results['acc'],
            'Validation rocauc': valid_results['rocauc'],
            'Validation aupr': valid_results['aupr'],
        })

        # log_util.add_result(0, epoch, avg_loss, train_results, test_results, valid_results)
        log_util.add_resultwithouttrain(0, epoch, avg_loss, test_results, valid_results)

        # 保存训练、验证、测试的性能曲线
        # train_curve.append((train_results['precision'], train_results['recall'], train_results['F1']))
        test_curve.append((test_results['precision'], test_results['recall'], test_results['F1']))
        valid_curve.append((valid_results['precision'], valid_results['recall'], valid_results['F1']))

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
                'ddi_predictor_state_dict': ddi_predictor.state_dict(),
            }
            torch.save(checkpoint, path)
            log.info(f"Checkpoint saved to {path}")

    log.info('Finished training!')
    # 选择验证 f1 最大的模型
    best_val_epoch = np.argmax([x[2] for x in valid_curve])
    # best_train = max([x[2] for x in train_curve])

    # log.info(f'Best train precision: {best_train:.4f}')
    log.info(f'Best validation f1: {valid_curve[best_val_epoch][2]:.4f}')
    log.info(f'Test precision: {test_curve[best_val_epoch][0]:.4f}')
    log.info(f'Test recall: {test_curve[best_val_epoch][1]:.4f}')
    log.info(f'Test f1: {test_curve[best_val_epoch][2]:.4f}')

    if not args.model_save_path == '':
        log_util.save_to_csv()
        torch.save(model, args.model_save_path)
        path = args.model_save_path.replace('model', 'prompt_tuning_module')
        torch.save(prompt_tuning_module, path)
        path = args.model_save_path.replace('model', 'ddi_predictor')
        torch.save(ddi_predictor, path)


if __name__ == '__main__':
    main()
