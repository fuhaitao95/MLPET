import logging

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, precision_score, recall_score

from sklearn.metrics import f1_score
from collections import Counter



# 假设 QM9SMILESDataset 中的 masked_char 字段包含所有类别的标签
def compute_class_weights(dataset, num_classes, converter):
    """
    计算类别权重，返回一个 PyTorch 张量。

    Args:
        dataset: 数据集，假设每个数据项是包含 'masked_char' 的字典。
        num_classes: int，总类别数。

    Returns:
        torch.Tensor: 类别权重张量，形状为 [num_classes]。
    """
    # 检查数据集格式
    if not all('masked_char' in data for data in dataset):
        raise ValueError("Dataset items must contain the 'masked_char' field.")

    # 统计每个类别的出现次数
    temp = converter.encoder([data['masked_char'] for data in dataset]).squeeze(-1)
    class_counts = torch.bincount(temp, minlength=num_classes)
    # 总样本数
    total_samples = class_counts.sum().item()

    # 计算类别权重：总样本数 / (类别数 * 每个类别的频次)
    # 避免除以 0
    class_weights = total_samples / (num_classes * class_counts.clamp(min=1).float())
    class_weights[class_counts == 0] = 1e-2
    class_weights = class_weights / class_weights.sum()
    return class_weights


def compute_class_weights4ddi(dataset, num_classes):
    """
    计算类别权重，返回一个 PyTorch 张量。

    Args:
        dataset: 数据集。
        num_classes: int，总类别数。

    Returns:
        torch.Tensor: 类别权重张量，形状为 [num_classes]。
    """

    # 统计每个标签的频率
    labels = [data.label.item() for data in dataset]
    label_counts = Counter(labels)

    # 打印每个标签的样本数量
    # print("Label Frequencies:")
    # for label, count in label_counts.items():
    #     print(f"Label {label}: {count} samples")

    # 总样本数
    total_samples = sum(label_counts.values())

    # 初始化权重张量
    class_weights = torch.ones(num_classes, dtype=torch.float)

    # 计算类别权重：总样本数 / (类别数 * 每个类别的频次)
    for label, count in label_counts.items():
        if count > 0:
            class_weights[label] = total_samples / (num_classes * count)

    # 归一化权重
    class_weights = class_weights / class_weights.sum()

    return class_weights


def eval_F1(seq_ref, seq_pred):
    # '''
    #     compute F1 score averaged over samples
    # '''

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
        label = set(l)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {'precision': np.average(precision_list),
            'recall': np.average(recall_list),
            'F1': np.average(f1_list)}


def eval_F1_multiclass(targets, logits):
    """
    计算多分类任务中的 F1 分数（宏平均）。

    Args:
        logits: 2D 数组或张量，形状为 [n_batch, n_class]，模型输出的 logits。
        targets: 1D 数组或张量，形状为 [n_batch]，真实标签。

    Returns:
        dict: 包含 precision, recall, 和 F1 的字典。
    """
    # 如果 logits 是 tensor，转为 numpy

    # 从 logits 中获取预测类别
    preds = np.argmax(logits, axis=1)
    # reuslt = eval_F1(targets, preds)
    # 计算 precision, recall 和 F1 分数

    f1 = f1_score(targets, preds, average='macro')

    return {'F1': f1}


def eval_rocauc_multiclass(targets, logits, average='micro'):
    try:
        # 从 logits 中获取预测概率
        probabilities = softmax(logits, axis=1)

        # 检查 probabilities 的合法性
        # if not np.allclose(np.sum(probabilities, axis=1), 1.0):
        #     raise ValueError("Probabilities do not sum to 1 for all samples.")

        # 检查 targets 的形状
        # if len(targets.shape) == 1:
        #     num_classes = logits.shape[1]
        #     targets = np.eye(num_classes)[targets]  # 转换为 one-hot

        # 计算 ROC-AUC 分数
        score = roc_auc_score(targets, probabilities, multi_class="ovr", average=average)
        return {'rocauc': score}

    except Exception as e:
        # 捕获异常并记录
        logging.getLogger().info(f"Warning: An error occurred in eval_rocauc_multiclass - {str(e)}")
        return {'rocauc': 0.0}


def eval_rocauc(y_true, y_pred):
    '''
    Compute ROC-AUC for edge prediction tasks.
    '''
    # AUC is only defined when有至少一个正类和负类数据
    if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
        # 计算 ROC-AUC 分数
        rocauc = roc_auc_score(y_true, y_pred)
    else:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': rocauc}


def eval_rmse(y_true, y_pred):
    '''
        compute RMSE score averaged across tasks
    '''
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return {'rmse': sum(rmse_list) / len(rmse_list)}


def eval_acc_multiclass(targets, logits):
    # 从 logits 中获取预测类别
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(targets, preds)

    return {'acc': acc}


def eval_acc(y_true, y_pred, threshold=0.5, batch_size=10000):
    '''
    Compute accuracy for binary classification tasks (e.g., edge prediction) using batch processing.

    Args:
        y_true: 1D numpy array of ground truth labels (binary: 0 or 1).
        y_pred: 1D numpy array of predicted probabilities (or binary predictions).
        threshold: Threshold to convert probabilities into binary predictions if necessary.
        batch_size: Size of batches for processing large datasets to prevent memory overload.

    Returns:
        Dictionary containing accuracy.
    '''

    total_correct = 0
    total = len(y_true)

    # Batch processing
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        # 处理当前批次的预测和真实标签
        y_true_batch = y_true[start:end]
        y_pred_batch = y_pred[start:end]

        # 将预测的概率转化为二值预测 (0 or 1)，根据给定的阈值
        y_pred_binary = (y_pred_batch >= threshold).astype(int).squeeze()

        # 计算当前批次的正确预测数
        total_correct += np.sum(y_true_batch == y_pred_binary)

    # 计算准确率
    accuracy = float(total_correct) / total

    return {'acc': accuracy}


def eval_aupr_multiclass(targets, logits):
    num_classes = logits.shape[1]
    logits = softmax(logits, axis=1)
    aupr_per_class = {}
    for i in range(num_classes):
        y_true = targets[:]
        y_scores = logits[:, i]
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=i)
        aupr = auc(recall, precision)
        aupr_per_class[f'class_{i}'] = aupr
    macro_aupr = np.mean(list(aupr_per_class.values()))
    return {'aupr': macro_aupr}

    # # 从 logits 中获取预测类别
    # preds = np.argmax(logits, axis=1)
    #
    # # Micro-AUPR (global precision-recall curve)
    # micro_precision, micro_recall, _ = precision_recall_curve(
    #     targets.ravel(), logits.ravel()
    # )
    # micro_aupr = auc(micro_recall, micro_precision)
    #
    # return {'aupr': micro_aupr}micro_aupr


def eval_precision_multiclass(targets, logits, average="macro"):
    # 从 logits 中获取预测类别
    preds = np.argmax(logits, axis=1)
    macro_precision = precision_score(targets, preds, average=average)
    macro_recall = recall_score(targets, preds, average=average)

    return {'precision': macro_precision, 'recall': macro_recall}
