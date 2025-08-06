from collections import defaultdict

from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import Subset
import torch
import numpy as np
import random


def stratified_split(dataset, lengths, seed=42):
    """
    对数据集进行分层划分，确保每个数据集中标签分布一致。

    Args:
        dataset: 数据集。
        lengths: 数据集数量。

    Returns:
        train_dataset, valid_dataset, test_dataset: 分层划分后的子集。
    """
    train_size, test_size, valid_size = lengths

    # 转换标签为 NumPy 数组，确保兼容性
    labels = [data.label.item() for data in dataset]
    labels = tensor(labels).numpy()

    # 划分训练集和临时集
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=(test_size + valid_size),
        stratify=labels,
        random_state=seed  # 设置随机种子
    )

    # 从临时集中进一步划分验证集和测试集
    valid_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_size,
        stratify=labels[temp_indices],
        random_state=seed  # 设置随机种子
    )

    # 根据索引创建子集
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    assert len(train_dataset) + len(valid_dataset) + len(test_dataset) == len(dataset)
    return train_dataset, valid_dataset, test_dataset


# 对稀有类别进行过采样
def oversample_rare_classes(dataset, min_samples=10):
    label_to_samples = defaultdict(list)
    for data in dataset:
        label_to_samples[data.label.item()].append(data)

    augmented_dataset = []
    for label, samples in label_to_samples.items():
        if len(samples) < min_samples:
            samples = samples * (min_samples // len(samples)) + samples[:min_samples % len(samples)]
        augmented_dataset.extend(samples)
    return augmented_dataset


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
