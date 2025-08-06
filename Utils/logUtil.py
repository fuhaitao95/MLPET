import logging
import os
import shutil
import sys
import time

import pandas as pd
import torch


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


# Redirect tqdm output to logging
class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, msg):
        # Remove extra newlines
        if msg.strip():
            self.logger.log(self.level, msg.strip())

    def flush(self):
        pass


class log_Util(object):
    def __init__(self, parser):
        self.args = parser
        self.results = [[] for _ in range(parser.runs)]

    def save_exp(self):
        self.args.save = '{}-D_{}-S_{}-DR_{}-L_{}-A_{}-DIM_{}-B_{}-LR_{}-E_{}'.format(self.args.save + "_" + self.args.sub_name,
                                                                           self.args.dataset,
                                                                           self.args.split,
                                                                           self.args.dropout,
                                                                           self.args.layer_num,
                                                                           self.args.attention_head_num,
                                                                           self.args.embed_dim, self.args.batch_size,
                                                                           self.args.lr, self.args.epochs)

        self.args.save = '{}-{}'.format(self.args.save, time.strftime("%Y%m%d-%H%M"))
        if not self.args.model_save_path == '':
            self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)
            # create_exp_dir(self.args.save, scripts_to_save=glob.glob('*.py'))
            create_exp_dir(self.args.save)
            fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        else:
            fh = logging.FileHandler('{}-{}'.format(self.args.save, '.txt'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format=log_format,
                            datefmt='%m/%d %I:%M:%S')
        fh.setFormatter(logging.Formatter(log_format))
        log = logging.getLogger()
        log.addHandler(fh)

        return self.args

    def add_result(self, run, epoch, loss, train_metrics, test_metrics, valid_metrics):
        """
        添加一个 epoch 的结果。

        Args:
            run (int): 当前运行的索引。
            epoch (int ): 当前 epoch。
            loss (float): 平均损失。
            train_metrics (dict): 训练集指标。
            valid_metrics (dict): 验证集指标。
            test_metrics (dict): 测试集指标。
        """
        assert 0 <= run < len(self.results)

        # 合并所有结果
        result = {
            'epoch': epoch,
            'loss': loss,
        }

        # 将训练、验证和测试的指标动态添加到 result 中
        for prefix, metrics in zip(['train', 'valid', 'test'], [train_metrics, valid_metrics, test_metrics]):
            for metric_name, value in metrics.items():
                result[f'{prefix}_{metric_name}'] = value

        # 添加到结果集合
        self.results[run].append(result)

    def add_resultwithouttrain(self, run, epoch, loss, test_metrics, valid_metrics):
        """
        添加一个 epoch 的结果。

        Args:
            run (int): 当前运行的索引。
            epoch (int ): 当前 epoch。
            loss (float): 平均损失。
            valid_metrics (dict): 验证集指标。
            test_metrics (dict): 测试集指标。
        """
        assert 0 <= run < len(self.results)

        # 合并所有结果
        result = {
            'epoch': epoch,
            'loss': loss,
        }

        # 将训练、验证和测试的指标动态添加到 result 中
        for prefix, metrics in zip(['valid', 'test'], [valid_metrics, test_metrics]):
            for metric_name, value in metrics.items():
                result[f'{prefix}_{metric_name}'] = value

        # 添加到结果集合
        self.results[run].append(result)

    def save_to_csv(self):
        """
        将结果保存到 CSV 文件。

        Args:
            filepath (str): 文件路径。
        """
        for i in range(len(self.results)):
            filepath = os.path.join(self.args.save, f'result_{i}.csv')
            df = pd.DataFrame(self.results[i])
            df.to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")

    # def print_statistics(self, run=None):
    #     if run is not None:
    #         result = 100 * torch.tensor(self.results[run])
    #         argmax = result[:, 1].argmax().item()
    #         print(f'Run {run + 1:02d}:')
    #         print(f'Highest Train: {result[:, 0].max():.2f}')
    #         print(f'Highest Valid: {result[:, 1].max():.2f}')
    #         print(f'  Final Train: {result[argmax, 0]:.2f}')
    #         print(f'   Final Test: {result[argmax, 2]:.2f}')
    #     else:
    #         result = 100 * torch.tensor(self.results)
    #
    #         best_results = []
    #         for r in result:
    #             train1 = r[:, 0].max().item()
    #             valid = r[:, 1].max().item()
    #             train2 = r[r[:, 1].argmax(), 0].item()
    #             test = r[r[:, 1].argmax(), 2].item()
    #             best_results.append((train1, valid, train2, test))
    #
    #         best_result = torch.tensor(best_results)
    #
    #         print(f'All runs:')
    #         r = best_result[:, 0]
    #         print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
    #         r = best_result[:, 1]
    #         print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
    #         r = best_result[:, 2]
    #         print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
    #         r = best_result[:, 3]
    #         print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
