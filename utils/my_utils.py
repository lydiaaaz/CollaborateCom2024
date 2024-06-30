import json
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import pickle as pk
from sklearn.metrics import mean_absolute_error, mean_squared_error


def save_model(model: nn.Module, save_path, run): # 保存PyTorch模型的权重
    torch.save(model.state_dict(), f'{save_path}_{run}.pth') # model.state_dict() 返回模型的当前参数状态字典，其中键是参数的名称，值是对应的张量


def load_model(model: nn.Module, load_path, run): # 加载预训练的模型权重
    model_dict = torch.load(f"{load_path}_{run}.pth")
    model.load_state_dict(model_dict)


class EarlyStopMonitor(object): # Early Stop 监控器，用于在训练过程中根据验证集性能停止训练，防止过拟合
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10, save_path=None, logger=None,
                 model: nn.Module = None,
                 run=0):
        self.max_round = max_round # 最大轮数
        self.num_round = 0
        self.run = run

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance # 允许的误差容忍度
        self.save_path = save_path # 保存模型的路径
        self.logger = logger
        self.model = model

    def early_stop_check(self, curr_val): # 检查是否满足早期停止条件
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            save_model(self.model, self.save_path, self.run)
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            save_model(self.model, self.save_path, self.run)
        else:
            self.num_round += 1
        self.epoch_count += 1
        if self.num_round <= self.max_round:
            return False
        return True


def set_config(args): # 根据传入的参数设置配置信息
    param = vars(args)
    param['prefix'] = f'{args.prefix}_{args.dataset}' # 日志保存前缀
    param['model_path'] = f"saved_models/{param['prefix']}" # 模型保存路径
    param['result_path'] = f"results/{param['prefix']}" # 结果保存路径
    param['log_path'] = f"log/{param['prefix']}.log"
    data_config = json.load(open('config/config.json', 'r'))[param['dataset']]
    param.update(data_config)
    return param


def msle(pred, label): # 均方对数误差
    return np.around(mean_squared_error(label, pred, multioutput='raw_values'), 4)[0]


def male(pred, label): # 平均绝对误差
    return np.around(mean_absolute_error(label, pred, multioutput='raw_values'), 4)[0]


def mape(pred, label): # 平均百分比误差
    label = 2 ** label
    pred = 2 ** pred
    result = np.mean(np.abs(np.log2(pred + 1) - np.log2(label + 1)) / np.log2(label + 2))
    return np.around(result, 4)


def pcc(pred, label): # 皮尔逊相关系数
    pred_mean, label_mean = np.mean(pred, axis=0), np.mean(label, axis=0)
    pre_std, label_std = np.std(pred, axis=0), np.std(label, axis=0)
    return np.around(np.mean((pred - pred_mean) * (label - label_mean) / (pre_std * label_std), axis=0), 4)


def accuracy(pred, label):
    """
    计算分类模型的准确率
    参数:
    pred -- 预测值，类型为数组或列表
    label -- 标签值，类型为数组或列表
    返回:
    acc -- 准确率，类型为浮点数
    """
    # 检查预测值和标签值的长度是否一致
    if len(pred) != len(label):
        raise ValueError("预测值和标签值的长度不一致")
    # 计算正确预测的数量
    correct = sum(1 for p, l in zip(pred, label) if p == l)
    # 计算总样本数量
    total = len(pred)
    # 计算准确率
    acc = correct / total
    return acc


class Metric: # 跟踪和计算训练、验证和测试阶段的评价指标
    def __init__(self, path, logger, fig_path):
        self.template = {'target': [], 'pred': [], 'label': [], 'msle': 0, 'male': 0, 'pcc': 0, 'mape': 0, 'loss': 0}
        self.final = {'train': deepcopy(self.template), 'val': deepcopy(self.template), 'test': deepcopy(self.template)}
        self.history = {'train': {'msle': [], 'male': [], 'mape': [], 'pcc': [], 'loss': []},
                        'val': {'msle': [], 'male': [], 'mape': [], 'pcc': [], 'loss': []},
                        'test': {'msle': [], 'male': [], 'mape': [], 'pcc': [], 'loss': []},
                        }
        self.temp = None
        self.path = path
        self.fig_path = fig_path
        self.logger = logger

    def fresh(self): # 初始化临时数据
        self.temp = {'train': deepcopy(self.template), 'val': deepcopy(self.template), 'test': deepcopy(self.template)}

    def update(self, target, pred, label, dtype): # 更新数据
        self.temp[dtype]['target'].append(target)
        self.temp[dtype]['pred'].append(pred)
        self.temp[dtype]['label'].append(label)

    def calculate_metric(self, dtype, move_history=True, move_final=False, loss=0): # 计算指标
        targets, preds, labels = self.temp[dtype]['target'], self.temp[dtype]['pred'], self.temp[dtype]['label']

        targets, preds, labels = np.concatenate(targets, axis=0), \
                                 np.concatenate(preds, axis=0), \
                                 np.concatenate(labels, axis=0)
        self.temp[dtype]['target'] = targets
        self.temp[dtype]['pred'] = preds
        self.temp[dtype]['label'] = labels
        self.temp[dtype]['msle'] = msle(preds, labels)
        self.temp[dtype]['male'] = male(preds, labels)
        self.temp[dtype]['mape'] = mape(preds, labels)
        self.temp[dtype]['pcc'] = pcc(preds, labels)
        self.temp[dtype]['loss'] = loss

        if move_history:
            for metric in ['msle', 'male', 'mape', 'pcc', 'loss']:
                self.history[dtype][metric].append(self.temp[dtype][metric])
        if move_final:
            self.move_final(dtype)
        return deepcopy(self.temp[dtype])

    def move_final(self, dtype): # 移动到最终结果
        self.final[dtype] = self.temp[dtype]

    def save(self): # 将最终结果保存到文件
        pk.dump(self.final, open(self.path, 'wb'))
        # with open(self.path, 'a') as f:
        #     f.write(str(self.final))

    def info(self, dtype): # 打印评价指标的信息
        s = []
        for metric in ['loss', 'msle', 'male', 'mape', 'pcc']:
            s.append(f'{metric}:{self.temp[dtype][metric]:.4f}')
        self.logger.info(f'{dtype}: ' + '\t'.join(s))
