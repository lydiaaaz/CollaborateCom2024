import torch
import numpy as np
import torch.nn as nn
from typing import Union, Mapping, Dict


class TimeDifferenceEncoder(torch.nn.Module):
    def __init__(self, dimension: int):
        super(TimeDifferenceEncoder, self).__init__()
        self.dimension = dimension # 编码器输出的向量维度
        self.w = torch.nn.Linear(1, dimension) # 线性层，将一个标量输入映射到维度为 dimension 的向量

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1)) # 权重是一个对数尺度的递减序列，用于将时间差异映射到向量
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float()) # 初始化线性层的权重和偏置

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Mapping the time difference into a vector
        :param t: time difference, a tensor of shape (batch)
        :return: time vector, a tensor of shape (batch,dimension)
        """
        # t has shape [batch_size]
        t = t.unsqueeze(dim=1) # 在张量的第二个维度上增加一个维度，形状从 [batch_size] 变为 [batch_size, 1]
        output = torch.cos(self.w(t)) # 使用余弦函数和线性层的组合将t映射为一个向量
        return output


class TimeSlotEncoder(torch.nn.Module):
    def __init__(self, dimension: int, max_time: float, time_num: int):
        super(TimeSlotEncoder, self).__init__()
        self.max_time = max_time # 时间的最大值
        self.time_num = time_num # 时间槽的数量
        self.emb = nn.Embedding(time_num, dimension) # 嵌入层，将时间戳映射到维度为 dimension 的向量

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Mapping a timestamp into a vector
        :param t: timestamp, a tensor of shape (batch)
        :return: time vector, a tensor of shape (batch,dimension)
        """
        t = (t / self.max_time * (self.time_num - 1)).to(torch.long)
        return self.emb(t) # 使用嵌入层将t映射为一个向量


def get_time_encoder(model_type: str, dimension: int, max_time: Dict[str, float] = None, time_num: int = 20,
                     single: bool = False) -> Mapping[str, Union[TimeDifferenceEncoder, TimeSlotEncoder]]:
    # model_type 表示时间编码器的类型（'difference' 或 'slot'）
    # dimension 表示输出向量的维度
    # max_time 表示最大时间的字典
    # time_num 表示时间槽的数量
    # single 表示是否使用相同的编码器
    if model_type == 'difference':
        user_time_encoder = TimeDifferenceEncoder(dimension)
        if single:
            cas_time_encoder = user_time_encoder
        else:
            cas_time_encoder = TimeDifferenceEncoder(dimension)
        return nn.ModuleDict({'user': user_time_encoder, 'cas': cas_time_encoder})
    elif model_type == 'slot':
        return nn.ModuleDict({
            'user': TimeSlotEncoder(dimension, max_time['user'], time_num),
            'cas': TimeSlotEncoder(dimension, max_time['cas'], time_num)
        })
    else:
        raise ValueError("Not Implemented Model Type")
