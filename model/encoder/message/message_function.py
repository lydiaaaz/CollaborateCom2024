# 生成消息表征
import torch
from torch import nn


class MessageFunction(nn.Module):
    """
    Module which computes the message for a given interaction.
    """

    def compute_message(self, raw_messages: torch.Tensor):
        """generate message for an interaction"""
        return None


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int):  # 构造函数接收原始消息维度raw_message_dimension和目标消息维度message_dimension
        super(MLPMessageFunction, self).__init__()

        self.mlp = nn.Sequential(  # 一个简单的MLP模型，使用两个线性层和ReLU激活函数
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
            nn.ReLU(),
        )

    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """
        :param raw_messages: raw message, tensor of shape (batch,raw_message_dim)
        :return message: returned message, tensor of shape (batch,message_dim)
        """
        messages = self.mlp(raw_messages)
        return messages


class IdentityMessageFunction(MessageFunction):
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """
        :param raw_messages: raw message, tensor of shape (batch,raw_message_dim)
        :return message: returned message, tensor of shape (batch,raw_message_dim)
        """
        return raw_messages


def get_message_function(module_type: str, raw_message_dimension: int, message_dimension: int) -> MessageFunction:
    if module_type == "mlp":  # 默认，通过多层感知机（MLP）生成消息
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":  # 直接返回原始消息，即起到恒等映射的作用
        return IdentityMessageFunction()
    else:
        raise NotImplementedError(f"Not message function type {module_type}")
