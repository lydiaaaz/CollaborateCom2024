# 消息聚合器（允许通过不同的策略对节点接收到的消息进行聚合）
from collections import defaultdict
import torch
from typing import List, Dict, Tuple
import numpy as np


class MessageAggregator(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(MessageAggregator, self).__init__()
        self.device = device

    # 接收节点的ID列表和一个包含节点消息的字典，然后根据特定策略对消息进行聚合
    # 返回值是一个包含三个元素的元组：聚合后的节点ID列表、聚合后的消息张量和聚合后的时间戳张量
    def aggregate(self, node_ids: List[int], messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]) -> \
            Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Given a list of node ids, and a dict of messages of nodes in the list, aggregate different
        messages for the same id using one of the possible strategies
        :param node_ids: a list of node ids of length batch_size
        :param messages: a dict that stores messages for nodes, and message[i][j] is a tuple of
                        (message_vector,timestamp) which indicate the message vector and timestamp of the
                        `j` the interaction that node `i` occurs in
        :return: a tuple of (to_update_node_ids,unique_messages,unique_timestamps), where
                  to_update_node_ids  is a list of length n_unique_node_ids with the unique node ids
                  unique_messages is a tensor of shape (n_unique_node_ids, message_dim) with the aggregated messages
                  unique_timestamps is a tensor of shape (n_unique_node_ids, 1) with aggregated timestamps
        """


# 对于每个节点，仅保留其接收到的消息中的最后一条
class LastMessageAggregator(MessageAggregator):
    def __init__(self, device: torch.device):
        super(LastMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """Keep the last message for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []

        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                # Last Update
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


# 默认，对于每个节点，计算其接收到的所有消息的均值
class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device: torch.device):
        super(MeanMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """Mean all messages for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []
        n_messages = 0

        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                n_messages += len(messages[node_id])
                to_update_node_ids.append(node_id)
                unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
                unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type: str, device: torch.device) -> MessageAggregator:
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":  # 默认
        return MeanMessageAggregator(device=device)
    else:
        raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
