# 用于表示图的动态状态
import torch
from torch import nn
import numpy as np
from typing import Sequence


class DynamicState(nn.Module):
    def __init__(self, n_nodes: int, state_dimension: int, input_dimension: int, message_dimension: int = None,
                 device: torch.device = None, single: bool = False):
        super(DynamicState, self).__init__()
        self.n_nodes = n_nodes  # 节点数量
        self.state_dimension = state_dimension  # 状态维度
        self.input_dimension = input_dimension  # 输入维度
        self.message_dimension = message_dimension
        self.device = device
        self.is_single = single
        self.__init_state__()

    def __init_state__(self):  # 初始化具有参数的状态，如state['src']和state['dst']（如果is_single为False）
        self.state = nn.ParameterDict().to(self.device)
        # cache is used to store the updated states in each batch
        self.cache = dict()  # 初始化一个缓存字典
        state_dim = self.state_dimension
        state_src = nn.Parameter(torch.zeros((self.n_nodes, state_dim)).to(self.device),
                                 requires_grad=False)
        self.state['src'] = state_src
        self.cache['src'] = []
        if not self.is_single:
            self.cache['dst'] = []
            self.state['dst'] = nn.Parameter(torch.zeros((self.n_nodes, state_dim)).to(self.device),
                                             requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                        requires_grad=False)

    # 检索指定节点的状态
    def get_state(self, node_idxs: Sequence, type: str = 'src', from_cache: bool = False) -> torch.Tensor:
        if from_cache:
            node_map, temp_idx, temp_state = self.cache[type]
            return temp_state[list(map(lambda x: node_map[x], node_idxs))]
        else:
            return self.state[type][node_idxs, :]

    # 设置指定节点的状态
    def set_state(self, node_idxs: Sequence, values: torch.Tensor, type: str = 'src', set_cache: bool = False):
        if set_cache:
            node_map = dict(zip(node_idxs, np.arange(len(node_idxs))))
            self.cache[type] = [node_map, node_idxs, values]
        else:
            self.state[type][node_idxs, :] = values.detach()

    def get_last_update(self, node_idxs: Sequence):  # 获取指定节点的最后更新值
        return self.last_update[node_idxs]

    def set_last_update(self, node_idxs: Sequence, values: torch.Tensor): # 设置指定节点的最后更新值
        self.last_update[node_idxs] = values

    def detach_state(self):  # 分离源状态和目标状态
        self.state['src'].detach_()
        if not self.is_single:
            self.state['dst'].detach_()

    def reset_state(self):  # 将状态重置为全零（在每个周期的开始调用）
        """
        Reinitialize the state to all zeros. It should be called at the start of each epoch.
        """
        for u in self.state:
            self.state[u].data = self.state[u].new_zeros(self.state[u].shape)
            self.cache[u] = []
        self.last_update.data = self.last_update.new_zeros(self.last_update.shape)

    def store_cache(self):  # 将缓存中的值更新到状态中
        for ntype in self.cache:
            _, temp_node_idx, temp_state = self.cache[ntype]
            self.state[ntype][temp_node_idx, :] = temp_state
            self.cache[ntype] = []
