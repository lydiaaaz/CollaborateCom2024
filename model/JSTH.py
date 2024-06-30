# MyModel.py
# 处理动态图上的节点状态更新和级联预测任务，其中动态状态和时间信息对模型的设计起到关键作用
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from model.encoder.state.dynamic_state import DynamicState
from model.encoder.state.state_updater import get_state_updater
from model.encoder.message.message_generator import get_message_generator
from model.decoder.embedding_module import get_embedding_module
from model.decoder.prediction import get_predictor
from model.time_encoder import get_time_encoder
from utils.hgraph import HGraph


class MainModel(nn.Module):
    def __init__(self, device: torch.device, node_dim: int = 100, embedding_module_type: str = "seq",
                 state_updater_type: str = "gru", predictor: str = 'linear', time_enc_dim: int = 8,
                 single: bool = False, ntypes: set = None, dropout: float = 0.1, n_nodes: Dict = None,
                 max_time: float = None, use_static: bool = False, merge_prob: float = 0.5,
                 max_global_time: float = 0, use_dynamic: bool = False, use_temporal: bool = False,
                 use_structural: bool = False):
        # 接受一系列超参数，包括设备类型、节点维度、嵌入模块类型、状态更新器类型、预测器类型
        super(MainModel, self).__init__()
        if max_time is None:
            max_time = {'user': 1, 'cas': 1}
        self.ntypes = ntypes
        self.device = device
        self.predictor_type=predictor
        self.cas_num = n_nodes['cas']
        self.user_num = n_nodes['user']
        self.single = single
        self.hgraph = HGraph(num_user=n_nodes['user'], num_cas=n_nodes['cas'])  # 创建动态图
        self.time_encoder = get_time_encoder('difference', dimension=time_enc_dim, single=self.single) # 时间编码器
        self.use_dynamic = use_dynamic
        self.use_static = use_static
        self.dynamic_state = nn.ModuleDict({  # 动态状态
            'user': DynamicState(n_nodes['user'], state_dimension=node_dim,
                                 input_dimension=node_dim, message_dimension=node_dim,
                                 device=device, single=False),
            'cas': DynamicState(n_nodes['cas'], state_dimension=node_dim,
                                input_dimension=node_dim, message_dimension=node_dim,
                                device=device, single=True)})
        self.init_state()
        self.message_generator = get_message_generator(generator_type='concat', state=self.dynamic_state,
                                                       time_encoder=self.time_encoder,
                                                       time_dim=time_enc_dim,
                                                       message_dim=node_dim, node_feature_dim=node_dim,
                                                       device=self.device, message_aggregator_type='mean',
                                                       single=single, max_time=max_time)

        self.state_updater = get_state_updater(module_type=state_updater_type,
                                               state=self.dynamic_state,
                                               message_dimension=node_dim,
                                               state_dimension=node_dim,
                                               device=self.device, single_updater=single, ntypes=ntypes)
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     dynamic_state=self.dynamic_state, embedding_dimension=node_dim,
                                                     device=self.device, dropout=dropout, hgraph=self.hgraph,
                                                     input_dimension=node_dim, max_time=max_time['cas'],
                                                     use_static=use_static, user_num=n_nodes['user'],
                                                     max_global_time=max_global_time, use_dynamic=use_dynamic,
                                                     use_temporal=use_temporal, use_structural=use_structural)
        self.predictor = get_predictor(emb_dim=node_dim, predictor_type=predictor, merge_prob=merge_prob)

    def update_state(self):  # 更新动态状态
        if self.use_dynamic:
            for ntype in self.ntypes:
                self.dynamic_state[ntype].store_cache()

    def forward(self, source_nodes: np.ndarray, destination_nodes: np.ndarray, trans_cascades: np.ndarray,
                edge_times: torch.Tensor, pub_times: torch.Tensor, target_idx: np.ndarray) -> torch.Tensor:
        # 接受一批次的交互数据，包括源节点、目标节点、级联 ID、边的时间、级联发布时间以及目标索引
        """
        given a batch of interactions, update the corresponding nodes' dynamic states and give the popularity of the
        cascades that have reached the observation time.
        :param source_nodes: the sending users' id of the interactions, ndarray of shape (batch)
        :param destination_nodes: the receiving users' id of the interactions, ndarray of shape (batch)
        :param trans_cascades: the cascade id of the interactions,ndarray of shape (batch)
        :param edge_times: the happening timestamps of the interactions, tensor of shape (batch)
        :param pub_times: the publication timestamps of the cascades in the interactions, tensor of shape (batch)
        :param target_idx: a mask tensor to indicating which cascade has reached the observation time,
               tensor of shape (batch)
        :return pred: the popularity of cascades that have reached the observation time, tensor of shape (batch)
        """
        if self.use_dynamic:  # 调用消息生成器和状态更新器，根据输入的交互数据更新节点的动态状态
            nodes, messages, times = self.message_generator.get_message(source_nodes, destination_nodes,
                                                                        trans_cascades, edge_times, pub_times, 'all')
            self.state_updater.update_state(nodes, messages, times)
        self.hgraph.insert(trans_cascades, source_nodes, destination_nodes, edge_times, pub_times) # 动态图被更新，记录了节点之间的交互关系和时间信息
        target_cascades = trans_cascades[target_idx]  # 针对到达观察时间的级联
        pred = torch.zeros(len(trans_cascades)).to(self.device)
        event_ll = torch.zeros(len(trans_cascades)).to(self.device)
        non_event_ll = torch.zeros(len(trans_cascades)).to(self.device)
        # if len(target_cascades) > 0:
        #     emb = self.embedding_module.compute_embedding(target_cascades)  # 通过嵌入模块获取级联的嵌入表示
        #     pred[target_idx] = self.predictor.forward(emb)  # 通过预测器得到级联的预测结果
        if len(target_cascades) > 0:
            if self.use_static and self.use_dynamic:
                emb1, emb2, event_ll[target_idx], non_event_ll[target_idx] = self.embedding_module.compute_embedding(target_cascades)
                if self.predictor_type == "fusion":
                    pred[target_idx] = self.predictor.forward(emb1, emb2)
                    # pred[target_idx] = self.predictor.forward(emb1, emb2).squeeze(dim=1)
                else:
                    emb = (emb1, emb2)
                    pred[target_idx] = self.predictor.forward(emb)
            else:
                emb, _ = self.embedding_module.compute_embedding(target_cascades)  # 通过嵌入模块获取级联的嵌入表示
                pred[target_idx] = self.predictor.forward(emb)  # 通过预测器得到级联的预测结果
        return pred, event_ll, non_event_ll  # 返回级联的预测值

    def init_state(self):  # 初始化动态状态和图
        for ntype in self.ntypes:
            self.dynamic_state[ntype].__init_state__()
        self.hgraph.init()

    def reset_state(self):  # 重置动态状态和图
        for ntype in self.ntypes:
            self.dynamic_state[ntype].reset_state()
        self.hgraph.init()

    def detach_state(self):  # 分离动态状态
        for ntype in self.ntypes:
            self.dynamic_state[ntype].detach_state()

# 模块结构：
# 动态状态 (DynamicState)：
    # 使用 nn.ModuleDict 存储不同类型节点的动态状态，如用户节点和级联节点。
    # 包括 store_cache、reset_state、detach_state 等方法用于更新、重置和分离状态。
# 消息生成器 (MessageGenerator)：
    # 根据输入的交互数据、时间信息等生成节点之间的消息。
    # 使用消息生成器的 get_message 方法获取消息。
# 状态更新器 (StateUpdater)：
    # 根据给定的 RNN 单元类型更新动态状态。
    # 通过 get_state_updater 函数创建不同类型节点的状态更新器。
# 嵌入模块 (EmbeddingModule)：
    # 根据更新后的动态状态计算节点嵌入表示。
    # 使用 get_embedding_module 函数创建不同类型节点的嵌入模块。
# 预测器 (Predictor)：
    # 根据节点嵌入表示进行级联的预测。
    # 使用 get_predictor 函数创建预测器。
# 动态图 (HGraph)：
    # 用于管理和维护动态图中的节点和边信息。
# 输入：交互数据，包括源节点、目标节点、级联 ID、边的时间、级联发布时间以及目标索引
# 输出：针对级联的预测值
