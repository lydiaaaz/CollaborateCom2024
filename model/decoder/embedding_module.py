# embedding_module-transformer+hawkes.py
# 将用户、时间等信息嵌入到一个低维度的向量空间中
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.modules.module
from typing import Dict, Mapping, List, Union, Tuple
import dgl
from model.encoder.state.dynamic_state import DynamicState
from model.decoder.structural_aggregator import TreeAggregator
from model.decoder.transformer_block import TransformerBlock
from model.time_encoder import TimeSlotEncoder
from utils.hgraph import HGraph
#from model.hawkes import *

class EmbeddingModule(nn.Module):
    def __init__(self, dynamic_state: Mapping[str, DynamicState], embedding_dimension: int, device: torch.device,
                 dropout: float, hgraph: HGraph):
        super(EmbeddingModule, self).__init__()
        self.dynamic_state = dynamic_state
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.hgraph = hgraph

    # 计算给定级联的嵌入
    def compute_embedding(self, cascades: np.ndarray) -> \
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the embedding of given cascades by different methods
        :param cascades: the ids of cascades, ndarray of shape (n_node)
        :return: the embedding of the corresponding cascades,tensor of shape (n_node,emb_dim)
        """
        ...


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, cascade):
        """return the dynamic states of cascades"""
        return self.dynamic_state['cas'].get_state(cascade, from_cache=True) # 返回级联的动态状态


class ConcatEmbedding(EmbeddingModule):
    def __init__(self, dynamic_state: Mapping[str, DynamicState], embedding_dimension: int, device: torch.device,
                 dropout: float, hgraph: HGraph, max_global_time: float = 0.0, global_time_num: int = 0):
        super(ConcatEmbedding, self).__init__(dynamic_state, embedding_dimension, device,
                                              dropout, hgraph)
        self.time_embedding = TimeSlotEncoder(embedding_dimension, max_global_time, global_time_num)

    def compute_embedding(self, cascades):  # 将级联的动态状态、发送用户和接收用户的嵌入以及级联的嵌入连接在一起
        """concat the dynamic states of the sending user, receiving user and cascade in the last interaction"""
        cas_interaction = self.hgraph.batch_cas_info()
        cas_pub_times = torch.tensor(self.hgraph.get_cas_pub_time(cascades), dtype=torch.float, device=self.device)
        src_embs, dst_embs = [], []
        for cas in cascades:
            src, dst = zip(*cas_interaction[cas])
            src_emb = self.dynamic_state['user'].get_state(src, 'src', from_cache=True).squeeze(dim=0)
            dst_emb = self.dynamic_state['user'].get_state(dst, 'dst', from_cache=True).squeeze(dim=0)
            if len(src) > 1:
                src_emb = src_emb[-1]
                dst_emb = dst_emb[-1]
            src_embs.append(src_emb)
            dst_embs.append(dst_emb)
        cas_embs = self.dynamic_state['cas'].get_state(cascades, from_cache=True)
        cas_embs += self.time_embedding(cas_pub_times)
        src_embs = torch.stack(src_embs, dim=0)
        dst_embs = torch.stack(dst_embs, dim=0)
        return torch.cat([src_embs, dst_embs, cas_embs], dim=1)


class AggregateEmbedding(EmbeddingModule): # 根据配置决定是否使用静态嵌入、动态嵌入、结构嵌入和/或时间嵌入等方法
    def __init__(self, dynamic_state: Mapping[str, DynamicState], input_dimension: int, embedding_dimension: int,
                 device: torch.device, dropout: float, hgraph: HGraph, max_time: float, time_num: int,
                 use_static: bool, user_num: int, max_global_time: float, global_time_num: int, use_dynamic: bool,
                 use_temporal: bool, use_structural: bool):
        super(AggregateEmbedding, self).__init__(dynamic_state, embedding_dimension, device, dropout, hgraph)
        self.use_dynamic = use_dynamic
        self.use_temporal = use_temporal
        self.use_structural = use_structural
        self.use_static = use_static
        self.dropout = nn.Dropout(dropout)
        self.time_position_encoder = TimeSlotEncoder(embedding_dimension, max_time, time_num) # 时间编码器
        self.position_embedding = nn.Embedding(100, embedding_dimension) # 位置编码器
        # self.norm_static = nn.LayerNorm(input_dimension) # if is_layer_norm else nn.Identity()
        # self.norm_dynamic = nn.LayerNorm(input_dimension) # if is_layer_norm else nn.Identity()
        #self.hawkes_module = Hawkes(d_model=64, num_types=1)
        if self.use_dynamic:
            dynamic_trans_input_dim = 3 * input_dimension
            self.concat_emb = ConcatEmbedding(dynamic_state, embedding_dimension, device, dropout, hgraph,
                                              max_global_time, global_time_num)
            if use_temporal:
                dynamic_trans_input_dim += input_dimension
                # self.temporal_aggregator = nn.LSTM(input_size=input_dimension, hidden_size=embedding_dimension, batch_first=True)
                self.temporal_aggregator = TransformerBlock(input_size=input_dimension, n_heads=8, is_layer_norm=True, attn_dropout=dropout, device=device)
            if use_structural:
                dynamic_trans_input_dim += 2 * input_dimension
                self.structural_agg_root = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
                self.structural_agg_leaf = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
            self.trans = nn.Sequential(nn.Linear(in_features=dynamic_trans_input_dim, out_features=embedding_dimension),
                                       nn.ReLU())
        if self.use_static:
            static_trans_input_dim = 0
            self.static_state = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dimension)
            nn.init.uniform_(self.static_state.weight, 0, 1)
            if use_temporal:
                # self.static_rnn = nn.LSTM(input_size=input_dimension, hidden_size=embedding_dimension, batch_first=True)
                self.temp_attention = TransformerBlock(input_size=input_dimension, n_heads=8, is_layer_norm=True, attn_dropout=dropout, device=device)
                static_trans_input_dim += input_dimension

            if use_structural:
                self.static_root_gnn = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
                self.static_leaf_gnn = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
                static_trans_input_dim += 2 * input_dimension
            self.static_trans = nn.Sequential(
                nn.Linear(in_features=static_trans_input_dim, out_features=embedding_dimension),
                nn.ReLU())

    # 计算静态嵌入
    def compute_static_emb(self, cascades: np.ndarray, cas_history: torch.Tensor, cas_times: torch.Tensor, length: List[int],
                           cas_time_emb: torch.Tensor, cas_pos_emb: torch.Tensor, graph_root: dgl.DGLHeteroGraph,
                           graph_leaf: dgl.DGLHeteroGraph) -> torch.Tensor:
        final_embedding = []
        if self.use_temporal:
            cas_his_emb = self.dropout(self.static_state(cas_history.reshape(-1)).reshape(*cas_history.shape, -1))
            cas_his_emb = cas_his_emb + cas_time_emb + cas_pos_emb
            cas_his_emb = pack_padded_sequence(cas_his_emb, lengths=length, batch_first=True, enforce_sorted=False)
            # _, (h, c) = self.static_rnn.forward(cas_his_emb)
            cas_his_emb, _ = pad_packed_sequence(cas_his_emb, batch_first=True)  # Convert PackedSequence to tensor
            # cas_his_emb = self.norm_static(cas_his_emb)  # 应用层归一化
            h, event_ll, non_event_ll = self.temp_attention.forward(Q=cas_his_emb, K=cas_his_emb, V=cas_his_emb, hawkes_time=cas_times)
            # cas_his_emb = cas_his_emb.unsqueeze(0).expand(len(cascades), -1, -1)
            # cas_his_emb = cas_his_emb.unsqueeze(0).repeat(len(cascades), 1, 1)
            # cas_his_emb = cas_his_emb.unsqueeze(0).repeat_interleave(len(cascades), dim=0)
            # final_embedding.append(h.squeeze(dim=0))
            final_embedding.append(h)
            # print("Sizes of tensors in final_embedding - static - temporal:")
            # for i, tensor in enumerate(final_embedding):
            #     print(f"Tensor {i}: {tensor.size()}")
        # if self.use_temporal:
            # # Local temporal learning
            # global_time = self.local_time_embedding(timestamps[data_idx])
            # att_hidden = adj_with_fea + global_time
            # att_out_tem = self.temp_attention(att_hidden, att_hidden, att_hidden, mask = att_mask )
            # news_out_tem = torch.einsum("abc,ab->ac", (att_out_tem, nor_input))
            # # Concatenate temporal propagation status
            # news_out_tem = torch.cat([news_out_tem, spread_status[data_idx][:, 2:]/3600/24], dim=-1)
            # news_out_tem = news_out_tem.matmul(self.weight)
        if self.use_structural:
            root_feature = self.static_state(graph_root.ndata['id'])
            leaf_feature = self.static_state(graph_leaf.ndata['id'])
            graph_root.ndata['x'] = self.dropout(root_feature)
            graph_leaf.ndata['x'] = self.dropout(leaf_feature)
            root_emb, leaf_emb = self.static_root_gnn(graph_root), self.static_leaf_gnn(graph_leaf)
            final_embedding.extend([root_emb, leaf_emb])
            # print("Sizes of tensors in final_embedding - static - structural:")
            # for i, tensor in enumerate(final_embedding):
            #     print(f"Tensor {i}: {tensor.size()}")
        # print("Shape of concatenated static_final_embedding:", torch.cat(final_embedding, dim=1).shape)
        return self.static_trans(torch.cat(final_embedding, dim=1)), event_ll, non_event_ll

    # 计算动态嵌入
    def compute_dynamic_emb(self, cascades: np.ndarray, cas_history: torch.Tensor, cas_times: torch.Tensor, length: List[int],
                            cas_time_emb: torch.Tensor, cas_pos_emb: torch.Tensor, graph_root: dgl.DGLHeteroGraph,
                            graph_leaf: dgl.DGLHeteroGraph) -> torch.Tensor:
        newest_dynamic_state = self.concat_emb.compute_embedding(cascades)
        final_embedding = [newest_dynamic_state]
        if self.use_temporal:
            cas_his_emb = self.dynamic_state['user'].get_state(cas_history.reshape(-1), 'src', from_cache=False). \
                reshape(*cas_history.shape, -1)
            cas_his_emb = cas_his_emb + cas_time_emb + cas_pos_emb
            cas_his_emb = pack_padded_sequence(cas_his_emb, lengths=length, batch_first=True, enforce_sorted=False)
            # _, (h, c) = self.temporal_aggregator.forward(cas_his_emb)
            cas_his_emb, _ = pad_packed_sequence(cas_his_emb, batch_first=True)  # Convert PackedSequence to tensor
            # cas_his_emb = self.norm_dynamic(cas_his_emb)  # 应用层归一化
            h, event_ll, non_event_ll = self.temporal_aggregator.forward(Q=cas_his_emb, K=cas_his_emb, V=cas_his_emb, hawkes_time=cas_times)
            # cas_his_emb = cas_his_emb.unsqueeze(0).expand(len(cascades), -1, -1)
            # final_embedding.append(h.squeeze(dim=0))
            # final_embedding.append(h.reshape(1, -1))
            # final_embedding.append(h.view(1, -1))
            final_embedding.append(h)
            # print("Sizes of tensors in final_embedding - dynamic - temporal:")
            # for i, tensor in enumerate(final_embedding):
            #     print(f"Tensor {i}: {tensor.size()}")
        # if self.use_temporal:
            # # Local temporal learning
            # global_time = self.local_time_embedding(timestamps[data_idx])
            # att_hidden = adj_with_fea + global_time
            # att_out_tem = self.temp_attention(att_hidden, att_hidden, att_hidden, mask = att_mask )
            # news_out_tem = torch.einsum("abc,ab->ac", (att_out_tem, nor_input))
            # # Concatenate temporal propagation status
            # news_out_tem = torch.cat([news_out_tem, spread_status[data_idx][:, 2:]/3600/24], dim=-1)
            # news_out_tem = news_out_tem.matmul(self.weight)
        if self.use_structural:
            graph_root.ndata['x'] = self.dynamic_state['user'].get_state(graph_root.ndata['id'], 'dst', from_cache=False)
            graph_leaf.ndata['x'] = self.dynamic_state['user'].get_state(graph_leaf.ndata['id'], 'src', from_cache=False)
            root_emb, leaf_emb = self.structural_agg_root(graph_root), self.structural_agg_leaf(graph_leaf)
            final_embedding.extend([root_emb, leaf_emb])
            # final_embedding.extend([root_emb.unsqueeze(dim=1), leaf_emb].unsqueeze(dim=1))
            # print("Sizes of tensors in final_embedding - dynamic - structural:")
            # for i, tensor in enumerate(final_embedding):
            #     print(f"Tensor {i}: {tensor.size()}")
        # 在 torch.cat 前打印每个张量的大小
        # print("Sizes of tensors in final_embedding:")
        # for i, tensor in enumerate(final_embedding):
        #     print(f"Tensor {i}: {tensor.size()}")
        # 进行 torch.cat 操作
        # print("Shape of concatenated dynamic_final_embedding:", torch.cat(final_embedding, dim=1).shape)
        return self.trans(torch.cat(final_embedding, dim=1)), event_ll, non_event_ll

    # 拼接静态嵌入和动态嵌入，生成级联的最终嵌入表示
    def compute_embedding(self, cascades):
        """aggregate the representations of users into a cascade embedding by
        structural learning and temporal learning"""
        # concat_state
        cas_history, cas_times, length = self.hgraph.get_cas_seq(cascades)
        cas_history = pad_sequence(cas_history, batch_first=True).to(self.device)
        """set all position&time emb"""
        cas_times = pad_sequence(cas_times, batch_first=True).to(self.device)
        cas_time_emb = self.time_position_encoder(cas_times.reshape(-1)).reshape(*cas_times.shape, -1)
        single_range = list(range(cas_times.shape[1]))
        cas_pos = torch.tensor([single_range] * cas_times.shape[0]).to(self.device)
        cas_pos_emb = self.position_embedding(cas_pos)
        """set graph feature"""
        graph_root, graph_leaf = None, None
        if self.use_structural:
            graph_root, graph_leaf = self.hgraph.get_cas_graph(cascades)
            graph_root, graph_leaf = graph_root.to(self.device), graph_leaf.to(self.device)
            graph_root.edata['time'] = self.time_position_encoder(graph_root.edata['time'])
            graph_leaf.edata['time'] = self.time_position_encoder(graph_leaf.edata['time'])
        static_emb = self.compute_static_emb(cascades, cas_history, cas_times, length, cas_time_emb, cas_pos_emb, graph_root, graph_leaf)
        dynamic_emb = self.compute_dynamic_emb(cascades, cas_history, cas_times, length, cas_time_emb, cas_pos_emb, graph_root, graph_leaf)
        # print("cas_history.shape:", cas_history.shape)
        # print("cas_times.shape:", cas_times.shape)
        # # 假设 max_length 是填充到的长度
        # max_length = 64
        #
        # # 处理 cas_history 序列
        # for i in range(len(cas_history)):
        #     if cas_history[i].size(0) < max_length:
        #         padding_size = max_length - cas_history[i].size(0)
        #         padding_tensor = torch.zeros(padding_size, dtype=cas_history[i].dtype)
        #         padding_tensor = padding_tensor.to(cas_history[i].device)
        #         cas_history[i] = torch.cat((cas_history[i], padding_tensor), dim=0)
        #     else:
        #         cas_history[i] = cas_history[i][:max_length]
        #
        # # 处理 cas_times 序列
        # for i in range(len(cas_times)):
        #     if cas_times[i].size(0) < max_length:
        #         padding_size = max_length - cas_times[i].size(0)
        #         padding_tensor = torch.zeros(padding_size, dtype=cas_times[i].dtype)
        #         padding_tensor = padding_tensor.to(cas_times[i].device)
        #         cas_times[i] = torch.cat((cas_times[i], padding_tensor), dim=0)
        #     else:
        #         cas_times[i] = cas_times[i][:max_length]
        #
        # # 现在 cas_history 和 cas_times 序列的长度已经达到了 max_length
        # # 可以调用 pad_sequence 函数进行填充
        # padded_cas_history = torch.nn.utils.rnn.pad_sequence(cas_history, batch_first=True, padding_value=0)
        # padded_cas_times = torch.nn.utils.rnn.pad_sequence(cas_times, batch_first=True, padding_value=0)
        #
        # event_ll, non_event_ll = log_likelihood(model=self.hawkes_module, data=padded_cas_history.float(), time=padded_cas_times.float())
        # print("event_ll.shape:", event_ll.shape)
        # print("non_event_ll.shape:", non_event_ll.shape)
        # print("Type of event_ll:", type(event_ll))
        # print("Type of non_event_ll:", type(non_event_ll))
        if self.use_dynamic:
            dynamic_emb, event_ll, non_event_ll = self.compute_dynamic_emb(cascades, cas_history, cas_times, length, cas_time_emb, cas_pos_emb, graph_root, graph_leaf)
        if self.use_static:
            static_emb, event_ll, non_event_ll = self.compute_static_emb(cascades, cas_history, cas_times, length, cas_time_emb, cas_pos_emb, graph_root, graph_leaf)
        if self.use_static and self.use_dynamic:
            return static_emb, dynamic_emb, event_ll, non_event_ll
        elif self.use_static:
            return static_emb, static_emb, event_ll, non_event_ll
        elif self.use_dynamic:
            return dynamic_emb, dynamic_emb, event_ll, non_event_ll


def get_embedding_module(module_type: str, dynamic_state: Mapping[str, DynamicState],
                         input_dimension: int, embedding_dimension: int, device: torch.device,
                         dropout: float = 0.1, hgraph: HGraph = None, max_time: float = 1.0,
                         time_num: int = 20, use_static: bool = False, user_num: int = -1,
                         max_global_time: float = 100.0, global_time_num: int = 20,
                         use_dynamic: bool = True, use_temporal: bool = True,
                         use_structural: bool = True) -> EmbeddingModule:
    if module_type == "identity":
        return IdentityEmbedding(dynamic_state=dynamic_state,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout,
                                 hgraph=hgraph)
    elif module_type == 'aggregate': # 默认
        return AggregateEmbedding(dynamic_state=dynamic_state,
                                  embedding_dimension=embedding_dimension,
                                  device=device,
                                  dropout=dropout,
                                  hgraph=hgraph,
                                  input_dimension=input_dimension,
                                  max_time=max_time,
                                  time_num=time_num,
                                  use_static=use_static, user_num=user_num, max_global_time=max_global_time,
                                  global_time_num=global_time_num, use_dynamic=use_dynamic, use_temporal=use_temporal,
                                  use_structural=use_structural)
    elif module_type == 'concat':
        return ConcatEmbedding(dynamic_state=dynamic_state, embedding_dimension=embedding_dimension,
                               device=device, dropout=dropout, hgraph=hgraph,
                               max_global_time=max_global_time,
                               global_time_num=global_time_num)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
