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
        super(MainModel, self).__init__()
        if max_time is None:
            max_time = {'user': 1, 'cas': 1}
        self.ntypes = ntypes
        self.device = device
        self.predictor_type=predictor
        self.cas_num = n_nodes['cas']
        self.user_num = n_nodes['user']
        self.single = single
        self.hgraph = HGraph(num_user=n_nodes['user'], num_cas=n_nodes['cas'])
        self.time_encoder = get_time_encoder('difference', dimension=time_enc_dim, single=self.single)
        self.use_dynamic = use_dynamic
        self.use_static = use_static
        self.dynamic_state = nn.ModuleDict({
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

    def update_state(self):
        if self.use_dynamic:
            for ntype in self.ntypes:
                self.dynamic_state[ntype].store_cache()

    def forward(self, source_nodes: np.ndarray, destination_nodes: np.ndarray, trans_cascades: np.ndarray,
                edge_times: torch.Tensor, pub_times: torch.Tensor, target_idx: np.ndarray) -> torch.Tensor:
        if self.use_dynamic:
            nodes, messages, times = self.message_generator.get_message(source_nodes, destination_nodes,
                                                                        trans_cascades, edge_times, pub_times, 'all')
            self.state_updater.update_state(nodes, messages, times)
        self.hgraph.insert(trans_cascades, source_nodes, destination_nodes, edge_times, pub_times)
        target_cascades = trans_cascades[target_idx]
        pred = torch.zeros(len(trans_cascades)).to(self.device)
        event_ll = torch.zeros(len(trans_cascades)).to(self.device)
        non_event_ll = torch.zeros(len(trans_cascades)).to(self.device)
        if len(target_cascades) > 0:
            if self.use_static and self.use_dynamic:
                emb1, emb2, event_ll[target_idx], non_event_ll[target_idx] = self.embedding_module.compute_embedding(target_cascades)
                if self.predictor_type == "fusion":
                    pred[target_idx] = self.predictor.forward(emb1, emb2)
                else:
                    emb = (emb1, emb2)
                    pred[target_idx] = self.predictor.forward(emb)
            else:
                emb, _ = self.embedding_module.compute_embedding(target_cascades)
                pred[target_idx] = self.predictor.forward(emb)
        return pred, event_ll, non_event_ll

    def init_state(self):
        for ntype in self.ntypes:
            self.dynamic_state[ntype].__init_state__()
        self.hgraph.init()

    def reset_state(self):
        for ntype in self.ntypes:
            self.dynamic_state[ntype].reset_state()
        self.hgraph.init()

    def detach_state(self):
        for ntype in self.ntypes:
            self.dynamic_state[ntype].detach_state()
