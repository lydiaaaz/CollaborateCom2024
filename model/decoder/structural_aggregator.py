import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.udf import EdgeBatch, NodeBatch
import dgl

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size)
        self.u = nn.Linear(2 * hidden_size, 1)

    def forward(self, nodes):
        combined_features = torch.cat((self.W(nodes.mailbox['h']), self.W(nodes.data['x'].unsqueeze(1)).expand(-1, nodes.mailbox['h'].size(1), -1)), dim=2)
        scores = F.leaky_relu(self.u(combined_features))
        weights = F.softmax(scores, dim=1)
        return weights

class TreeAggregatorCell(nn.Module):
    def __init__(self, x_size: int, h_size: int, edge_time: float):
        super(TreeAggregatorCell, self).__init__()
        self.edge_time = edge_time
        self.attention = Attention(x_size, h_size)
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))

    def message_func(self, edges):
        if self.edge_time:
            return {'h': edges.src['h'], 'c': edges.src['c'], 'time': edges.data['time']}
        else:
            return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        if self.edge_time:
            nodes.mailbox['h'] += nodes.mailbox['time']
        f = th.sigmoid(self.U_f(nodes.mailbox['h']) + self.W_f(nodes.data['x']).unsqueeze(dim=1) + self.b_f)
        c = th.sum(f * nodes.mailbox['c'], 1)
        attention_weights = self.attention(nodes)
        weighted_h_tilde = nodes.mailbox['h'] * attention_weights
        h_tilde = torch.sum(weighted_h_tilde, 1)
        return {'h_tilde': h_tilde, 'c': c}

    def apply_node_func(self, nodes):
        iou = self.U_iou(nodes.data['h_tilde']) + self.W_iou(nodes.data['x']) + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

class TreeAggregator(nn.Module):
    def __init__(self, x_size: int, h_size: int, device: torch.device = None, dropout: float = 0.1, edge_time: bool = False):
        super(TreeAggregator, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        cell = TreeAggregatorCell
        self.cell = cell(x_size, h_size, edge_time)

    def forward(self, g):
        g.ndata['h'] = torch.zeros(g.num_nodes(), self.h_size).to(self.device)
        g.ndata['c'] = torch.zeros(g.num_nodes(), self.h_size).to(self.device)
        g.ndata['h_tilde'] = torch.zeros(g.num_nodes(), self.h_size).to(self.device)
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        h = dgl.readout_nodes(g, 'h', 'mask') / dgl.readout_nodes(g, 'mask')
        return self.dropout(h)
