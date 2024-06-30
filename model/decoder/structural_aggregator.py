import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.udf import EdgeBatch, NodeBatch
import dgl


# class Attention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Attention, self).__init__()
#         self.W = nn.Linear(input_size, hidden_size)
#         self.u = nn.Linear(hidden_size, 1)

#     def forward(self, nodes):
#         scores = self.u(torch.tanh(self.W(nodes)))  # Compute scores
#         weights = F.softmax(scores, dim=1)  # Apply softmax to get attention weights
#         return weights

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        # self.W = nn.Linear(input_size * 2, hidden_size)  # Combine input_size and hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.u = nn.Linear(2 * hidden_size, 1)

    def forward(self, nodes):
        # print("Nodes data 'h':", nodes.data['h'])
        # print("Mailbox data 'h':", nodes.mailbox['h'])
        # print("Nodes data 'h' shape:", nodes.data['h'].shape)
        # print("Mailbox data 'h' shape:", nodes.mailbox['h'].shape)
        # print("Nodes data 'x' shape:", nodes.data['x'].shape)

        # combined_features = torch.cat((nodes.data['x'], nodes.mailbox['h']), dim=1)  # Combine node features and neighbor features
        # combined_features = torch.cat((nodes.data['x'].unsqueeze(1), nodes.mailbox['h']), dim=1)  # 在第二个维度上添加一个维度
        # print("self.W(nodes.mailbox['h']).shape:", self.W(nodes.mailbox['h']).shape)
        # print("self.W(nodes.data['x'].unsqueeze(1)).shape:", self.W(nodes.data['x'].unsqueeze(1)).shape)
        combined_features = torch.cat((self.W(nodes.mailbox['h']), self.W(nodes.data['x'].unsqueeze(1)).expand(-1, nodes.mailbox['h'].size(1), -1)), dim=2)
        scores = F.leaky_relu(self.u(combined_features))  # Compute scores
        weights = F.softmax(scores, dim=1)  # Apply softmax to get attention weights
        # print("combined_features.shape:", combined_features.shape)
        # print("scores.shape:", scores.shape)
        # print("weights.shape:", weights.shape)
        return weights


# class Attention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Attention, self).__init__()
#         self.W = nn.Linear(input_size * 2, hidden_size)  # 输入大小变为目标节点特征和邻居节点特征的拼接
#         self.u = nn.Linear(hidden_size, 1)

#     def forward(self, target_node, neighbor_nodes):
#         # 将目标节点特征和邻居节点特征拼接起来
#         combined = torch.cat([target_node, neighbor_nodes], dim=1)
#         scores = self.u(torch.tanh(self.W(combined)))  # 计算分数
#         weights = F.softmax(scores, dim=1)  # 应用 softmax 获取注意力权重
#         return weights

# class MutualAttention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(MutualAttention, self).__init__()
#         self.W_query = nn.Linear(input_size, hidden_size)
#         self.W_key = nn.Linear(input_size, hidden_size)
#         self.W_value = nn.Linear(input_size, hidden_size)
#         self.u = nn.Linear(hidden_size, 1)

#     def forward(self, queries, keys):
#         query = self.W_query(queries)
#         key = self.W_key(keys)
#         values = self.W_value(keys)  # Assume the same keys as values

#         # Calculate attention scores
#         scores = self.u(torch.tanh(query + key))
#         weights = F.softmax(scores, dim=1)

#         # Apply attention weights to values
#         attended_values = torch.sum(weights * values, dim=1)
#         return attended_values


class TreeAggregatorCell(nn.Module):
    def __init__(self, x_size: int, h_size: int, edge_time: float):
        super(TreeAggregatorCell, self).__init__()
        self.edge_time = edge_time
        self.attention = Attention(x_size, h_size)  # Attention mechanism
        # self.attention = MutualAttention(x_size, h_size)  # 使用互注意力机制
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
        # attention_weights = self.attention(nodes.mailbox['h'])  # Calculate attention weights
        attention_weights = self.attention(nodes)
        # print("nodes.mailbox['h'].shape:", nodes.mailbox['h'].shape)
        # print("attention_weights.shape:", attention_weights.shape)
        weighted_h_tilde = nodes.mailbox['h'] * attention_weights  # Weighted sum of h_tilde
        # attention_weights_expanded = attention_weights.expand(-1, nodes.mailbox['h'].size(1), -1)
        # weighted_h_tilde = nodes.mailbox['h'] * attention_weights_expanded
        # print("weighted_h_tilde.shape:", weighted_h_tilde.shape)
        h_tilde = torch.sum(weighted_h_tilde, 1)  # Aggregate h_tilde using attention weights
        return {'h_tilde': h_tilde, 'c': c}

    # def reduce_func(self, nodes):
    #     if self.edge_time:
    #         nodes.mailbox['h'] += nodes.mailbox['time']
    #     target_node_features = nodes.data['h_tilde'].unsqueeze(1).expand(-1, nodes.mailbox['h'].size(1), -1)
    #     attention_weights = self.attention(target_node_features, nodes.mailbox['h'])  # 计算注意力权重
    #     weighted_h_tilde = nodes.mailbox['h'] * attention_weights  # 加权求和得到加权后的 h_tilde
    #     h_tilde = torch.sum(weighted_h_tilde, 1)  # Aggregate h_tilde using attention weights
    #     c = torch.sum(nodes.mailbox['c'], 1)
    #     return {'h_tilde': h_tilde, 'c': c}

    # def reduce_func(self, nodes):
    #     if self.edge_time:
    #         nodes.mailbox['h'] += nodes.mailbox['time']
    #     # 使用互注意力机制计算注意力权重，将目标节点的特征 'x' 作为查询向量，邻居节点的特征 'h' 作为键和值
    #     attention_weights = self.attention(nodes.data['x'].unsqueeze(1), nodes.mailbox['h'])
    #     # 应用注意力权重到邻居节点的特征上
    #     print("nodes.mailbox['h'] shape:", nodes.mailbox['h'].shape)
    #     print("attention_weights.unsqueeze(2) shape:", attention_weights.unsqueeze(2).shape)
    #     attended_messages = nodes.mailbox['h'] * attention_weights.unsqueeze(2)
    #     h_tilde = torch.sum(attended_messages, 1)  # 使用注意力权重对 h_tilde 进行聚合
    #     c = torch.sum(nodes.mailbox['c'], 1)
    #     return {'h_tilde': h_tilde, 'c': c}

    def apply_node_func(self, nodes):
        iou = self.U_iou(nodes.data['h_tilde']) + self.W_iou(nodes.data['x']) + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class TreeAggregator(nn.Module):
    def __init__(self, x_size: int, h_size: int, device: torch.device = None, dropout: float = 0.1,
                 edge_time: bool = False):
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
