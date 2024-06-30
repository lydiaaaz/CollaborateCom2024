import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Union


class Linear(nn.Module):
    def __init__(self, emb_dim: int):
        super(Linear, self).__init__()
        self.lins = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                  nn.ReLU(), nn.Linear(emb_dim // 2, 1))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Map the cascade embeddings into popularity counts
        :param emb: tensor of shape (batch,emb_dim)
        :return: tensor of shape (batch)
        """
        return torch.squeeze(self.lins(emb), dim=1)


class MergeLinear(nn.Module):
    def __init__(self, emb_dim: int, prob: float):
        super(MergeLinear, self).__init__()
        self.prob = nn.Parameter(torch.tensor(prob), requires_grad=False)
        self.dynamic_fn = Linear(emb_dim)
        self.static_fn = Linear(emb_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Map the cascade embeddings into popularity counts
        :param emb: tensor of shape (batch,emb_dim)
        :return: tensor of shape (batch)
        """
        static_emb, dynamic_emb = emb
        pred = self.prob * self.static_fn(static_emb) + (1 - self.prob) * self.dynamic_fn(dynamic_emb)
        return pred


class GatedFusion(nn.Module):
    def __init__(self, input_size, out_size=64, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.linear3 = Linear(input_size)

    def forward(self, X1, X2):
        emb = torch.cat([X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)], dim=0)
        # emb = torch.stack([X1, X2], dim=0)  # 沿着第一个维度将 X1 和 X2 拼接起来
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        # out = F.relu(self.linear3(out))
        out = self.linear3(out)
        return out


def get_predictor(emb_dim: int, predictor_type: str = 'linear', merge_prob: float = 0.5) -> Union[Linear, MergeLinear]:
    if predictor_type == 'linear':
        return Linear(emb_dim)
    elif predictor_type == 'merge':
        return MergeLinear(emb_dim, merge_prob)
    elif predictor_type == 'fusion':
        return GatedFusion(emb_dim)
    else:
        raise ValueError('Not implemented predictor type!')
