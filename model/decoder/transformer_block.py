# transformer_block.py
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import math
from model.hawkes import *


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=800):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):

    def __init__(self, input_size, n_heads=2, is_layer_norm=True, attn_dropout=0.1, device: torch.device = None):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = input_size
        self.d_v = input_size
        self.device = device

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_v))

        self.W_o = nn.Parameter(torch.Tensor(self.d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        self.hawkes_module = Hawkes(d_model=64, num_types=1)

    def __init_weights(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, epsilon=1e-6):
        temperature = self.d_k ** 0.5

        Q_K = (torch.einsum("bqd,bkd->bqk", Q, K)) / (temperature + epsilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(self.device)
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32 + 1)

        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output


    def forward(self, Q, K, V, hawkes_time, mask=None, pos=False):
        if pos:
            Q = self.pos_encoding(Q)
            K = self.pos_encoding(K)
            V = self.pos_encoding(V)

        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        event_ll, non_event_ll = log_likelihood(model=self.hawkes_module, data=output, time=hawkes_time)

        output = F.avg_pool1d(output.permute(0, 2, 1), kernel_size=output.size(1)).squeeze(dim=-1)

        return output, event_ll, non_event_ll
