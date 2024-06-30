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

        # Compute the positional encodings once in log space.
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

        self.pos_encoding= PositionalEncoding(d_model=input_size, dropout=0.5)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_v))

        self.W_o = nn.Parameter(torch.Tensor(self.d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        self.hawkes_module = Hawkes(d_model=64, num_types=1)

    def __init_weights__(self):
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

    # 缩放点积注意力机制
    def scaled_dot_product_attention(self, Q, K, V, mask, epsilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param mask: (*, max_q_words)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5

        Q_K = (torch.einsum("bqd,bkd->bqk", Q, K)) / (temperature + epsilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(self.device)
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)
        #维度为3的两个矩阵的乘法
        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        '''
        :param Q:
        :param K:
        :param V:
        :param mask: (bsz, max_q_words)
        :return:
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        #print(self.W_q.size(), bsz, q_len, self.n_heads, self.d_k)
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)
        #print(Q_.size(), bsz, q_len, self.n_heads, self.d_k)
        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)
        

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V, hawkes_time, mask=None, pos = False):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        if pos:
            Q = self.pos_encoding(Q)
            K = self.pos_encoding(K)
            V = self.pos_encoding(V)

        V_att = self.multi_head_attention(Q, K, V, mask)

        # event_ll, non_event_ll = log_likelihood(model=self.hawkes_module, data=V_att, time=hawkes_time)

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        # 打印 output 的维度
        # print("output size:", output.size())

        event_ll, non_event_ll = log_likelihood(model=self.hawkes_module, data=output, time=hawkes_time)

        # 添加平均池化操作，减少输出维度，将输出的维度从 [batch_size, max_q_words, input_size] 缩减为 [batch_size, input_size]
        output = F.avg_pool1d(output.permute(0, 2, 1), kernel_size=output.size(1)).squeeze(dim=-1)
        # output = torch.mean(output, dim=1)  # 沿着 max_q_words 维度取平均

        # 最大池化
        # output = F.max_pool1d(output.permute(0, 2, 1), kernel_size=output.size(1)).squeeze(dim=-1)

        # 使用自注意力机制/加权平均池化
        # 1. 计算注意力权重
        # attention_weights = F.softmax(self.dropout(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)), dim=-1)
        # attention_weights = self.scaled_dot_product_attention(output, output, output, mask)
        #attention_weights = F.softmax(torch.matmul(output, output.transpose(1, 2)) / math.sqrt(self.d_k), dim=-1)
        # 2. 加权求和得到聚合结果
        #weighted_output = torch.matmul(attention_weights, output)
        # 3. 沿着 max_q_words 维度求和
        #output = torch.sum(weighted_output, dim=1)

        return output, event_ll, non_event_ll

        # 除了平均池化，还有其他操作可以减少输出的维度。以下是一些可能的操作：
        # 1.最大池化（Max Pooling）：取输入中每个通道的最大值，从而减少输出的维度。
        # 2.全局平均池化（Global Average Pooling）：取输入中每个通道的平均值，产生一个值作为每个通道的输出。
        # 3.特征选择（Feature Selection）：选择最相关或最重要的特征，丢弃其他特征。
        # 4.降维操作：如PCA（主成分分析）、t - SNE等，通过线性或非线性变换将特征降至较低的维度。
        # 5.自注意力机制（Self - Attention）：使用注意力机制对输入的不同部分进行加权，然后将加权后的结果加和，得到每个通道的输出。
        # 6.卷积操作（Convolution）：通过卷积操作从输入中提取特征，然后使用池化操作减少输出的维度。
        # 7.信息聚合（Information Aggregation）：将输入的信息进行聚合，生成一个固定长度的表示。
        # 这些操作可以根据具体的情况和需求进行选择和组合，以便得到适合任务要求的输出。
