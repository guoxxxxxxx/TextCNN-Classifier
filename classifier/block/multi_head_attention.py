import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, is_casual=False, max_seq_len=2048):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)

        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)

        self.is_casual = is_casual

    def forward(self, q, k, v):
        bsz, _, seq_len, embed_dim = q.size()
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 掩码子注意力额外部分
        if self.is_casual :
            if not hasattr(self, 'mask'):
                mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("inf"))
                mask = torch.triu(mask, diagonal=1)
                self.register_buffer('mask', mask)
            scores += self.mask[:, :, :seq_len, :seq_len]

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, 1, seq_len, -1)
        output = self.wo(output)
        output = self.res_dropout(output)
        return output
