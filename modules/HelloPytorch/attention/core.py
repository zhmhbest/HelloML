import math
import torch
from torch import nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F
from .base import clones


def attention(
        q: Tensor, k: Tensor, v: Tensor,
        mask: BoolTensor = None, dropout: nn.Dropout = None
):
    """
    Compute 'Scaled Dot Product Attention'
    :param q: [batch_size, time_step, feature_size, ...]
    :param k: [batch_size, time_step, feature_size, ...]
    :param v: [batch_size, time_step, feature_size, ...]
    :param mask:
        tensor([[ True, False],
                [ True, True]])
    :param dropout:
    :return:
    """
    # d_k = query.size(-1)
    d_k = q.shape[-1]
    k_t = k.transpose(-2, -1)
    scores = torch.matmul(q, k_t) / math.sqrt(d_k)
    if mask is not None:
        # Fills elements of tensor with value where mask is True.
        scores = torch.masked_fill(scores, mask, -1e9)
    result = F.softmax(scores, dim=-1)
    if dropout is not None:
        result = dropout(result)
    return torch.matmul(result, v), result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.qkv_linears = clones(nn.Linear(d_model, d_model), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.qkv_linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.final_linear(x)


class PositionWiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
