from typing import Union
from attention.common_header import *
from attention.base import duplicate_module, Dense, subsequent_mask
from attention.architecture import AttentionInterface, FeedForwardInterface


def attention(
        q: Tensor, k: Tensor, v: Tensor,
        mask: Union[BoolTensor, Tensor] = None,
        dropout_module: nn.Dropout = None
):
    """
    Compute 'Scaled Dot Product Attention'
    :param q: [batch_size, time_step, feature_size] | [batch_size, h, time_step, d_k]
    :param k:
    :param v:
    :param mask: [1, time_step, time_step] | [1, 1, time_step, time_step]
    :param dropout_module:
    :return:
    """
    d_k = q.shape[-1]
    k_t = k.transpose(-2, -1)  # Transpose the last two dimensions
    scores = torch.matmul(q, k_t) / math.sqrt(d_k)
    # [batch_size, time_step, time_step] | [batch_size, h, time_step, time_step]

    if mask is not None:
        # Fills elements of tensor with value where mask is True.
        scores = torch.masked_fill(scores, mask, -1e9)
    parm = F.softmax(scores, dim=-1)
    if dropout_module is not None:
        parm = dropout_module(parm)
    return torch.matmul(scores, v), parm


if __name__ == '__main__':
    print("Test attention")
    import matplotlib.pyplot as plt
    _batch_size = 1
    _time_step = 5
    _feature_size = 20
    _x = torch.rand(_batch_size, _time_step, _feature_size)
    _y, _ = attention(_x, _x, _x, None)
    _mask = subsequent_mask(_time_step)
    _y_masked, _ = attention(_x, _x, _x, _mask)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(311)
    ax1.imshow(_x[0])
    ax1.set_title("x")
    ax1.grid()
    ax2 = fig.add_subplot(312)
    ax2.imshow(_y[0])
    ax2.set_title("y")
    ax2.grid()
    ax3 = fig.add_subplot(313)
    ax3.imshow(_y_masked[0])
    ax3.set_title("Masked y")
    ax3.grid()
    plt.show()


class MultiHeadedAttention(AttentionInterface):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        # d_model = d_k * head

        self.qkv_linears = duplicate_module(Dense(d_model, d_model), 3)
        self.final_linear = Dense(d_model, d_model)
        self.attention = None
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor = None
    ):
        batch_size = query.shape[0]
        if mask is not None:
            mask = torch.unsqueeze(mask, 1)

        # 1) Do all the linear projections in batch from d_model => h x d_k .
        query, key, value = [
            # [batch_size, time_step, d_model]
            # [batch_size, time_step, h, d_k]
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            # [batch_size, h, time_step, d_k]
            for linear, x in zip(self.qkv_linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attention = attention(
            query, key, value, mask=mask, dropout_module=self.dropout
        )
        # [batch_size, h, time_step, d_k]

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # [batch_size, time_step, h, d_k]
        # [batch_size, time_step, d_model]
        return self.final_linear(x)


class PositionWiseFeedForward(FeedForwardInterface):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1
    ):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = Dense(d_model, d_ff)
        self.w_2 = Dense(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

