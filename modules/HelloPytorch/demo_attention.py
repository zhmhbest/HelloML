from torch import nn, Tensor
from attention import *
from copy import deepcopy
import torch
from torch import optim
from torch.nn import MSELoss
from attention.test import get_test_data

data_form = {
    "num_batches": 30,
    "batch_size": 20,
    "time_step": 10,
    "feature_size": 10,
    "target_size": 2
}
epoch = 2


class Transformer(nn.Module):
    def __init__(
            self,
            feature_size: int,
            target_size: int,
            d_model: int = 512,
            h: int = 8,
            d_ff: int = 2048,
            duplicate: int = 2,
            dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        self.dense_i = Dense(feature_size, d_model)
        self.dense_o = Dense(target_size, d_model)
        self.dense_r = Dense(d_model, target_size)

        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(
            EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout),
            duplicate
        )
        self.decoder = Decoder(
            DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout),
            duplicate
        )

    def forward(
            self,
            x: Tensor, y: Tensor,
            x_mask: Tensor, y_mask: Tensor
    ) -> Tensor:
        """
        :param x: [batch_size, time_step, feature_size]
        :param y: [batch_size, time_step, target_size]
        :param x_mask:
        :param y_mask:
        :return:
        """
        x = self.dense_i(x)
        # print("x =", x.shape)  # [batch_size, time_step, d_model]
        y = self.dense_o(y)
        # print("y =", y.shape)  # [batch_size, time_step, d_model]

        m_mask = x_mask
        m = self.encoder(x, x_mask)
        # print("m =", m.shape)  # [batch_size, time_step, d_model]

        r = self.decoder(y, m, y_mask, m_mask)
        # print("r =", r.shape)  # [batch_size, time_step, d_model]
        return self.dense_r(r)


model = Transformer(data_form['feature_size'], data_form['target_size'])
loss_fn = MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
data_holder = get_test_data(**data_form)
print("data_i_shape =", data_holder[0][0].shape)
print("data_o_shape =", data_holder[0][1].shape)
print()

for i in range(epoch):
    for x_batch, y_batch in data_holder:
        # 前向传播
        y_pred = model(x_batch, y_batch, None, None)
        # print("前向传播", y_pred.shape, y_batch.shape)

        # 计算损失
        loss_val = loss_fn(y_batch, y_pred)
        print("LossValue =", loss_val)

        # 反向传播
        loss_val.backward()

        # 优化权重
        optimizer.step()
