import torch
from torch import nn, Tensor
from attention import *
from copy import deepcopy
from attention.test import generate_test_data

num_batches = 30
batch_size = 20
feature_size = 10
target_size = 2

d_model: int = 512
h: int = 8
d_ff: int = 2048
dropout: float = 0.1
duplicate = 2


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.dense_i = Dense(feature_size, d_model)
        self.dense_o = Dense(target_size, d_model)

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
        :param x: [batch_size, feature_size]
        :param y: [batch_size, target_size]
        :param x_mask:
        :param y_mask:
        :return:
        """
        x = self.dense_i(x)
        print(x.shape)
        y = self.dense_o(y)
        print(y.shape)

        m = self.encoder(x, x_mask)
        m_mask = x_mask
        print(m.shape)
        r = self.decoder(y, m, y_mask, m_mask)
        print(r.shape)
        return r


model = Transformer()

for x_batch, y_batch in generate_test_data(num_batches, batch_size, feature_size, target_size, factor=11):
    print(x_batch.shape, y_batch.shape)
    model(x_batch, y_batch, None, None)
    break






# class Transformer(nn.Module):
#     def __init__(
#             self,
#             x_vocab: int,
#             y_vocab: int,
#             N: int = 6,
#             d_model: int = 512,
#             h: int = 8,
#             d_ff: int = 2048,
#             dropout: float = 0.1
#     ):
#         super(Transformer, self).__init__()
#         print(f"{x_vocab} -> {y_vocab}")
#
#         attn = MultiHeadedAttention(h, d_model)
#         ff = PositionWiseFeedForward(d_model, d_ff, dropout)
#         position = PositionalEncoding(d_model, dropout)
#
#         self.encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N)
#         self.decoder = Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N)
#         self.embeddings_x = nn.Sequential(Embeddings(d_model, x_vocab), deepcopy(position))
#         self.embeddings_y = nn.Sequential(Embeddings(d_model, y_vocab), deepcopy(position))
#         self.generator = Generator(d_model, y_vocab)
#
#     def forward(self, x: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor):
#         m = self.encoder(self.embeddings_x(x), x_mask)
#
#         # y, y_mask, m, x_mask
#         # y, y_mask, m, m_mask
#         r = self.decoder(self.embeddings_y(y), y_mask, m, x_mask)
#         return r




# dense = Dense(10, 512)
# yy = dense(x_one)
# print(yy.shape)

# model = Transformer(10, 1, 2)
# # # for p in model.parameters():
# # #     print(p.size())
# #
# z = model(x_one, None, y_one, None)
# print(z)
