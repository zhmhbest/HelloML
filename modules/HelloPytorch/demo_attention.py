from torch import nn, Tensor
from attention import *
from copy import deepcopy


class Transformer(nn.Module):
    def __init__(
            self,
            x_vocab: int,
            y_vocab: int,
            N: int = 6,
            d_model: int = 512,
            h: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N)
        self.embeddings_x = nn.Sequential(Embeddings(d_model, x_vocab), deepcopy(position))
        self.embeddings_y = nn.Sequential(Embeddings(d_model, y_vocab), deepcopy(position))
        self.generator = Generator(d_model, y_vocab)

    def forward(self, x: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor):
        m = self.encoder(self.embeddings_x(x), x_mask)
        # y, y_mask, m, x_mask
        # y, y_mask, m, m_mask
        r = self.decoder(self.embeddings_y(y), y_mask, m, x_mask)
        return r


model = Transformer(10, 10, 2)
for p in model.parameters():
    print(p.size())


