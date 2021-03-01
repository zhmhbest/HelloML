from torch import nn
from torch import Tensor
from .base import LayerNorm
from .architecture import Layer, Encoder, Decoder, EncoderDecoder
from .core import MultiHeadedAttention, PositionWiseFeedForward
from .embed import PositionalEncoding, Embeddings


class EncoderLayer(Layer):
    def __init__(
            self,
            feature_size: int,
            ma: MultiHeadedAttention,
            ff: PositionWiseFeedForward,
            dropout: float,
            norm: nn.Module = LayerNorm
    ):
        super(EncoderLayer, self).__init__()
        self.feature_size = feature_size

        self.norm1 = norm(feature_size)
        self.multi_head_attention = ma
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = norm(feature_size)
        self.feed_forward = ff
        self.dropout2 = nn.Dropout(dropout)

    def __len__(self) -> int:
        return self.feature_size

    def forward(self, x, mask):
        x = self.dropout1(
            (lambda _x_: self.multi_head_attention(_x_, _x_, _x_, mask))(self.norm1(x))
        ) + x
        x = self.dropout2(
            self.feed_forward(self.norm2(x))
        ) + x
        return x


class DecoderLayer(Layer):
    def __init__(
            self,
            feature_size: int,
            mma: MultiHeadedAttention,
            ma: MultiHeadedAttention,
            ff: PositionWiseFeedForward,
            dropout: float,
            norm: nn.Module = LayerNorm
    ):
        super(DecoderLayer, self).__init__()
        self.feature_size = feature_size

        self.norm1 = norm(feature_size)
        self.masked_multi_head_attention = mma
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = norm(feature_size)
        self.multi_head_attention = ma
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = norm(feature_size)
        self.feed_forward = ff
        self.dropout3 = nn.Dropout(dropout)

    def __len__(self) -> int:
        return self.feature_size

    def forward(self, x, m, m_mask, x_mask):
        x = self.dropout1(
            (lambda _x_: self.masked_multi_head_attention(_x_, _x_, _x_, x_mask))(self.norm1(x))
        ) + x
        x = self.dropout2(
            (lambda _x_: self.masked_multi_head_attention(_x_, m, m, m_mask))(self.norm2(x))
        ) + x
        x = self.dropout3(
            self.feed_forward(self.norm3(x))
        ) + x
        return x


class Transformer(EncoderDecoder):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor, x_mask) -> Tensor:
        return self.encoder(x, x_mask)

    def decode(self, m: Tensor, m_mask, x: Tensor, x_mask) -> Tensor:
        return self.decoder(x, m, m_mask, x_mask)
