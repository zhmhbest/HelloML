from attention.common_header import *
from attention.base import LayerNorm, duplicate_module


class AttentionInterface(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        raise NotImplementedError()


class FeedForwardInterface(nn.Module):
    def forward(self, x: Tensor):
        raise NotImplementedError()


class LayerInterface(nn.Module):
    def __len__(self) -> int:
        # 可以使用len(layer)返回层的维度
        raise NotImplementedError()


class Coder(nn.Module):
    def __init__(
            self,
            layer: LayerInterface,
            loop_times: int
    ):
        """
        :param layer: 基础层
        :param loop_times:  基础层循环次数
        """
        super(Coder, self).__init__()
        self.layers = duplicate_module(layer, loop_times)
        self.norm = LayerNorm(len(layer))

    def forward(self, x: Tensor, *args):
        for layer in self.layers:
            x = layer(x, *args)
        return self.norm(x)


class Encoder(Coder):
    pass


class Decoder(Coder):
    pass


class EncoderLayer(LayerInterface):
    def __init__(
            self,
            d_model: int,
            ma: AttentionInterface,
            ff: FeedForwardInterface,
            dropout: float = 0.1
    ):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.norm1 = LayerNorm(d_model)
        self.ma = ma
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.ff = ff
        self.dropout2 = nn.Dropout(dropout)

    def __len__(self) -> int:
        return self.d_model

    def forward(self, x: Tensor, x_mask: Tensor = None):
        x = self.dropout1(
            (lambda _x_: self.ma(_x_, _x_, _x_, x_mask))(self.norm1(x))
        ) + x
        x = self.dropout2(
            self.ff(self.norm2(x))
        ) + x
        return x


class DecoderLayer(LayerInterface):
    def __init__(
            self,
            feature_size: int,
            mma: AttentionInterface,
            ma: AttentionInterface,
            ff: FeedForwardInterface,
            dropout: float = 0.1
    ):
        super(DecoderLayer, self).__init__()
        self.feature_size = feature_size

        self.norm1 = LayerNorm(feature_size)
        self.mma = mma
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(feature_size)
        self.ma = ma
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = LayerNorm(feature_size)
        self.ff = ff
        self.dropout3 = nn.Dropout(dropout)

    def __len__(self) -> int:
        return self.feature_size

    def forward(self, x: Tensor, m: Tensor, x_mask: Tensor, m_mask: Tensor):
        x = self.dropout1(
            (lambda _x_: self.mma(_x_, _x_, _x_, x_mask))(self.norm1(x))
        ) + x
        x = self.dropout2(
            (lambda _x_: self.ma(_x_, m, m, m_mask))(self.norm2(x))
        ) + x
        x = self.dropout3(
            self.ff(self.norm3(x))
        ) + x
        return x
