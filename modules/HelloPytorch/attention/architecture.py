from torch import Tensor
from torch import nn
from .base import clones, LayerNorm


class Layer(nn.Module):
    def __len__(self) -> int:
        raise NotImplementedError()


class Coder(nn.Module):
    def __init__(self, layer: Layer, loop: int, norm: nn.Module = LayerNorm):
        """
        :param layer: 基础层
        :param loop:  基础层循环次数
        """
        super(Coder, self).__init__()
        self.layers = clones(layer, loop)
        self.norm = norm(len(layer))

    def forward(self, x, *args):
        for layer in self.layers:
            x = layer(x, *args)
        return self.norm(x)


class Encoder(Coder):
    pass
    #  forward(self, x, mask)


class Decoder(Coder):
    pass
    #  forward(self, x, memory, src_mask, tgt_mask)


class EncoderDecoder(nn.Module):
    def encode(self, x: Tensor, x_mask) -> Tensor:
        raise NotImplementedError()

    def decode(self, m: Tensor, m_mask, x: Tensor, x_mask) -> Tensor:
        raise NotImplementedError()

    def forward(self, x, y, x_mask, y_mask):
        return self.decode(
            self.encode(x, x_mask), x_mask,
            y, y_mask
        )
