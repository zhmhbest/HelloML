import torch.nn as nn
import torch.nn.functional as F
import math
from .base import clones, LayerNorm, subsequent_mask


class Layer(nn.Module):
    pass


class Coder(nn.Module):
    def __init__(self, layer: Layer, loop: int, norm: nn.Module = LayerNorm):
        """
        :param layer: 基础层
        :param loop:  循环次数
        """
        super(Coder, self).__init__()
        self.layers = clones(layer, loop)
        self.norm = norm

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


class Generator(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EncoderDecoder(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embed: nn.Module,
            tgt_embed: nn.Module,
            generator: Generator
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask),
            src_mask, tgt, tgt_mask
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
