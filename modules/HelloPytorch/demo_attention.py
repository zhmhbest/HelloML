from copy import deepcopy
from attention import *


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N),
        Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N)
    )

    for p in model.parameters():
        print(p.shape)
    return model


tmp_model = make_model(10, 10, 2)
