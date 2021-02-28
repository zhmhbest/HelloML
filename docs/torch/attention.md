# [Attention is all you need](./index.html)

## Architecture

```mermaid
graph TD;
    EncoderInput{DataInput} --> EncoderLayer1Input;
    DecoderInput{PreTimeStepOutput} --> DecoderLayer1Input;

    subgraph TheEncoder[Encoder]
        style TheEncoder stroke:#333,stroke-width:2px;
        subgraph EncoderLayer1
            style EncoderLayer1 stroke:#333,stroke-width:2px;
            EncoderLayer1Input{Input};
            EncoderLayer1Output{Output};
            EncoderLayer1Add((+));
            EncoderLayer1Model[Self Atention Layer];
            EncoderLayer1Input --> EncoderLayer1Model;
            EncoderLayer1Input --> EncoderLayer1Model;
            EncoderLayer1Input --> EncoderLayer1Model;
            EncoderLayer1Model --> EncoderLayer1Add;
            EncoderLayer1Input --> EncoderLayer1Add -- LayerNorm --> EncoderLayer1Output;
        end

        subgraph EncoderLayer2
            style EncoderLayer2 stroke:#333,stroke-width:2px;
            EncoderLayer2Input{Input};
            EncoderLayer2Output{Output};
            EncoderLayer2Add((+));
            EncoderLayer2Model[Feed Forward Layer];
            EncoderLayer2Input --> EncoderLayer2Model --> EncoderLayer2Add;
            EncoderLayer2Input --> EncoderLayer2Add -- LayerNorm --> EncoderLayer2Output;
        end

        EncoderLayer1Output --> EncoderLayer2Input;
    end

    EncoderLayer2Output --> DecoderLayer2Model;
    EncoderLayer2Output --> DecoderLayer2Model;

    subgraph TheDecoder[Decoder]
        style TheDecoder stroke:#333,stroke-width:2px;
         subgraph DecoderLayer1
            style DecoderLayer1 stroke:#333,stroke-width:2px;
            DecoderLayer1Input{Input};
            DecoderLayer1Output{Output};
            DecoderLayer1Add((+));
            DecoderLayer1Model[Masked Self Atention Layer];
            DecoderLayer1Input --> DecoderLayer1Model;
            DecoderLayer1Input --> DecoderLayer1Model;
            DecoderLayer1Input --> DecoderLayer1Model;
            DecoderLayer1Model --> DecoderLayer1Add;
            DecoderLayer1Input --> DecoderLayer1Add -- LayerNorm --> DecoderLayer1Output;
        end

        DecoderLayer1Output --Q--> DecoderLayer2Model;
        DecoderLayer1Output --> DecoderLayer2Add;

        subgraph DecoderLayer2
            style DecoderLayer2 stroke:#333,stroke-width:2px;
            DecoderLayer2Output{Output};
            DecoderLayer2Add((+));
            DecoderLayer2Model[Self Atention Layer];
            DecoderLayer2Model --> DecoderLayer2Add -- LayerNorm --> DecoderLayer2Output;
        end

        DecoderLayer2Output --> DecoderLayer3Input

        subgraph DecoderLayer3
            style DecoderLayer3 stroke:#333,stroke-width:2px;
            DecoderLayer3Input{Input};
            DecoderLayer3Output{Output};
            DecoderLayer3Add((+));
            DecoderLayer3Model[Feed Forward Layer];
            DecoderLayer3Input --> DecoderLayer3Model --> DecoderLayer3Add;
            DecoderLayer3Input --> DecoderLayer3Add -- LayerNorm --> DecoderLayer3Output;
        end
    end

    DecoderLayer3Output --> Liner --> Softmax;
```

## Base

### BatchNorm

pass

### LayerNorm

$$Norm(x) = w \dfrac{x - Mean(x)}{Std(x) + eps} + b$$

```python
import torch


class LayerNorm(torch.nn.Module):
    """
    Construct a layernorm module.
    https://arxiv.org/abs/1607.06450
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # Parameter: A kind of Tensor that is to be considered a module parameter.
        self.w = torch.nn.Parameter(torch.ones(features))
        self.b = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # calculate by dimension.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b

```

## Mask

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

```python
def subsequent_mask(size):
    """Mask out subsequent positions."""
    import numpy as np
    from torch import from_numpy
    mask = np.triu(np.ones((1, size, size), dtype=np.uint8), k=1) == 0
    return from_numpy(mask)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print("Black: F")
    print("White: T")
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0], cmap=plt.cm.gray)
    plt.grid()
    plt.show()

```

## Core

### Self-Attention

```mermaid
graph LR;
    subgraph SelfAttentionLayer[Self Attention Layer]
        x{输入};
        y{输出};
        dotQK((.));
        dotQKV((.));
        x --> Q --> dotQK;
        x --> K -- Transpose --> dotQK;
        dotQK -- Softmax --> dotQKV;
        x --> V --> dotQKV;
        dotQKV --> y;
    end
```

$$\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

- $d$ is the dimension of  $Q$ and $K$.

```python
def attention(q, k, v, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    :param q: query
    :param k: key
    :param v: value
    :param mask:
    :param dropout:
    :return:
    """
    from math import sqrt
    from torch import matmul
    from torch.nn.functional import softmax

    d_k = q.size(-1)
    k_t = k.transpose(-2, -1)
    scores = matmul(q, k_t) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores_head = softmax(scores, dim=-1)
    if dropout is not None:
        scores_head = dropout(scores_head)
    output = matmul(scores_head, v)

    return output, scores

```

### Feed-Forward

pass
