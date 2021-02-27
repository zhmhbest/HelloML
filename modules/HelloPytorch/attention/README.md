# Attention is all you need

## Architecture

```mermaid
graph LR;
    inputs{输入};
    outputs{输出};
    inputs -- inputEmbedding --> Encoder -- outputEmbedding --> Decoder --> outputs;

    subgraph TheEncoder[Encoder]
        EncoderOutput{y};
        subgraph EncoderLayer1
            EncoderLayer1Input{x};
            EncoderLayer1Add((+));
            EncoderLayer1Model[SelfAtention];
            EncoderLayer1Input --Norm--> EncoderLayer1Model --Dropout--> EncoderLayer1Add;
            EncoderLayer1Input --> EncoderLayer1Add;
        end
        subgraph EncoderLayer2
            EncoderLayer2Input{x};
            EncoderLayer2Add((+));
            EncoderLayer2Model[FeedForward];
            EncoderLayer2Input --Norm--> EncoderLayer2Model --Dropout--> EncoderLayer2Add;
            EncoderLayer2Input --> EncoderLayer2Add;
        end
        EncoderLayer1Add --> EncoderLayer2Input;
        EncoderLayer2Add --Norm--> EncoderOutput;
    end

    subgraph TheDecoder[Decoder]
        DecoderOutput{y};
        subgraph DecoderLayer1
            DecoderLayer1Input{x};
            DecoderLayer1Add((+));
            DecoderLayer1Model[SelfAtention];
            DecoderLayer1Input --Norm--> DecoderLayer1Model --Dropout--> DecoderLayer1Add;
            DecoderLayer1Input --> DecoderLayer1Add;
        end
        subgraph DecoderLayer2
            DecoderLayer2Input{x};
            DecoderLayer2Add((+));
            DecoderLayer2Model[SelfAtention];
            DecoderLayer2Input --Norm--> DecoderLayer2Model --Dropout--> DecoderLayer2Add;
            DecoderLayer2Input --> DecoderLayer2Add;
        end
        subgraph DecoderLayer3
            DecoderLayer3Input{x};
            DecoderLayer3Add((+));
            DecoderLayer3Model[FeedForward];
            DecoderLayer3Input --Norm--> DecoderLayer3Model --Dropout--> DecoderLayer3Add;
            DecoderLayer3Input --> DecoderLayer3Add;
        end
        DecoderLayer1Add --> DecoderLayer2Input;
        DecoderLayer2Add --> DecoderLayer3Input;
        DecoderLayer3Add --Norm--> DecoderOutput;
    end
```

## Base

### norm

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

## mask

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

```python
import numpy as np
import torch


def subsequent_mask(size):
    """Mask out subsequent positions."""
    mask = np.triu(np.ones((1, size, size), dtype=np.uint8), k=1) == 0
    return torch.from_numpy(mask)


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

### SelfAttention

pass

### FeedForward

pass
