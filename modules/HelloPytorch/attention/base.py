from torch import nn


class LayerNorm(nn.Module):
    """
    Construct a layernorm module.
    """

    def __init__(self, features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        import torch
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module: nn.Module, loop: int):
    """Produce N identical layers."""
    from copy import deepcopy
    from torch.nn import ModuleList
    return ModuleList([deepcopy(module) for _ in range(loop)])


def subsequent_mask(size: int):
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
