from attention.common_header import *


def duplicate_module(module: nn.Module, loop_times: int):
    """
    Duplicate the module
    :param module: 基础模型
    :param loop_times: 循环次数
    :return:
    """
    return nn.ModuleList([deepcopy(module) for _ in range(loop_times)])


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    """
    def __init__(self, feature_size: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        import torch
        self.w = nn.Parameter(torch.ones(feature_size))
        self.b = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b


if __name__ == '__main__':
    print("Test LayerNorm")
    _batch_size = 2
    _feature_size = 3
    _x = torch.arange(_batch_size * _feature_size, dtype=torch.float).view(_batch_size, _feature_size)
    print(_x)
    print(LayerNorm(_feature_size)(_x))


def subsequent_mask(size: int):
    """
    Mask out subsequent positions
    """
    mask = np.triu(np.ones((1, size, size), dtype=np.uint8), k=1) == 0
    return torch.from_numpy(mask)


if __name__ == '__main__':
    print("Test Mask")
    import matplotlib.pyplot as plt
    _size = 100
    _data = torch.rand(1, _size, _size)
    _mask = subsequent_mask(_size)
    _result = torch.masked_fill(_data, _mask, 0)
    plt.figure(figsize=(5, 5))
    plt.imshow(_result[0])
    plt.grid()
    plt.show()
