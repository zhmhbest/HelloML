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
    - 《Deep Residual Learning for Image Recognition》 https://arxiv.org/abs/1512.03385
    - 《Layer Normalization》 https://arxiv.org/abs/1607.06450
    """
    def __init__(self, feature_size: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        import torch
        self.w = nn.Parameter(torch.ones(feature_size))
        self.b = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b


class Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        # (n, i_dim) × (i_dim, o_dim) = (n, o_dim)
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = torch.add(x, self.bias)
        return x


if __name__ == '__main__':
    print("Test LayerNorm")
    _batch_size = 2
    _feature_size = 3
    _x = torch.arange(_batch_size * _feature_size, dtype=torch.float).view(_batch_size, _feature_size)
    print(_x)
    print(LayerNorm(_feature_size)(_x))


def subsequent_mask(size: int) -> BoolTensor:
    """
    Mask out subsequent positions
    :param size: time_step
    :return:
    """
    mask = np.triu(np.ones((1, size, size), dtype=np.uint8), k=1) == 0
    return BoolTensor(mask)


# if __name__ == '__main__':
#     print("Test Mask")
#     import matplotlib.pyplot as plt
#     _size = 100
#     _data = torch.rand(1, _size, _size)
#     _mask = subsequent_mask(_size)
#     _result = torch.masked_fill(_data, _mask, 0)
#     print(_data)
#     print(_mask)
#     print(_result)
#     plt.figure(figsize=(5, 5))
#     plt.imshow(_result[0])
#     plt.grid()
#     plt.show()
