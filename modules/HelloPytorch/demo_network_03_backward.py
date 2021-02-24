import torch
from torch.nn import MSELoss
from torch.nn import Sequential
from torch.nn import ReLU
from support.cnn import get_cnn_filtered_size, get_flatten_size

# Conv2d(in_channels: int, out_channels: int, kernel_size, stride=1, padding=0)
from torch.nn import Conv2d
# MaxPool2d(kernel_size, stride=kernel_size, padding=0)
from torch.nn import MaxPool2d
# Linear(in_features: int, out_features: int, bias: bool = True)
from torch.nn import Linear
# Flatten(tart_dim: int = 1, end_dim: int = -1)
from torch.nn import Flatten


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


DATA_CHANNEL = 1
DATA_SIZE_W = 32
DATA_SIZE_H = 32

# Conv2d
convoluted_size = get_cnn_filtered_size((DATA_SIZE_W, DATA_SIZE_H), 3, 1, 0)
# MaxPool2d
convoluted_size = get_cnn_filtered_size(convoluted_size, 2, 2, 0)
# Flatten
flattened_size = get_flatten_size(convoluted_size, 6)

model = Sequential(
    Conv2d(1, 6, 3),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(flattened_size, 120),
    ReLU(),
    Linear(120, 84),
    ReLU(),
    Linear(84, 10)
)
print('\n'.join([str(it.size()) for it in model.parameters()]))


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


if __name__ == '__main__':
    x_data = torch.randn(1, DATA_CHANNEL, DATA_SIZE_W, DATA_SIZE_H)
    y_data = torch.randn(1, 10)
    y_pred = model(x_data)

    # 损失函数
    loss_fn = MSELoss()
    loss_val = loss_fn(y_data, y_pred)
    print("LossValue =", loss_val)

    loss_val.backward()
    for it in model.parameters():
        print(it.grad[0])
        break

    # 梯度置为0
    model.zero_grad()
    for it in model.parameters():
        print(it.grad[0])
        break
