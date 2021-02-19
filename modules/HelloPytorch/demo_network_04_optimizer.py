import torch
from torch import optim
from torch.nn import MSELoss
from demo_network_03_backward import model, DATA_CHANNEL, DATA_SIZE_W, DATA_SIZE_H

x_data = torch.randn(1, DATA_CHANNEL, DATA_SIZE_W, DATA_SIZE_H)
y_data = torch.randn(1, 10)
loss_fn = MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for i in range(50):
    # 前向传播
    y_pred = model(x_data)

    # 计算损失
    loss_val = loss_fn(y_data, y_pred)
    print("LossValue =", loss_val)

    # 反向传播
    loss_val.backward()

    # 优化权重
    optimizer.step()
