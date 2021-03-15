from CICIDS2017 import make_iscx_dump, get_iscx_dump
from demo_attention import Transformer
import numpy as np
import torch
from torch import optim
from torch.nn import MultiLabelSoftMarginLoss
from sklearn.preprocessing import OneHotEncoder


"""
    数据
"""
BATCH_SIZE = 32
make_iscx_dump(BATCH_SIZE)
info, reader = get_iscx_dump()
hot = OneHotEncoder()
label = np.array(info['label']).reshape(-1, 1)
hot.fit(label)
# print(hot.transform(label).toarray())
INPUT_SIZE = 78  # x_batch.shape[1]
OUTPUT_SIZE = len(label)
print(INPUT_SIZE, OUTPUT_SIZE)


"""
    模型
"""
model = Transformer(INPUT_SIZE, OUTPUT_SIZE)
loss_fn = MultiLabelSoftMarginLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()


"""
    训练
"""
index = 0
for index in range(info['num_batches']):
    x_batch, y_batch = reader.pop()
    x_batch = torch.tensor(x_batch, dtype=torch.float32)
    y_batch = torch.tensor(hot.transform(y_batch).toarray(), dtype=torch.float32)

    # 前向传播
    y_pred = model(x_batch, y_batch, None, None)

    # 计算损失
    loss_val = loss_fn(y_batch, y_pred)
    print("LossValue =", loss_val)

    # 反向传播
    loss_val.backward()

    # 优化权重
    optimizer.step()

    if 10 == index:
        break
