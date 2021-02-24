import os
import time
import torch
from torch import optim
from torch.nn import Module
from torch.nn import ReLU, Flatten
from torch.nn import Conv2d, MaxPool2d, Linear
from torch.nn import CrossEntropyLoss

from support.dataset.cifar10 import load_data, show_images
from support.dataset import DataHolder
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader

from support.cnn import get_cnn_filtered_size, get_flatten_size
print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")

# 模型存储位置
if not os.path.exists("dump"):
    os.mkdir("dump")
dump_model = f"dump/{os.path.splitext(os.path.basename(__file__))[0]}.pt"


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    加载数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
print("\n--- Data Loading")
x_raw_train, y_raw_train, x_raw_test, y_raw_test, y_classes = load_data()
size_train = len(x_raw_train)
size_test = len(x_raw_test)

# 图片（高、宽、通道）
input_shape = x_raw_train[0].shape
print(f"(H, W, C) = {input_shape}")
print(f"Size(train) = {size_train}")
print(f"Size(test) = {size_test}")

# 显示头10张图片
# show_images(x_raw_train[0:10], y_raw_train[0:10], y_classes)


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    数据预处理
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
xy_train_loader = DataLoader(DataHolder(x_raw_train, y_raw_train, transform), batch_size=128, shuffle=True, num_workers=0)
xy_test_loader = DataLoader(DataHolder(x_raw_test, y_raw_test, transform), batch_size=128, shuffle=False, num_workers=0)
# for i, (x_batch, y_batch) in enumerate(xy_train_loader, 0):
print("--- Data Loaded\n")


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    创建模型
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
print("\n--- Model Building")
filtered_size = get_cnn_filtered_size(input_shape[0:2], 5)
filtered_size = get_cnn_filtered_size(filtered_size, 2, 2)
filtered_size = get_cnn_filtered_size(filtered_size, 5)
filtered_size = get_cnn_filtered_size(filtered_size, 2, 2)
flattened_size = get_flatten_size(filtered_size, 16)


class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu1 = ReLU()
        self.conv1 = Conv2d(3, 6, 5)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.flatten1 = Flatten()
        self.fc1 = Linear(flattened_size, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool1(self.relu1(self.conv2(x)))
        x = self.flatten1(x)
        x = self.relu1(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.fc3(x)
        return x


model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# 模块可训练参数
for item in model.state_dict():
    print(item, model.state_dict()[item].size())
print("--- Model Builded\n")

# 加载历史模型数据
if os.path.exists(dump_model):
    print("\n--- Model Loading")
    checkpoint = torch.load(dump_model)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    accuracy_last_train, accuracy_last_test = checkpoint['accuracy']
    print(f"Accuracy of last train: {accuracy_last_train}")
    print(f"Accuracy of last test : {accuracy_last_test}")
    print("--- Model Loaded\n")


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    训练模型
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


def train(epoch, print_each=100):
    for ep in range(1, 1 + epoch):
        loss_running = 0.0
        for i, (x_batch, y_batch) in enumerate(xy_train_loader, 0):
            # 前向传播
            optimizer.zero_grad()
            y_predict = model(x_batch)

            # 计算损失
            loss_val = criterion(y_predict, y_batch)

            # 反向传播及优化
            loss_val.backward()
            optimizer.step()

            loss_running += loss_val.item()
            if 1 == (print_each - (i % print_each)):
                print(f"epoch:{ep}, index:{i + 1}, loss:{loss_running / print_each}")
                loss_running = 0.0


print("\n--- Training")
train(2)
print("--- Trained\n")


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    预测
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


def get_correction(loader: DataLoader):
    global size_train
    total = 0
    correction = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            _, predicted = torch.max(y_predict, 1)
            correct = predicted == y_batch
            assert isinstance(correct, torch.Tensor)
            correction += torch.sum(correct).item()
            total += y_batch.size(0)
        return correction / total


print("\n--- Testing")
accuracy_train = get_correction(xy_train_loader)
accuracy_test = get_correction(xy_test_loader)
print(f"Accuracy of train: {accuracy_train}")
print(f"Accuracy of test : {accuracy_test}")
print("--- Tested\n")


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    模型持久化
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
print("\n--- Saving")
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "accuracy": [accuracy_train, accuracy_test]
}, dump_model)
print("--- Saved\n")
