"""

"""
import os
import numpy as np
import pandas as pd
import torch
from torch.nn import Unfold
from einops import rearrange

if not os.path.exists("./dump"):
    os.mkdir("./dump")

IMAGE_W = 32
IMAGE_H = 32
IMAGE_C = 1
PATCH_SIZE = 6
STEP_SIZE = 1
PADDING_SIZE = PATCH_SIZE // 2

arr = torch.arange(IMAGE_W * IMAGE_H * IMAGE_C).view(IMAGE_C, IMAGE_H, IMAGE_W)
arr = arr.unsqueeze(0)
print(f"B C H W = {tuple(arr.shape)}")
pd.DataFrame(arr.tolist()[0][0]).to_csv("./dump/arr.csv", index=False, header=False)

res = Unfold(
    kernel_size=(PATCH_SIZE, PATCH_SIZE),  # PATCH大小
    stride=(STEP_SIZE, STEP_SIZE),  # PATCH的滑动步长
    dilation=1,  # 输出形式是否有间隔
    padding=PADDING_SIZE
)(arr.double()).long()
print(f"B CKK L = {tuple(res.shape)}")
L_SIZE = res.shape[-1]
RC_SIZE = int(np.sqrt(L_SIZE).tolist())
print(f"L = {L_SIZE}, RC = {RC_SIZE}, RC^2 = {np.power(RC_SIZE, 2)}")
res = rearrange(res, "B CKK L -> (B L) CKK")
res = res.view(-1, IMAGE_C, PATCH_SIZE, PATCH_SIZE)
print(f"N C P P = {tuple(res.shape)}")

pd.DataFrame(
    np.vstack(rearrange(res, "N C P1 P2 -> C N P1 P2")[0])
).to_csv("./dump/res.csv", index=False, header=False)
