import os
from einops import rearrange
import numpy as np
import cv2
from PIL import Image
from PIL.ImageFile import ImageFile

img_path = "./dump/test.png"
if not os.path.exists(img_path):
    dirname = os.path.dirname(img_path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    img_w = 64
    img_h = 32
    channel_r = np.random.uniform(0, 0.2, size=(img_w, img_h))
    channel_g = np.random.uniform(0, 0.6, size=(img_w, img_h))
    channel_b = np.random.uniform(0, 0.8, size=(img_w, img_h))
    img_arr = (np.array([channel_r, channel_g, channel_b]) * 255).astype(np.uint8)
    img_img = Image.fromarray(rearrange(img_arr, 'C W H -> H W C'))
    img_img.save(img_path)

img_cv2: np.ndarray = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
img_arr_cv2 = img_cv2
print(type(img_cv2), f"(H,W,RGB)={img_arr_cv2.shape}")

# cv2.imshow("", img_cv2)
# cv2.waitKey()

img_pil: ImageFile = Image.open(img_path).convert('RGB')
img_arr_pil = np.asarray(img_pil)
print(type(img_pil), f"(H,W,RGB)={img_arr_pil.shape}")

img_pil.show()
