# coding: utf-8
import os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_save(img, name: str, dir: str):
    pil_img = Image.fromarray(np.uint8(img))
    with open(os.path.join(dir,name), mode="w") as file:
      pil_img.save(file.name)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

for i in range(len(x_train)):
  img = x_train[i]
  label = t_train[i]
  img_save(img.reshape(28, 28), f"{i}_{label}.BMP", os.path.join(os.path.dirname(os.path.abspath(__file__)),"all_imgs"))
