import os
import os.path
import re
import shutil
from enum import Enum
from glob import glob
from typing import NamedTuple, Callable
from collections import OrderedDict
from endaaman import Commander
from endaaman.torch import calc_mean_and_std, pil_to_tensor, tensor_to_pil

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True




class Item(NamedTuple):
    path: str
    image: ImageType
    label: int
    label_str: str


class ClassificationDataset(Dataset):
    def __init__(self, src='data/generate/'):
        self.src = src
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.load_data()

    def load_data(self):
        self.labels = os.listdir('data/generate/')

        self.items = []
        for i, label_str in enumerate(self.labels):
            for path in glob(os.path.join(self.src, label_str, '*.png')):
                self.items.append(Item(
                    path=path,
                    image=Image.open(path).copy(),
                    label=i,
                    label_str=label_str,
                ))
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = self.transform(item.image)
        y = torch.tensor(item.label)
        return x, y


class CMD(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--src', '-s', default='./data/generate')

    def pre_common(self):
        self.ds = ClassificationDataset(
            # target=self.args.target,
            src=self.args.src,
        )

    def run_t(self):
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            print(y, x.shape)
            self.i = tensor_to_pil(x)
            break

if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
