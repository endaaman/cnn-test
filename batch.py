import os
import os.path
from glob import glob

import torch
from PIL import Image, ImageOps, ImageFile, ImageDraw
import numpy as np
from tqdm import tqdm

from endaaman import Commander

class CMD(Commander):
    def random_rect(self, width=(50, 100), height=(50, 100)):
        w = np.random.randint(width[0], width[1])
        h = np.random.randint(height[0], height[1])
        x = np.random.randint(w//2, 256-w//2)
        y = np.random.randint(h//2, 256-h//2)
        rect = (x-w, y-h, x+w, y+h)
        return rect

    def random_square(self, size=(50, 100)):
        s = np.random.randint(size[0], size[1])
        x = np.random.randint(s//2, 256-s//2)
        y = np.random.randint(s//2, 256-s//2)
        rect = (x-s, y-s, x+s, y+s)
        return rect

    def genetate_image_circle(self, size=256, object_size=(50, 100), bg=(0, 0, 0), fg=(255, 255, 255)):
        np.random.random_sample((256, 256, 3))
        img = Image.new('RGB', (size, size), bg)
        rect = self.random_square(size=object_size)
        draw = ImageDraw.Draw(img)
        draw.ellipse(rect, fill=fg)
        return img

    def genetate_image_square(self, size=256, object_size=(50, 100), bg=(0, 0, 0), fg=(255, 255, 255)):
        img = Imew('RGB', (size, size), bg)
        rect = self.random_square(size=object_size)
        draw = ImageDraw.Draw(img)
        draw.rectangle(rect, fill=fg)
        return img

    def genetate_image_triangle(self):
        pass

    def arg_generate(self, parser):
        parser.add_argument('--size', type=int, default=256)
        parser.add_argument('--count', type=int, default=100)
        parser.add_argument('--dest', type=str, default='data/generate')


    def run_generate(self):
        data = (
            ('circle', self.genetate_image_circle),
            ('square', self.genetate_image_square),
        )
        for (shape, fn) in data:
            dest = os.path.join(self.args.dest, shape)
            os.makedirs(dest, exist_ok=True)
            for i in range(self.args.count):
                img = fn(size=self.args.size)
                filename = f'{i}.png'
                path = os.path.join(dest, filename)
                img.save(path)

if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
