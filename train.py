import os
import re
from glob import glob

import numpy as np
import torch
import timm
from torch import nn
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
from endaaman.torch import Trainer, TrainCommander
from endaaman.metrics import MultiAccuracy

from datasets import ClassificationDataset



class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-24):
        super().__init__()
        self.eps = eps
        self.loss_fn = nn.NLLLoss()
        # self.num_classes = num_classes

    def forward(self, x, y):
        return self.loss_fn((x + self.eps).log(), y)


class T(Trainer):
    def prepare(self):
        self.criterion = CrossEntropyLoss()

    # def get_scheduler(self, optimizer):
    #     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.99 ** x)

    def eval(self, inputs, labels):
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs


    def get_batch_metrics(self):
        return {
            'acc': MultiAccuracy(),
        }

    def get_epoch_metrics(self):
        return { }


class TimmModel(nn.Module):
    def __init__(self, name='tf_efficientnetv2_b0', num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(name, pretrained=True, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x):
        x =  self.base(x)
        if self.num_classes > 1:
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x



class C(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0')

    def arg_start(self, parser):
        pass

    def run_start(self):
        model = TimmModel(self.args.model, 3).to(self.device)

        loaders = [self.as_loader(ClassificationDataset(
            src='data/generate/'
        )) for t in ['train', 'test']]

        trainer = T(
            name=self.args.model,
            model=model,
            loaders=loaders,
            device=self.device,
        )

        trainer.start(
            max_epoch=self.args.epoch,
            lr=self.args.lr
        )


if __name__ == '__main__':
    c = C({
        'epoch': 50,
        'lr': 0.0001,
        'batch_size': 16,
    })
    c.run()
