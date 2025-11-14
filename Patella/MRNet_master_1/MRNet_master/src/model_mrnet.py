#!/usr/bin/env python3.6
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from MRNet_master_1.MRNet_master.src.RESNET import resnet18
from MRNet_master_1.MRNet_master.src.VGG19 import vgg19_bn_my


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = resnet18()
        #self.alexnet = vgg19_bn_my()
        self.fc = nn.Linear(512, 1)

        #self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=None, padding=0)
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.Sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight.data)
                #nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    @property
    def features(self):
        return self.alexnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        out = torch.tensor([]).to(batch.device)
        for i in range(batch.shape[0]):
            out = torch.cat((out, self.features(batch[i:i+1, 0:1, :, :])), 0)

        out = self.avg_pool(out).squeeze()
        out = out.max(dim=0, keepdim=True)[0].squeeze()

        batch_out = self.classifier(self.dropout(out))

        batch_out = self.Sigmoid(batch_out)
        return batch_out
    '''
    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
    '''