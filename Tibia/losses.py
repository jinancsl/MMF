import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class BCEFocalLosswithLogits_jinggu_wai(nn.Module):
    def __init__(self, gamma=0, alpha=4, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_wai, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_nei(nn.Module):
    def __init__(self, gamma=0, alpha=0.3, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_nei, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_wai_qian(nn.Module):
    def __init__(self, gamma=0, alpha=5, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_wai_qian, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_wai_zhong(nn.Module):
    def __init__(self, gamma=0, alpha=1, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_wai_zhong, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_wai_hou(nn.Module):
    def __init__(self, gamma=0, alpha=5, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_wai_hou, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_nei_qian(nn.Module):
    def __init__(self, gamma=0, alpha=0.8, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_nei_qian, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_nei_zhong(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_nei_zhong, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_jinggu_nei_hou(nn.Module):
    def __init__(self, gamma=0, alpha=2, reduction='mean'):
        super(BCEFocalLosswithLogits_jinggu_nei_hou, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class BCEFocalLosswithLogits_wuruangufugaiqu(nn.Module):
    def __init__(self, gamma=0, alpha=0.25, reduction='mean'):
        super(BCEFocalLosswithLogits_wuruangufugaiqu, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        epsilon = 1e-10
        #logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        #loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
        #       (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        loss = - alpha * (1 - logits + epsilon) ** gamma * target * torch.log(logits + epsilon) - \
               (logits + epsilon) ** gamma * (1 - target) * torch.log(1 - logits + epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss