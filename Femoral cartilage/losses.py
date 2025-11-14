import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass
'''
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, prob, target):
        epsilon = 1e-10
        loss = - (target * torch.log(prob + epsilon) + (1 - target) * (torch.log(1 - prob)))
        loss = torch.sum(loss) / torch.numel(target)
        return loss
'''
class BCEFocalLosswithLogits_guguruangu_nei(nn.Module):
    def __init__(self, gamma=0, alpha=0.2, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_nei, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_wai(nn.Module):
    def __init__(self, gamma=0, alpha=1.5, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_wai, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_nei_qian(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_nei_qian, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_nei_zhong(nn.Module):
    def __init__(self, gamma=0, alpha=0.8, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_nei_zhong, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_nei_hou(nn.Module):
    def __init__(self, gamma=0, alpha=2, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_nei_hou, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_wai_qian(nn.Module):
    def __init__(self, gamma=0, alpha=1.2, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_wai_qian, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_wai_zhong(nn.Module):
    def __init__(self, gamma=0, alpha=5, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_wai_zhong, self).__init__()
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

class BCEFocalLosswithLogits_guguruangu_wai_hou(nn.Module):
    def __init__(self, gamma=0, alpha=8, reduction='mean'):
        super(BCEFocalLosswithLogits_guguruangu_wai_hou, self).__init__()
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
