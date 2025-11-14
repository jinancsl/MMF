import random

import torch
import torch.nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#from models._3d import resnet
import utils
import argparse
import time
import os
import torch.nn.functional as F
import copy
import SimpleITK as sitk
from skimage.measure import label
import torch.nn as nn
import torch.nn.functional as F
from read_fatpad import people_label
import cv2
from skimage import transform

from sklearn import metrics
from MRNet_master_1.MRNet_master.src.utils import calculate_aucs

def train(net, optimizer, criterion, train_dataset, train_label, batch):
    net.train()

    sum_loss = 0.0
    sum_shuju = np.zeros([3])
    crect_sum_shuju = np.zeros([3])
    jishu = 0
    zong_image = []
    zong_label = []

    #for auc
    all_label = []
    all_out = []
    for i in range(len(train_dataset)):
        zong_image.append(train_dataset[i])
        zong_label.append(train_label[i])
        if len(zong_image) == batch:
            optimizer.zero_grad()  # 优化器梯度归零
            losses = 0
            for j_6 in range(len(zong_image)):
                ##########
                inputs = torch.from_numpy(zong_image[j_6]).cuda().float()
                labels_batch = torch.from_numpy(zong_label[j_6]).cuda().float()

                # forward + backward + optimize
                outputs = net(inputs)
                outputs = outputs.squeeze(dim=-1)
                labels_batch = labels_batch.squeeze(dim=-1)
                loss = criterion(outputs, labels_batch)
                losses = loss + losses

                single_shuju, correct_single_shuju = utils.compute_accuracy(
                    outputs,
                    labels_batch,
                    augmentation=False,
                    topk=(1, 1))  # acc1分别表示[总图像数、1类图像数、2类图像数]， correct_sum_single分别表示[分类正确的图像总数、1类分类正确数、2类分类正确数]
                sum_loss = sum_loss + loss
                sum_shuju = sum_shuju + single_shuju
                crect_sum_shuju = crect_sum_shuju + correct_single_shuju
                jishu = jishu + 1

                all_label.append(labels_batch.cpu().detach().numpy())
                all_out.append(outputs.cpu().detach().numpy())

            optimizer.zero_grad()  # 清空过往梯度；
            losses.backward()  # 反向传播，计算当前梯度；
            optimizer.step()  # 根据梯度更新网络参数
            zong_image = []
            zong_label = []
    avg_loss = sum_loss / (jishu)
    auc = calculate_aucs(all_label, all_out)
    return net, avg_loss, sum_shuju, crect_sum_shuju, auc


def val(net, criterion, train_dataset, train_label, batch):
    net.eval()
    with torch.no_grad():
        sum_loss = 0.0
        sum_shuju = np.zeros([3])
        crect_sum_shuju = np.zeros([3])
        jishu = 0
        zong_image = []
        zong_label = []

        # for auc
        all_label = []
        all_out = []
        for i in range(len(train_dataset)):
            zong_image.append(train_dataset[i])
            zong_label.append(train_label[i])
            if len(zong_image) == batch:
                losses = 0
                for j_6 in range(len(zong_image)):
                    ##########
                    inputs = torch.from_numpy(zong_image[j_6]).cuda().float()
                    labels_batch = torch.from_numpy(zong_label[j_6]).cuda().float()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    outputs = outputs.squeeze(dim=-1)
                    labels_batch = labels_batch.squeeze(dim=-1)
                    loss = criterion(outputs, labels_batch)
                    losses = loss + losses

                    single_shuju, correct_single_shuju = utils.compute_accuracy(
                        outputs,
                        labels_batch,
                        augmentation=False,
                        topk=(1, 1))  # acc1分别表示[总图像数、1类图像数、2类图像数]， correct_sum_single分别表示[分类正确的图像总数、1类分类正确数、2类分类正确数]
                    sum_loss = sum_loss + loss
                    sum_shuju = sum_shuju + single_shuju
                    crect_sum_shuju = crect_sum_shuju + correct_single_shuju
                    jishu = jishu + 1
                    all_label.append(labels_batch.cpu().detach().numpy())
                    all_out.append(outputs.cpu().detach().numpy())
                zong_image = []
                zong_label = []
        avg_loss = sum_loss / (jishu)
        auc = calculate_aucs(all_label, all_out)
        return avg_loss, sum_shuju, crect_sum_shuju, auc

def test(net, criterion, train_dataset, train_label, batch):
    net.eval()
    with torch.no_grad():
        sum_loss = 0.0
        sum_shuju = np.zeros([3])
        crect_sum_shuju = np.zeros([3])
        jishu = 0
        zong_image = []
        zong_label = []

        # for auc
        all_label = []
        all_out = []
        for i in range(len(train_dataset)):
            zong_image.append(train_dataset[i])
            zong_label.append(train_label[i])
            if len(zong_image) == batch:
                losses = 0
                for j_6 in range(len(zong_image)):
                    ##########
                    inputs = torch.from_numpy(zong_image[j_6]).cuda().float()
                    labels_batch = torch.from_numpy(zong_label[j_6]).cuda().float()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    outputs = outputs.squeeze(dim=-1)
                    labels_batch = labels_batch.squeeze(dim=-1)
                    loss = criterion(outputs, labels_batch)
                    losses = loss + losses

                    single_shuju, correct_single_shuju = utils.compute_accuracy(
                        outputs,
                        labels_batch,
                        augmentation=False,
                        topk=(
                            1, 1))  # acc1分别表示[总图像数、1类图像数、2类图像数]， correct_sum_single分别表示[分类正确的图像总数、1类分类正确数、2类分类正确数]
                    sum_loss = sum_loss + loss
                    sum_shuju = sum_shuju + single_shuju
                    crect_sum_shuju = crect_sum_shuju + correct_single_shuju
                    jishu = jishu + 1
                    all_label.append(labels_batch.cpu().detach().numpy())
                    all_out.append(outputs.cpu().detach().numpy())
                zong_image = []
                zong_label = []
        avg_loss = sum_loss / (jishu)
        label = np.zeros(len(all_label))
        out = np.zeros(len(all_out))
        for i in range(len(all_label)):
            label[i] = all_label[i]
            out[i] = all_out[i]
        auc = calculate_aucs(label, out)
        return avg_loss, sum_shuju, crect_sum_shuju, auc, label, out
