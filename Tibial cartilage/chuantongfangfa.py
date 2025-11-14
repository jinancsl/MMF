# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import pandas as pd
import cv2
import losses
from MRNet_master_1.MRNet_master.src import model_mrnet
from train_validate_youhua import train, val
from read_fatpad import people_label
import SimpleITK as sitk
from skimage.measure import label

import matplotlib.pyplot as plt

NII_train_jinggu_fenlei = '/media/chen/DATA2/knee_dataset/gu_gusui_fenlei/dataset_nfyk_zhengli/IOA'
NII_train_jinggu_fenlei_test = '/media/chen/DATA2/knee_dataset/gu_gusui_fenlei/dataset_nfyk_zhengli/test'

data_dir = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/train/image'
data_dir_lable = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/train/label'

data_dir_validate = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/validate/image'
data_dir_validate_lable = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/validate/label'

data_dir_test = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/test/image'
data_dir_test_label = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/test/label'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet', )
    parser.add_argument('--dataset', default="None",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--loss', default='BCEFocalLosswithLogits')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=200, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay')

    args = parser.parse_args()

    return args


def sxzy_slice(out_slice):
    ########处理每一个切片的结果
    jixian_people = [0, 0, 0, 0]
    test_label_people_1 = out_slice.copy()
    for j_1 in range(out_slice.shape[0]):
        if test_label_people_1[j_1, :].sum() > 0:
            jixian_people[0] = j_1  # 最高点
            break

    for j_2 in range(out_slice.shape[0]):
        if test_label_people_1[out_slice.shape[0] - j_2 - 1, :].sum() > 0:
            jixian_people[1] = out_slice.shape[0] - j_2 - 1
            break  # 最低点

    test_label_people_2 = out_slice.copy()
    for j_3 in range(out_slice.shape[1]):
        if test_label_people_2[:, j_3].sum() > 0:
            jixian_people[2] = j_3
            break  # 最左点

    for j_4 in range(out_slice.shape[1]):
        if test_label_people_2[:, out_slice.shape[1] - j_4 - 1].sum() > 0:
            jixian_people[3] = out_slice.shape[1] - j_4 - 1
            break  # 最右点
    return jixian_people


def largestConnectComponent(bw_img):
    labeled_img, num = label(bw_img, connectivity=2, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 1
    max_num = 0
    if num > 0:
        for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
    lcc = (labeled_img == max_label)
    return lcc


def Normalize_image(data):
    data_normalize = data.copy()
    max = data_normalize.max()
    min = data_normalize.min()
    data_normalize = (data_normalize - min) / (max - min)
    return data_normalize


def read_niigz(file_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = sitk.ReadImage(file_path)
    # C,H,W
    # SimpleITK读出的image的data的数组顺序为：Channel,Height，Width
    image_arr_all = sitk.GetArrayFromImage(nii_image)
    image_arr_all = image_arr_all.transpose(2, 1, 0).astype('float32')
    image_arr_all = image_arr_all.transpose(0, 2, 1)
    return image_arr_all


def read_niigz_fenge_label(file_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = sitk.ReadImage(file_path)
    # C,H,W
    # SimpleITK读出的image的data的数组顺序为：Channel,Height，Width
    image_arr_all = sitk.GetArrayFromImage(nii_image)
    image_arr_all = image_arr_all.transpose(2, 1, 0).astype('float32')
    image_arr_all = image_arr_all.transpose(0, 2, 1)
    image_arr_all_temp = np.zeros([image_arr_all.shape[0], image_arr_all.shape[1], image_arr_all.shape[2]])
    image_arr_all_temp_gu = np.zeros([image_arr_all.shape[0], image_arr_all.shape[1], image_arr_all.shape[2]])
    '''
    image_arr_all[image_arr_all == 4] = 0     #软骨外前
    image_arr_all[image_arr_all == 5] = 0     #软骨外中
    image_arr_all[image_arr_all == 6] = 0     #软骨外后
    image_arr_all[image_arr_all == 10] = 1    #软骨内前
    image_arr_all[image_arr_all == 11] = 1    #软骨内中
    image_arr_all[image_arr_all == 12] = 1    #软骨内后

    image_arr_all[image_arr_all == 1] = 0     #骨外前
    image_arr_all[image_arr_all == 2] = 0      #骨外中
    image_arr_all[image_arr_all == 3] = 0      #骨外后
    image_arr_all[image_arr_all == 7] = 0      #骨内前
    image_arr_all[image_arr_all == 8] = 0     #骨内中
    image_arr_all[image_arr_all == 9] = 0      #骨内后
    image_arr_all[image_arr_all == 13] = 0      #骨无软骨覆盖
    '''
    image_arr_all_temp[image_arr_all == 4] = 1
    return image_arr_all_temp

def datasets_loading(files_people, NII_dir_1, NII_dir_2, NII_dir_3):
    sheets2 = people_label()  # 标签表, ndarry格式
    sheets3 = []
    for i in range(sheets2.shape[0]):
        people_num = sheets2[i, 0]  # 是哪个人
        people_leg = sheets2[i, 1]  # 是哪条腿
        xulie = ''
        if people_leg == 1:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_RIGHT'
        if people_leg == 2:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_LEFT'
        sheets3.append(xulie)

    zong = []
    labels_batch_ndarry = np.zeros([1])
    labels_batch_ndarry_zong = np.zeros([2*len(files_people), 1])
    for i in files_people:

        #####读取用于训练的三维图像
        if os.path.exists(os.path.join(NII_dir_1, str(i[0:7]),
                                       str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT')):  # 如果没有这个人，则跳过这一次循环，否则读取相应的序列
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT'
        elif os.path.exists(os.path.join(NII_dir_1, str(i[0:7]),
                                         str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT')):
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT'
        else:
            raise Exception("不存在的图像文件")

        image_people_temp = read_niigz(os.path.join(NII_dir_2, str(i)))  # 读取每个人的三维图像
        fenge_label = read_niigz_fenge_label(os.path.join(NII_dir_3, str(i)))  # 读取图像的分割标签

        ######
        fenge_label = largestConnectComponent(fenge_label)
        fenge_label_slice = np.zeros([fenge_label.shape[1], fenge_label.shape[2]])  # 获取软骨的范围
        zong_temp = []
        for gh in range(fenge_label.shape[0]):
            fenge_label[gh, :, :] = largestConnectComponent(fenge_label[gh, :, :])

            jixian_people = sxzy_slice(fenge_label[gh, :, :])  # 计算软骨的极限，以裁剪软骨
            jixian_people_zuobiao = [0, 0, 0, 0]
            size_max = max(jixian_people[3] - jixian_people[2], jixian_people[1] - jixian_people[0])
            size1 = jixian_people[1] - jixian_people[0]
            size2 = jixian_people[3] - jixian_people[2]
            if size1 < 10 and size2 < 10:
                print(jixian_people)
                continue
            jixian_people_zuobiao[0] = max(0, jixian_people[0] - (size1 - jixian_people[1] + jixian_people[0]) // 2)
            jixian_people_zuobiao[1] = jixian_people_zuobiao[0] + size1
            if jixian_people_zuobiao[1] >= 384:
                jixian_people_zuobiao[1] = 384
                jixian_people_zuobiao[0] = jixian_people_zuobiao[1] - size1

            jixian_people_zuobiao[2] = max(0, jixian_people[2] - (size2 - jixian_people[3] + jixian_people[2]) // 2)
            jixian_people_zuobiao[3] = jixian_people_zuobiao[2] + size2
            if jixian_people_zuobiao[3] >= 384:
                jixian_people_zuobiao[3] = 384
                jixian_people_zuobiao[2] = jixian_people_zuobiao[3] - size2

            if fenge_label[gh, :, :].sum() == 0:
                image_people_temp[gh, :, :] = np.zeros([384, 384])
            else:
                zong_temp.append(image_people_temp[gh, jixian_people_zuobiao[0]:jixian_people_zuobiao[1],
                                 jixian_people_zuobiao[2]:jixian_people_zuobiao[3]])
        image_people = np.zeros([len(zong_temp), 64, 64])
        for u_9 in range(len(zong_temp)):
            # image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (128, 128), interpolation=cv2.INTER_NEAREST)
            zong_temp[u_9] = Normalize_image(zong_temp[u_9])
            image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (64, 64), interpolation=cv2.INTER_NEAREST)

        # image_people = transform.resize(image_people, (batch, 128, 128), order=1)

        ###################读取标签
        for j_9 in range(len(sheets3)):
            if save_name == sheets3[j_9]:
                if (sheets2[j_9, 2] + sheets2[j_9, 8] + sheets2[j_9, 14]) > 0.5:
                    labels_batch_ndarry[0] = 1
                elif (sheets2[j_9, 2] + sheets2[j_9, 8] + sheets2[j_9, 14]) == 0:
                    labels_batch_ndarry[0] = 0
                break
        if j_9 == len(sheets3) - 1:
            print('cuowu_cuowu')
            continue
        ###########################

        train_image = np.zeros([image_people.shape[0], 1, 64, 64])
        train_image = train_image.transpose(1, 0, 2, 3)
        for k_1 in range(train_image.shape[1]):
            train_image[0, k_1, :, :] = image_people[k_1, :, :]
        train_image = train_image.transpose(1, 0, 2, 3)
        '''
        for jjj in range(train_image.shape[0]):
            # 显示结果
            plt.figure("jieguo_biaoqian_1")
            plt.imshow(train_image[jjj, 0, :, :])
            plt.pause(0.01)
            plt.close("jieguo_biaoqian_1")
        '''
        zong.append(train_image)
        labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]
    return zong, labels_batch_ndarry_zong


def datasets_loading_1(files_people, NII_dir_1, NII_dir_2, NII_dir_3):
    sheets2 = people_label()  # 标签表, ndarry格式
    sheets3 = []
    for i in range(sheets2.shape[0]):
        people_num = sheets2[i, 0]  # 是哪个人
        people_leg = sheets2[i, 1]  # 是哪条腿
        xulie = ''
        if people_leg == 1:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_RIGHT'
        if people_leg == 2:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_LEFT'
        sheets3.append(xulie)

    zong = []
    labels_batch_ndarry = np.zeros([1])
    labels_batch_ndarry_zong = np.zeros([2*len(files_people), 1])
    for i in files_people:

        #####读取用于训练的三维图像
        if os.path.exists(os.path.join(NII_dir_1, str(i[0:7]),
                                       str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT')):  # 如果没有这个人，则跳过这一次循环，否则读取相应的序列
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT'
        elif os.path.exists(os.path.join(NII_dir_1, str(i[0:7]),
                                         str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT')):
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT'
        else:
            raise Exception("不存在的图像文件")

        image_people_temp = read_niigz(os.path.join(NII_dir_2, str(i)))  # 读取每个人的三维图像
        fenge_label = read_niigz_fenge_label(os.path.join(NII_dir_3, str(i)))  # 读取图像的分割标签

        ######
        fenge_label = largestConnectComponent(fenge_label)
        fenge_label_slice = np.zeros([fenge_label.shape[1], fenge_label.shape[2]])  # 获取软骨的范围
        for gh in range(fenge_label.shape[0]):
            fenge_label[gh, :, :] = largestConnectComponent(fenge_label[gh, :, :])
            fenge_label_slice[fenge_label[gh, :, :] == 1] = 1

            if fenge_label[gh, :, :].sum() > 0:
                plt.figure("jieguo_biaoqian_1")
                plt.subplot(211)
                plt.imshow(image_people_temp[gh, :, :])
                plt.subplot(212)
                plt.imshow(fenge_label[gh, :, :])
                plt.pause(0.01)
                plt.close("jieguo_biaoqian_1")


        fenge_label_slice = largestConnectComponent(fenge_label_slice)
        jixian_people = sxzy_slice(fenge_label_slice)  # 计算软骨的极限，以裁剪软骨
        jixian_people_zuobiao = [0, 0, 0, 0]
        size_max = max(jixian_people[3] - jixian_people[2], jixian_people[1] - jixian_people[0])
        size1 = size_max
        size2 = size1
        if size1 < 10:
            print(jixian_people)
            continue
        jixian_people_zuobiao[0] = max(0, jixian_people[0] - (size1 - jixian_people[1] + jixian_people[0]) // 2)
        jixian_people_zuobiao[1] = jixian_people_zuobiao[0] + size1
        if jixian_people_zuobiao[1] >= 384:
            jixian_people_zuobiao[1] = 384
            jixian_people_zuobiao[0] = jixian_people_zuobiao[1] - size1

        jixian_people_zuobiao[2] = max(0, jixian_people[2] - (size2 - jixian_people[3] + jixian_people[2]) // 2)
        jixian_people_zuobiao[3] = jixian_people_zuobiao[2] + size2
        if jixian_people_zuobiao[3] >= 384:
            jixian_people_zuobiao[3] = 384
            jixian_people_zuobiao[2] = jixian_people_zuobiao[3] - size2

        zong_temp = []
        for u_10 in range(fenge_label.shape[0]):
            if fenge_label[u_10, :, :].sum() == 0:
                image_people_temp[u_10, :, :] = np.zeros([384, 384])
            else:
                zong_temp.append(image_people_temp[u_10, jixian_people_zuobiao[0]:jixian_people_zuobiao[1],
                                 jixian_people_zuobiao[2]:jixian_people_zuobiao[3]])
        image_people = np.zeros([len(zong_temp), 64, 64])
        for u_9 in range(len(zong_temp)):
            # image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (128, 128), interpolation=cv2.INTER_NEAREST)
            zong_temp[u_9] = Normalize_image(zong_temp[u_9])
            image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (64, 64), interpolation=cv2.INTER_NEAREST)

        # image_people = transform.resize(image_people, (batch, 128, 128), order=1)

        ###################读取标签
        for j_9 in range(len(sheets3)):
            if save_name == sheets3[j_9]:
                if (sheets2[j_9, 2] + sheets2[j_9, 8] + sheets2[j_9, 14]) > 0.5:
                    labels_batch_ndarry[0] = 1
                elif (sheets2[j_9, 2] + sheets2[j_9, 8] + sheets2[j_9, 14]) == 0:
                    labels_batch_ndarry[0] = 0
                break
        if j_9 == len(sheets3) - 1:
            print('cuowu_cuowu')
            continue
        ###########################

        train_image = np.zeros([image_people.shape[0], 1, 64, 64])
        train_image = train_image.transpose(1, 0, 2, 3)
        for k_1 in range(train_image.shape[1]):
            train_image[0, k_1, :, :] = image_people[k_1, :, :]
        train_image = train_image.transpose(1, 0, 2, 3)

        for jjj in range(train_image.shape[0]):
            # 显示结果
            plt.figure("jieguo_biaoqian_1")
            plt.imshow(train_image[jjj, 0, :, :])
            plt.pause(0.01)
            plt.close("jieguo_biaoqian_1")

        zong.append(train_image)
        labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]
    return zong, labels_batch_ndarry_zong

def datasets_loading_2(files_people, NII_dir_1, NII_dir_2, NII_dir_3):
    sheets2 = people_label()  # 标签表, ndarry格式
    sheets3 = []
    for i in range(sheets2.shape[0]):
        people_num = sheets2[i, 0]  # 是哪个人
        people_leg = sheets2[i, 1]  # 是哪条腿
        xulie = ''
        if people_leg == 1:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_RIGHT'
        if people_leg == 2:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_LEFT'
        sheets3.append(xulie)

    zong = []
    labels_batch_ndarry = np.zeros([1])
    labels_batch_ndarry_zong = np.zeros([len(files_people), 1])
    for i in files_people:

        #####读取用于训练的三维图像
        if os.path.exists(os.path.join(NII_dir_1, str(i[0:7]),
                                       str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT')):  # 如果没有这个人，则跳过这一次循环，否则读取相应的序列
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT'
        elif os.path.exists(os.path.join(NII_dir_1, str(i[0:7]),
                                         str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT')):
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT'
        else:
            raise Exception("不存在的图像文件")

        image_people_temp = read_niigz(os.path.join(NII_dir_2, str(i)))  # 读取每个人的三维图像
        fenge_label = read_niigz_fenge_label(os.path.join(NII_dir_3, str(i)))  # 读取图像的分割标签

        ######
        fenge_label = largestConnectComponent(fenge_label)
        fenge_label_slice = np.zeros([fenge_label.shape[1], fenge_label.shape[2]])  # 获取软骨的范围
        for gh in range(fenge_label.shape[0]):
            fenge_label[gh, :, :] = largestConnectComponent(fenge_label[gh, :, :])
            fenge_label_slice[fenge_label[gh, :, :] == 1] = 1
            image_people_temp[gh, :, :] = fenge_label[gh, :, :] * image_people_temp[gh, :, :]
            '''
            #image_people_temp[gh, :, :] = fenge_label[gh, :, :] * image_people_temp[gh, :, :]
            plt.figure("jieguo_biaoqian_1")
            plt.subplot(211)
            plt.imshow(image_people_temp[gh, :, :])
            plt.subplot(212)
            plt.imshow(fenge_label[gh, :, :])
            plt.pause(0.01)
            plt.close("jieguo_biaoqian_1")
            '''

        fenge_label_slice = largestConnectComponent(fenge_label_slice)
        jixian_people = sxzy_slice(fenge_label_slice)  # 计算软骨的极限，以裁剪软骨
        jixian_people_zuobiao = [0, 0, 0, 0]
        size_max = max(jixian_people[3] - jixian_people[2], jixian_people[1] - jixian_people[0])
        size1 = size_max
        size2 = size1
        if size1 < 10:
            print(jixian_people)
            continue
        jixian_people_zuobiao[0] = max(0, jixian_people[0] - (size1 - jixian_people[1] + jixian_people[0]) // 2)
        jixian_people_zuobiao[1] = jixian_people_zuobiao[0] + size1
        if jixian_people_zuobiao[1] >= 384:
            jixian_people_zuobiao[1] = 384
            jixian_people_zuobiao[0] = jixian_people_zuobiao[1] - size1

        jixian_people_zuobiao[2] = max(0, jixian_people[2] - (size2 - jixian_people[3] + jixian_people[2]) // 2)
        jixian_people_zuobiao[3] = jixian_people_zuobiao[2] + size2
        if jixian_people_zuobiao[3] >= 384:
            jixian_people_zuobiao[3] = 384
            jixian_people_zuobiao[2] = jixian_people_zuobiao[3] - size2

        zong_temp = []
        for u_10 in range(fenge_label.shape[0]):
            if fenge_label[u_10, :, :].sum() == 0:
                image_people_temp[u_10, :, :] = np.zeros([384, 384])
            else:
                zong_temp.append(image_people_temp[u_10, jixian_people_zuobiao[0]:jixian_people_zuobiao[1],
                                 jixian_people_zuobiao[2]:jixian_people_zuobiao[3]])
        image_people = np.zeros([len(zong_temp), 64, 64])
        for u_9 in range(len(zong_temp)):
            # image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (128, 128), interpolation=cv2.INTER_NEAREST)
            zong_temp[u_9] = Normalize_image(zong_temp[u_9])
            image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (64, 64), interpolation=cv2.INTER_NEAREST)

        # image_people = transform.resize(image_people, (batch, 128, 128), order=1)

        ###################读取标签
        for j_9 in range(len(sheets3)):
            if save_name == sheets3[j_9]:
                if (sheets2[j_9, 2] + sheets2[j_9, 8] + sheets2[j_9, 14]) > 0.5:
                    labels_batch_ndarry[0] = 1
                elif (sheets2[j_9, 2] + sheets2[j_9, 8] + sheets2[j_9, 14]) == 0:
                    labels_batch_ndarry[0] = 0
                break
        if j_9 == len(sheets3) - 1:
            print('cuowu_cuowu')
            continue
        ###########################

        train_image = np.zeros([image_people.shape[0], 1, 64, 64])
        train_image = train_image.transpose(1, 0, 2, 3)
        for k_1 in range(train_image.shape[1]):
            train_image[0, k_1, :, :] = image_people[k_1, :, :]
        train_image = train_image.transpose(1, 0, 2, 3)
        '''
        for jjj in range(train_image.shape[0]):
            # 显示结果
            plt.figure("jieguo_biaoqian_1")
            plt.imshow(train_image[jjj, 0, :, :])
            plt.pause(0.01)
            plt.close("jieguo_biaoqian_1")
        '''
        zong.append(train_image)
        labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]
    return zong, labels_batch_ndarry_zong


def main():
    args = parse_args()
    # args.dataset = "datasets"

    if args.name is None:
        args.name = '%s_%s_woDS' % (args.dataset, args.arch)
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEFocalLosswithLogits().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % args.arch)
    model = model_mrnet.MRNet()

    # model = torch.load('model__best.pth')  # 模型
    '''
    # 直接丢弃不需要的模块
    state_dict.pop('outconv.weight')
    state_dict.pop('outconv.bias')
    '''
    # model.load_state_dict(state_dict, strict=False)

    model = model.cuda()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    max_acc = 0
    max_auc = 0
    files_slice_all = os.listdir(data_dir)  # 列出所有的用于训练的数据
    random.shuffle(files_slice_all)

    files_slice_test = os.listdir(data_dir_test)  # 列出所有的用于的数据
    random.shuffle(files_slice_test)

    files_slice_validate = os.listdir(data_dir_validate)  # 列出所有的用于的数据
    random.shuffle(files_slice_validate)

    acc_single = [0, 0]
    acc_single_train = [0, 0]
    '''
    train_dataset, train_label = datasets_loading(files_slice_all, NII_train_jinggu_fenlei, data_dir, data_dir_lable)
    test_dataset, test_label = datasets_loading(files_slice_test, NII_train_jinggu_fenlei_test, data_dir_test,
                                                data_dir_test_label)
    validate_dataset, validate_label = datasets_loading(files_slice_validate, NII_train_jinggu_fenlei,
                                                        data_dir_validate, data_dir_validate_lable)
    '''
    '''
    train_dataset_2, train_label_2 = datasets_loading_2(files_slice_all, NII_train_jinggu_fenlei, data_dir, data_dir_lable)
    test_dataset_2, test_label_2 = datasets_loading_2(files_slice_test, NII_train_jinggu_fenlei_test, data_dir_test,
                                                data_dir_test_label)
    validate_dataset_2, validate_label_2 = datasets_loading_2(files_slice_validate, NII_train_jinggu_fenlei,
                                                        data_dir_validate, data_dir_validate_lable)

    for i in range(len(train_dataset_2)):
        train_dataset.append(train_dataset_2[i])
        train_label[len(train_dataset_2)+i, 0] = train_label_2[i, 0]
    for i in range(len(test_dataset_2)):
        test_dataset.append(test_dataset_2[i])
        test_label[len(test_dataset_2)+i, 0] = test_label_2[i, 0]
    for i in range(len(validate_dataset_2)):
        validate_dataset.append(validate_dataset_2[i])
        validate_label[len(validate_dataset_2)+i, 0] = validate_label_2[i, 0]
    
    np.savez(os.path.join('/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/maked/',
                          'train_datasets_gugu_nei_qian'), image=train_dataset, label=train_label)  # 保存乱序后的数据集
    np.savez(os.path.join('/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/maked/',
                          'test_datasets_gugu_nei_qian'), image=test_dataset, label=test_label)  # 保存乱序后的数据集
    np.savez(os.path.join('/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/maked/',
                          'validate_datasets_gugu_nei_qian'), image=validate_dataset, label=validate_label)  # 保存乱序后的数据集
    '''
    datasets_jinggu_train = np.load(os.path.join('/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/maked/','train_datasets_gugu_nei_qian.npz'), allow_pickle=True)  # 加载train datasets
    train_dataset = datasets_jinggu_train['image']
    train_label = datasets_jinggu_train['label']
    datasets_jinggu_test = np.load(os.path.join('/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/maked/','test_datasets_gugu_nei_qian.npz'), allow_pickle=True)  # 加载train datasets
    test_dataset = datasets_jinggu_test['image']
    test_label = datasets_jinggu_test['label']
    datasets_jinggu_train_nei = np.load(os.path.join('/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/gugu/maked/','validate_datasets_gugu_nei_qian.npz'), allow_pickle=True)  # 加载train datasets
    validate_dataset = datasets_jinggu_train_nei['image']
    validate_label = datasets_jinggu_train_nei['label']


    zhongduan = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        net, avg_loss, cum_acc_crect_train, cum_acc_train, auc_train = train(model, optimizer, criterion, train_dataset,
                                                                             train_label)

        val_avg_loss, cum_acc_crect_validate, cum_acc_validate, auc_test = val(net, criterion, validate_dataset,
                                                                               validate_label)

        avg_acc_train = cum_acc_crect_train[0] / cum_acc_train[0]  # 总准确率
        acc_single_train[0] = cum_acc_crect_train[1] / cum_acc_train[1]  # 1类准确率
        acc_single_train[1] = cum_acc_crect_train[2] / cum_acc_train[2]  # 2类准确率

        avg_acc = cum_acc_crect_validate[0] / cum_acc_validate[0]
        acc_single[0] = cum_acc_crect_validate[1] / cum_acc_validate[1]
        acc_single[1] = cum_acc_crect_validate[2] / cum_acc_validate[2]

        zhongduan = zhongduan + 1
        if avg_acc > max_auc:
            zhongduan = 0
            max_auc = avg_acc
            torch.save(net, 'model__best.pth')
            print("=> saved best model")
        if zhongduan >= 5:
            break
        print(epoch, avg_loss, cum_acc_train, avg_acc_train, acc_single_train, auc_train)
        print(val_avg_loss, cum_acc_validate, avg_acc, acc_single, auc_test)
        torch.cuda.empty_cache()
    print('Finished Training')

    model = torch.load('model__best.pth')  # 模型
    model = model.cuda()
    val_avg_loss, cum_acc_crect_validate, cum_acc_validate, auc_test = val(model, criterion, test_dataset, test_label)
    avg_acc = cum_acc_crect_validate[0] / cum_acc_validate[0]
    acc_single[0] = cum_acc_crect_validate[1] / cum_acc_validate[1]
    acc_single[1] = cum_acc_crect_validate[2] / cum_acc_validate[2]
    print(val_avg_loss, cum_acc_validate, avg_acc, acc_single, auc_test)


if __name__ == '__main__':
    main()
