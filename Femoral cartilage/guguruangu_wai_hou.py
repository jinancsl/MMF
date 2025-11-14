# -*- coding: utf-8 -*-
import gc
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
import joblib
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
from train_validate_youhua import train, val, test
from read_fatpad import people_label
import SimpleITK as sitk
from skimage.measure import label
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

data_dir_train = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/train/image/'
data_dir_train_label = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/train/label'

data_dir_validate = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/validate/image'
data_dir_validate_lable = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/validate/label'

data_dir_test = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/test/image'
data_dir_test_label = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/test/label'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',)
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
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4
                        , type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-8, type=float,
                        help='weight decay')

    args = parser.parse_args()

    return args

def Normalize(data):
    data_normalize = data.copy()
    for i in range(data.shape[2]):
        data_normalize[:, :, data.shape[2] - i - 1] = data[:, :, i]
    return data_normalize

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
    #plt.figure(), plt.imshow(labeled_img, 'gray')

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
    image_arr_all = Normalize(image_arr_all)
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
    image_arr_all = Normalize(image_arr_all)
    image_arr_all = image_arr_all.transpose(0, 2, 1)
    image_arr_all_temp = np.zeros([image_arr_all.shape[0], image_arr_all.shape[1], image_arr_all.shape[2]])
    '''
    image_arr_all[image_arr_all == 4] = 0     #骨内前
    image_arr_all[image_arr_all == 5] = 0     #骨内中
    image_arr_all[image_arr_all == 6] = 0     #骨内后
    image_arr_all[image_arr_all == 10] = 1    #软骨nei前
    image_arr_all[image_arr_all == 11] = 1    #软骨nei中
    image_arr_all[image_arr_all == 12] = 1    #软骨nei后

    image_arr_all[image_arr_all == 1] = 0     #骨外前
    image_arr_all[image_arr_all == 2] = 0      #骨外中
    image_arr_all[image_arr_all == 3] = 0      #骨外后
    image_arr_all[image_arr_all == 7] = 0      #软骨外前
    image_arr_all[image_arr_all == 8] = 0     #软骨外中
    image_arr_all[image_arr_all == 9] = 0      #软骨外后
    image_arr_all[image_arr_all == 13] = 0      #骨无软骨覆盖
    '''
    image_arr_all_temp[image_arr_all == 19] = 1
    #image_arr_all_temp[image_arr_all == 21] = 1
    #image_arr_all_temp[image_arr_all == 22] = 1
    return image_arr_all_temp

def jietu(jixian_people, size1, size2, image_people_temp, fenge_label):
    jixian_people_zuobiao = [0, 0, 0, 0]
    jixian_people_zuobiao[0] = max(0, jixian_people[0] - (size1 - jixian_people[1] + jixian_people[0]) // 2)
    jixian_people_zuobiao[1] = jixian_people_zuobiao[0] + size1
    if jixian_people_zuobiao[1] >= image_people_temp.shape[1]:
        jixian_people_zuobiao[1] = image_people_temp.shape[1]
        jixian_people_zuobiao[0] = jixian_people_zuobiao[1] - size1

    jixian_people_zuobiao[2] = max(0, jixian_people[2] - (size2 - jixian_people[3] + jixian_people[2]) // 2)
    jixian_people_zuobiao[3] = jixian_people_zuobiao[2] + size2
    if jixian_people_zuobiao[3] >= image_people_temp.shape[1]:
        jixian_people_zuobiao[3] = image_people_temp.shape[1]
        jixian_people_zuobiao[2] = jixian_people_zuobiao[3] - size2

    zong_temp = []
    for u_10 in range(fenge_label.shape[0]):
        if fenge_label[u_10, :, :].max() == 0:
            image_people_temp[u_10, :, :] = np.zeros([image_people_temp.shape[1], image_people_temp.shape[1]])
        else:
            zong_temp.append(image_people_temp[u_10, jixian_people_zuobiao[0]:jixian_people_zuobiao[1],
                             jixian_people_zuobiao[2]:jixian_people_zuobiao[3]])

    image_people = np.zeros([len(zong_temp), size1, size2])
    for u_9 in range(len(zong_temp)):
        zong_temp[u_9] = Normalize_image(zong_temp[u_9])
        # image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (64, 64), interpolation=cv2.INTER_NEAREST)
        image_people[u_9, :, :] = zong_temp[u_9]

    train_image = np.zeros([image_people.shape[0], 1, size1, size2])
    train_image = train_image.transpose(1, 0, 2, 3)
    for k_1 in range(train_image.shape[1]):
        train_image[0, k_1, :, :] = image_people[k_1, :, :]
    train_image = train_image.transpose(1, 0, 2, 3)
    return train_image

def datasets_loading(files_people, data_dir_image, data_dir_label):
    sheets2 = people_label()  # 标签表, ndarry格式
    sheets3 = []
    for i in range(sheets2.shape[0]):
        people_num = sheets2[i, 0]  # 是哪个人
        people_leg = sheets2[i, 1]  # 是哪条腿
        xulie = ''
        if people_leg == 1:
            xulie = 'right' + str(int(people_num))
        if people_leg == 2:
            xulie = 'left' + str(int(people_num))
        sheets3.append(xulie)

    zong = []
    labels_batch_ndarry = np.zeros([1])
    labels_batch_ndarry_zong = np.zeros([10 * len(files_people), 1])
    max_shuzhou_hengzhou = [0, 0]
    for i in files_people:
        image_people = read_niigz(os.path.join(data_dir_image, i))  # 读取每个人的三维图像
        label_people = read_niigz_fenge_label(os.path.join(data_dir_label, i))  # 读取图像的分割标签

        ###################读取标签
        for j_9 in range(len(sheets3)):
            if i[:-7] == sheets3[j_9]:
                if (sheets2[j_9, 7]) > 0.1:
                    labels_batch_ndarry[0] = 1
                elif (sheets2[j_9, 7]) <= 0.05:
                    labels_batch_ndarry[0] = 0
                break
        if j_9 == len(sheets3) - 1:
            print('cuowu_cuowu')
            continue
        ###########################

        ######
        label_people = largestConnectComponent(label_people)
        fenge_label_slice = np.zeros([label_people.shape[1], label_people.shape[2]])  # 获取软骨的范围
        for gh in range(label_people.shape[0]):
            fenge_label_slice[label_people[gh, :, :] == 1] = 1
            '''
            plt.figure("jieguo_biaoqian_1")
            plt.subplot(221)
            plt.imshow(image_people[gh, :, :])
            plt.subplot(222)
            plt.imshow(label_people[gh, :, :])
            plt.subplot(223)
            plt.imshow(fenge_label_slice)
            plt.pause(0.01)
            plt.close("jieguo_biaoqian_1")
            '''
        fenge_label_slice = largestConnectComponent(fenge_label_slice)
        jixian_people = sxzy_slice(fenge_label_slice)  # 计算软骨的极限，以裁剪软骨
        size1 = 160
        size2 = 192
        if size1 < 10 and size2 < 10:
            print('taixiao', jixian_people, i[:-7])
            continue
        if jixian_people[3] - jixian_people[2] > size2 or jixian_people[1] - jixian_people[0] > size1:
            print('taida', jixian_people, i[:-7])
            continue
        if jixian_people[1] - jixian_people[0] > max_shuzhou_hengzhou[0]:
            max_shuzhou_hengzhou[0] = jixian_people[1] - jixian_people[0]
        if jixian_people[3] - jixian_people[2] > max_shuzhou_hengzhou[1]:
            max_shuzhou_hengzhou[1] = jixian_people[3] - jixian_people[2]
        print(max_shuzhou_hengzhou)    #

        train_image = jietu(jixian_people, size1, size2, image_people, label_people)
        '''
        for jjj in range(train_image.shape[0]):
            # 显示结果
            plt.figure("jieguo_biaoqian_1")
            plt.imshow(train_image[jjj, 0, :, :])
            plt.pause(0.01)
            plt.close("jieguo_biaoqian_1")
        '''
        if train_image.shape[0] <= 5 or train_image.shape[0] >= 120:
            print(train_image.shape[0], i[:-7])
            continue
        zong.append(train_image)
        labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]

        shuzhou_xunhuan = int((size1 - (jixian_people[1] - jixian_people[0])) / 20)
        hengzhou_xunhuan = int((size2 - (jixian_people[3] - jixian_people[2])) / 20)
        '''
        if shuzhou_xunhuan > 1:
            for shuzhou_jishu in range(shuzhou_xunhuan - 1):
                jixian_people_shuzhou = jixian_people.copy()
                jixian_people_shuzhou[1] = jixian_people_shuzhou[1] - (shuzhou_jishu + 1) * 10
                image_shuzhou = jietu(jixian_people_shuzhou, size1, size2, image_people, label_people)
                zong.append(image_shuzhou)
                labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]

                jixian_people_shuzhou = jixian_people.copy()
                jixian_people_shuzhou[0] = jixian_people_shuzhou[0] + (shuzhou_jishu + 1) * 10
                image_shuzhou = jietu(jixian_people_shuzhou, size1, size2, image_people, label_people)
                zong.append(image_shuzhou)
                labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]
        '''
        '''
        if hengzhou_xunhuan > 1:
            for hengzhou_jishu in range(hengzhou_xunhuan - 1):
                jixian_people_hengzhou = jixian_people.copy()
                jixian_people_hengzhou[3] = jixian_people_hengzhou[3] - (hengzhou_jishu + 1) * 10
                image_hengzhou = jietu(jixian_people_hengzhou, size1, size2, image_people, label_people)
                zong.append(image_hengzhou)
                labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]

                jixian_people_hengzhou = jixian_people.copy()
                jixian_people_hengzhou[2] = jixian_people_hengzhou[2] + (hengzhou_jishu + 1) * 10
                image_hengzhou = jietu(jixian_people_hengzhou, size1, size2, image_people, label_people)
                zong.append(image_hengzhou)
                labels_batch_ndarry_zong[len(zong) - 1, 0] = labels_batch_ndarry[0]
        '''
    return zong, labels_batch_ndarry_zong

def main():
    zuzi = 'guguruangu_wai_hou'
    files_slice_train = os.listdir(data_dir_train)  # 列出所有的用于训练的数据
    random.shuffle(files_slice_train)

    files_slice_test = os.listdir(data_dir_test)  # 列出所有的test用于的数据
    random.shuffle(files_slice_test)

    files_slice_validate = os.listdir(data_dir_validate)  # 列出所有的用于的数据
    random.shuffle(files_slice_validate)

    '''
    train_dataset_temp, train_label_temp = datasets_loading(files_slice_train, data_dir_train, data_dir_train_label)
    zhengxu = list(range(len(train_dataset_temp)))
    random.shuffle(zhengxu)
    train_dataset = []
    train_label = np.zeros([train_label_temp.shape[0], 1])
    for j in range(len(train_dataset_temp)):
        train_dataset.append(train_dataset_temp[zhengxu[j]])
        train_label[j, 0] = train_label_temp[zhengxu[j], 0]
    np.savez(os.path.join(
        '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/dataset/guguruangu/train/',
        'train_' + zuzi), image=train_dataset, label=train_label)  # 保存乱序后的数据集
    del train_dataset_temp, train_label_temp, zhengxu
    gc.collect()

    test_dataset_temp, test_label_temp = datasets_loading(files_slice_test, data_dir_test, data_dir_test_label)
    zhengxu = list(range(len(test_dataset_temp)))
    random.shuffle(zhengxu)
    test_dataset = []
    test_label = np.zeros([test_label_temp.shape[0], 1])
    for j in range(len(test_dataset_temp)):
        test_dataset.append(test_dataset_temp[zhengxu[j]])
        test_label[j, 0] = test_label_temp[zhengxu[j], 0]
    np.savez(os.path.join(
        '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/dataset/guguruangu/test/',
        'test_' + zuzi), image=test_dataset, label=test_label)  # 保存乱序后的数据集
    del test_dataset_temp, test_label_temp, zhengxu
    gc.collect()

    validate_dataset_temp, validate_label_temp = datasets_loading(files_slice_validate, data_dir_validate, data_dir_validate_lable)
    zhengxu = list(range(len(validate_dataset_temp)))
    random.shuffle(zhengxu)
    validate_dataset = []
    validate_label = np.zeros([validate_label_temp.shape[0], 1])
    for j in range(len(validate_dataset_temp)):
        validate_dataset.append(validate_dataset_temp[zhengxu[j]])
        validate_label[j, 0] = validate_label_temp[zhengxu[j], 0]
    np.savez(os.path.join(
        '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/dataset/guguruangu/validate/',
        'validate_' + zuzi), image=validate_dataset, label=validate_label)  # 保存乱序后的数据集
    del validate_dataset_temp, validate_label_temp, zhengxu
    gc.collect()

    '''
    datasets_jinggu_train = np.load(os.path.join(
        '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/dataset/guguruangu/train/',
        'train_' + zuzi + '.npz'), allow_pickle=True)  # 加载train datasets
    train_dataset = datasets_jinggu_train['image']
    train_label = datasets_jinggu_train['label']
    datasets_jinggu_test = np.load(os.path.join(
        '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/dataset/guguruangu/test/',
        'test_' + zuzi + '.npz'), allow_pickle=True)  # 加载train datasets
    test_dataset = datasets_jinggu_test['image']
    test_label = datasets_jinggu_test['label']
    datasets_jinggu_train_nei = np.load(os.path.join(
        '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/dataset/guguruangu/validate/',
        'validate_' + zuzi + '.npz'), allow_pickle=True)  # 加载train datasets
    validate_dataset = datasets_jinggu_train_nei['image']
    validate_label = datasets_jinggu_train_nei['label']

    for chongfu in range(5):
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
            criterion = losses.BCEFocalLosswithLogits_guguruangu_wai_hou().cuda()

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
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

        max_auc = 0
        zhongduan = 0
        avg_acc_train = [0, 0, 0]
        avg_acc_val = [0, 0, 0]
        avg_acc_test = [0, 0, 0]
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' % (epoch, args.epochs))
            batch = 4
            net, avg_loss_train, sum_shuju_train, crect_sum_shuju_train, auc_train = train(model, optimizer, criterion,
                                                                                 train_dataset,
                                                                                 train_label, batch)

            avg_loss_val, sum_shuju_val, crect_sum_shuju_val, auc_val = val(net, criterion, validate_dataset,
                                                                                   validate_label, batch)

            avg_acc_train[0] = crect_sum_shuju_train[0] / sum_shuju_train[0]  # 总准确率
            avg_acc_train[1] = crect_sum_shuju_train[1] / sum_shuju_train[1]  # 1类准确率
            avg_acc_train[2] = crect_sum_shuju_train[2] / sum_shuju_train[2]  # 2类准确率

            avg_acc_val[0] = crect_sum_shuju_val[0] / sum_shuju_val[0]
            avg_acc_val[1] = crect_sum_shuju_val[1] / sum_shuju_val[1]
            avg_acc_val[2] = crect_sum_shuju_val[2] / sum_shuju_val[2]

            zhongduan = zhongduan + 1
            if min(avg_acc_val[1], avg_acc_val[2]) > max_auc:
                zhongduan = 0
                max_auc = min(avg_acc_val[1], avg_acc_val[2])
                torch.save(net, '/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/model/guguruangu/' + str(chongfu) + zuzi + '.pth')
                print("=> saved best model")
                break
            if zhongduan >= 5:
                break
            print(epoch, avg_loss_train, sum_shuju_train, crect_sum_shuju_train, auc_train, avg_acc_train)
            print(epoch, avg_loss_val, sum_shuju_val, crect_sum_shuju_val, auc_val, avg_acc_val)
            torch.cuda.empty_cache()

        print('Finished Training')
        model = torch.load('/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/model/guguruangu/' + str(chongfu) + zuzi + '.pth')  # 模型
        model = model.cuda()
        batch = 4
        avg_loss_test, sum_shuju_test, crect_sum_shuju_test, auc_test, all_label_test, all_out_test = test(model, criterion, test_dataset, test_label, batch)

        avg_acc_test[0] = crect_sum_shuju_test[0] / sum_shuju_test[0]
        avg_acc_test[1] = crect_sum_shuju_test[1] / sum_shuju_test[1]
        avg_acc_test[2] = crect_sum_shuju_test[2] / sum_shuju_test[2]
        print(avg_loss_test, sum_shuju_test, crect_sum_shuju_test, auc_test, avg_acc_test)

        #############画图部分
        fpr, tpr, threshold = metrics.roc_curve(all_label_test, all_out_test)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # 6.展示图片和保存
        np.savez(os.path.join('/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/result/guguruangu/AUC/',
            zuzi + str(chongfu)), all_out_test=all_out_test, all_label_test=all_label_test)  # 保存结果和标签
        plt.savefig('/media/ExtHDD02/csl/methods_data/biyelunwen/twentynine_clc/result/guguruangu/AUC/' + zuzi + str(chongfu) + '.jpg', dpi=300)


if __name__ == '__main__':
    main()