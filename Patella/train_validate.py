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

device = torch.device('cuda')

NII_train_jinggu_fenlei = '/media/chen/DATA2/knee_dataset/gu_gusui_fenlei/dataset_nfyk_zhengli/IOA'
NII_train_jinggu_fenlei_test = '/media/chen/DATA2/knee_dataset/gu_gusui_fenlei/dataset_nfyk_zhengli/test'

data_dir = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/xi/train/image'
data_dir_lable = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/xi/train/label'

data_dir_test = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/xi/test/image'
data_dir_test_label = '/media/chen/DATA2/knee_dataset/gu_gusui/fenlei/dataset_my/xi/test/label'

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, prob, target):
        epsilon = 1e-10
        loss = - (target * torch.log(prob + epsilon) + (1 - target) * (torch.log(1 - prob)))
        loss = torch.sum(loss) / torch.numel(target)
        return loss

def Normalize(data):
    data_normalize = data.copy()
    for i in range(data.shape[2]):
        data_normalize[:, :, data.shape[2] - i - 1] = data[:, :, i]
    return data_normalize

def Normalize_image(data):
    data_normalize = data.copy()
    max = data_normalize.max()
    min = data_normalize.min()
    data_normalize = (data_normalize - min) / (max - min)
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
    '''
    image_arr_all[image_arr_all == 4] = 0
    image_arr_all[image_arr_all == 5] = 0
    image_arr_all[image_arr_all == 6] = 0
    image_arr_all[image_arr_all == 10] = 1
    image_arr_all[image_arr_all == 11] = 1
    image_arr_all[image_arr_all == 12] = 1

    image_arr_all[image_arr_all == 1] = 0
    image_arr_all[image_arr_all == 2] = 0
    image_arr_all[image_arr_all == 3] = 0
    image_arr_all[image_arr_all == 7] = 0
    image_arr_all[image_arr_all == 8] = 0
    image_arr_all[image_arr_all == 9] = 0
    image_arr_all[image_arr_all == 13] = 0
    '''
    image_arr_all_temp[image_arr_all == 11] = 1
    return image_arr_all_temp

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

def sxzy(out_slice):
    ########处理每一个切片的结果
    jixian_people = [0, 0, 0, 0]
    test_label_people_1 = out_slice.transpose(1, 0, 2)
    for j_1 in range(test_label_people_1.shape[0]):
        if test_label_people_1[j_1, :, :].sum() > 0:
            jixian_people[0] = j_1  # 最高点
            break

    for j_2 in range(test_label_people_1.shape[0]):
        if test_label_people_1[test_label_people_1.shape[0] - j_2 - 1, :, :].sum() > 0:
            jixian_people[1] = test_label_people_1.shape[0] - j_2 - 1
            break  # 最低点

    test_label_people_2 = out_slice.transpose(2, 1, 0)
    for j_3 in range(test_label_people_2.shape[0]):
        if test_label_people_2[j_3, :, :].sum() > 0:
            jixian_people[2] = j_3
            break  # 最左点

    for j_4 in range(test_label_people_2.shape[0]):
        if test_label_people_2[test_label_people_2.shape[1] - j_4 - 1, :, :].sum() > 0:
            jixian_people[3] = test_label_people_2.shape[1] - j_4 - 1
            break  # 最右点
    return jixian_people

def jingguruangu_zuoyou_jixian(image):   #计算胫骨软骨左右极限
    x = image.transpose(2, 1, 0).copy()
    jixian = [0, 0]
    for i in range(x.shape[2]):     #求右极限
        x2 = x[:, :, x.shape[2] - 1 - i].sum()
        if x2 > 0:
            jixian[1] = x.shape[2] - i - 1
            break

    for i in range(x.shape[2]):     #求左极限
        x2 = x[:, :, i].sum()
        if x2 > 0:
            jixian[0] = i
            break

    return jixian

def train(net, optimizer, criterion, files_slice):
    net.train()

    cum_loss = 0.0
    cum_acc = np.zeros([3])
    cum_acc_crect = np.zeros([3])
    sheets2 = people_label()    #标签表, ndarry格式
    sheets3 = []
    sheets4 = []
    for i in range(sheets2.shape[0]):
        people_num = sheets2[i, 0]  # 是哪个人
        people_leg = sheets2[i, 1]  # 是哪条腿
        xulie = ''
        if people_leg == 1:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_RIGHT'
        if people_leg == 2:
            xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_LEFT'
        sheets3.append(xulie)
        sheets4.append(sheets2[2])

    jishu1 = 0
    batch = 50  # 每次训练的数量
    zong = []
    labels_batch_ndarry = np.zeros([1])
    labels_batch_ndarry_zong = np.zeros([batch, 1])
    for i in files_slice:

        #####读取用于训练的三维图像
        if os.path.exists(os.path.join(NII_train_jinggu_fenlei, str(i[0:7]),
                                       str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT')):  # 如果没有这个人，则跳过这一次循环，否则读取相应的序列
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT'
        elif os.path.exists(os.path.join(NII_train_jinggu_fenlei, str(i[0:7]),
                                         str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT')):
            save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT'
        else:
            raise Exception("不存在的图像文件")

        image_people_temp = read_niigz(os.path.join(data_dir, str(i)))    #读取每个人的三维图像
        fenge_label = read_niigz_fenge_label(os.path.join(data_dir_lable, str(i)))     #读取图像的分割标签

        ######
        fenge_label = largestConnectComponent(fenge_label)
        fenge_label_slice = np.zeros([fenge_label.shape[1], fenge_label.shape[2]])    #获取软骨的范围
        for gh in range(fenge_label.shape[0]):
            fenge_label_slice[fenge_label[gh, :, :] == 1] = 1
        jixian_people = sxzy_slice(fenge_label_slice)     #计算软骨的极限，以裁剪软骨
        jixian_people_zuobiao = [0, 0, 0, 0]
        size1 = jixian_people[3] - jixian_people[2]
        size2 = size1
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
                zong_temp.append(image_people_temp[u_10, jixian_people_zuobiao[0]:jixian_people_zuobiao[1], jixian_people_zuobiao[2]:jixian_people_zuobiao[3]])
        image_people = np.zeros([len(zong_temp), 128, 128])
        for u_9 in range(len(zong_temp)):
            image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (128, 128), interpolation = cv2.INTER_NEAREST)
        image_people = Normalize_image(image_people)

        #image_people = transform.resize(image_people, (batch, 128, 128), order=1)

        ###################读取标签
        for j_9 in range(len(sheets3)):
            if save_name == sheets3[j_9]:
                label_people_temp = sheets2[j_9, :]
                break

        if label_people_temp[5]+label_people_temp[12]+label_people_temp[19] > 0:
            labels_batch_ndarry[0] = 1
        else:
            labels_batch_ndarry[0] = 0
        ###########################

        train_image = np.zeros([image_people.shape[0], 1, 128, 128])
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

        if len(zong) == batch:
            optimizer.zero_grad()  # 优化器梯度归零
            losses = 0
            for j_6 in range(len(zong)):
                ##########
                inputs = torch.from_numpy(zong[j_6]).cuda().float()
                labels_batch = torch.from_numpy(labels_batch_ndarry_zong[j_6, :]).cuda().float()

                # forward + backward + optimize
                outputs = net(inputs.clone())
                outputs = outputs.squeeze(dim=-1)
                labels_batch = labels_batch.squeeze(dim=-1)
                loss = criterion(outputs, labels_batch)
                losses = loss + losses

                acc1, correct_sum_single = utils.compute_accuracy(
                    outputs,
                    labels_batch,
                    augmentation=False,
                    topk=(1, 1))  # acc1分别表示[总图像数、1类图像数、2类图像数]， correct_sum_single分别表示[分类正确的图像总数、1类分类正确数、2类分类正确数]
                cum_loss = cum_loss + loss
                cum_acc = cum_acc + acc1
                cum_acc_crect = cum_acc_crect + correct_sum_single
                jishu1 = jishu1 + 1
            optimizer.zero_grad()  # 清空过往梯度；
            losses.backward()  # 反向传播，计算当前梯度；
            optimizer.step()  # 根据梯度更新网络参数
            zong = []
    avg_loss = cum_loss / (jishu1)

    return net, avg_loss, cum_acc_crect, cum_acc


def val(net, criterion, files_slice):
    net.eval()
    with torch.no_grad():
        cum_loss = 0.0
        cum_acc = np.zeros([3])
        cum_acc_crect = np.zeros([3])
        sheets2 = people_label()  # 标签表, ndarry格式
        sheets3 = []
        sheets4 = []
        for i in range(sheets2.shape[0]):
            people_num = sheets2[i, 0]  # 是哪个人
            people_leg = sheets2[i, 1]  # 是哪条腿
            xulie = ''
            if people_leg == 1:
                xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_RIGHT'
            if people_leg == 2:
                xulie = str(int(people_num)) + '-00m-SAG_3D_DESS_LEFT'
            sheets3.append(xulie)
            sheets4.append(sheets2[2])

        jishu1 = 0
        batch = 25  # 每次训练的数量
        zong = []
        labels_batch_ndarry = np.zeros([1])
        labels_batch_ndarry_zong = np.zeros([batch, 1])
        for i in files_slice:

            #####读取用于训练的三维图像
            if os.path.exists(os.path.join(NII_train_jinggu_fenlei_test, str(i[0:7]),
                                           str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT')):  # 如果没有这个人，则跳过这一次循环，否则读取相应的序列
                save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_RIGHT'
            elif os.path.exists(os.path.join(NII_train_jinggu_fenlei_test, str(i[0:7]),
                                             str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT')):
                save_name = str(i[0:7]) + '-00m-SAG_3D_DESS_LEFT'
            else:
                raise Exception("不存在的图像文件")

            image_people_temp = read_niigz(os.path.join(data_dir_test, str(i)))  # 读取每个人的三维图像
            fenge_label = read_niigz_fenge_label(os.path.join(data_dir_test_label, str(i)))  # 读取图像的分割标签

            ######
            fenge_label = largestConnectComponent(fenge_label)
            fenge_label_slice = np.zeros([fenge_label.shape[1], fenge_label.shape[2]])  # 获取软骨的范围
            for gh in range(fenge_label.shape[0]):
                fenge_label_slice[fenge_label[gh, :, :] == 1] = 1
            jixian_people = sxzy_slice(fenge_label_slice)  # 计算软骨的极限，以裁剪软骨
            jixian_people_zuobiao = [0, 0, 0, 0]
            size1 = jixian_people[3] - jixian_people[2]
            size2 = size1
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
            image_people = np.zeros([len(zong_temp), 128, 128])
            for u_9 in range(len(zong_temp)):
                image_people[u_9, :, :] = cv2.resize(zong_temp[u_9], (128, 128), interpolation = cv2.INTER_NEAREST)
            image_people = Normalize_image(image_people)

            # image_people = transform.resize(image_people, (batch, 128, 128), order=1)

            ###################读取标签
            for j_9 in range(len(sheets3)):
                if save_name == sheets3[j_9]:
                    label_people_temp = sheets2[j_9, :]
                    break

            if label_people_temp[5] + label_people_temp[12] + label_people_temp[19] > 0:
                labels_batch_ndarry[0] = 1
            else:
                labels_batch_ndarry[0] = 0
            ###########################

            train_image = np.zeros([image_people.shape[0], 1, 128, 128])
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

            if len(zong) == batch:
                losses = 0
                for j_6 in range(len(zong)):
                    ##########
                    inputs = torch.from_numpy(zong[j_6]).cuda().float()
                    labels_batch = torch.from_numpy(labels_batch_ndarry_zong[j_6, :]).cuda().float()

                    # forward + backward + optimize
                    outputs = net(inputs.clone())
                    outputs = outputs.squeeze(dim=-1)
                    labels_batch = labels_batch.squeeze(dim=-1)
                    loss = criterion(outputs, labels_batch)
                    losses = loss + losses

                    acc1, correct_sum_single = utils.compute_accuracy(
                        outputs,
                        labels_batch,
                        augmentation=False,
                        topk=(1, 1))  # acc1分别表示[总图像数、1类图像数、2类图像数]， correct_sum_single分别表示[分类正确的图像总数、1类分类正确数、2类分类正确数]
                    cum_loss = cum_loss + loss
                    cum_acc = cum_acc + acc1
                    cum_acc_crect = cum_acc_crect + correct_sum_single
                    jishu1 = jishu1 + 1

                zong = []

        avg_loss = cum_loss / (jishu1)
        return avg_loss, cum_acc_crect, cum_acc