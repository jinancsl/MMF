import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from skimage import morphology

import SimpleITK as sitk
from PIL import Image
import cv2
import random
import xlrd
import pandas
import openpyxl

def people_label_ruangu_biaozhun():
    #duqu gu
    excel_label_dir = '/home/csl/Downloads/cartilage MOAKS.xlsx'
    book0 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='sr')
    sheets0 = book0.values
    #sheets0 = np.delete(sheets0, 2, axis=1)

    book1 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='sl')
    sheets1 = book1.values
    #sheets1 = np.delete(sheets1, 2, axis=1)


    book2 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='mr')
    sheets2 = book2.values
    #sheets2 = np.delete(sheets2, 2, axis=1)

    book3 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='ml')
    sheets3 = book3.values
    #sheets3 = np.delete(sheets3, 2, axis=1)

    sheets = np.zeros([sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0] + sheets3.shape[0], 16])
    sheets[0:sheets0.shape[0], :] = sheets0
    sheets[sheets0.shape[0]:sheets0.shape[0] + sheets1.shape[0], :] = sheets1
    sheets[sheets0.shape[0] + sheets1.shape[0]:sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0], :] = sheets2
    sheets[sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0]:sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0] + sheets3.shape[0], :] = sheets3
    return sheets


def people_label_gu_biaozhun():
    # duqu gu
    excel_label_dir = '/home/csl/Downloads/BML MOAKS_my.xlsx'
    book0 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='sr')
    sheets0 = book0.values
    # sheets0 = np.delete(sheets0, 2, axis=1)

    book1 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='sl')
    sheets1 = book1.values
    # sheets1 = np.delete(sheets1, 2, axis=1)

    book2 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='mr')
    sheets2 = book2.values
    # sheets2 = np.delete(sheets2, 2, axis=1)

    book3 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='ml')
    sheets3 = book3.values
    # sheets3 = np.delete(sheets3, 2, axis=1)

    sheets = np.zeros([sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0] + sheets3.shape[0], 49])
    sheets[0:sheets0.shape[0], :] = sheets0
    sheets[sheets0.shape[0]:sheets0.shape[0] + sheets1.shape[0], :] = sheets1
    sheets[sheets0.shape[0] + sheets1.shape[0]:sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0], :] = sheets2
    sheets[
    sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0]:sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0] +
                                                           sheets3.shape[0], :] = sheets3
    return sheets

def people_label_shimei_ruangu():
    # duqu gu
    ###
    excel_label_dir = '/home/csl/Downloads/kexing_gai.xlsx'
    book0 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='ruangu')
    sheets0 = book0.values
    # sheets0 = np.delete(sheets0, 2, axis=1)
    return sheets0

def people_label_shimei_gu():
    # duqu gu
    ###
    excel_label_dir = '/home/csl/Downloads/kexing_gai.xlsx'
    book0 = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='gu')
    sheets0 = book0.values
    # sheets0 = np.delete(sheets0, 2, axis=1)
    return sheets0

def shimei_tongji():
    jinbiaozhun_ruangu = people_label_ruangu_biaozhun()
    jinbiaozhun_gu = people_label_gu_biaozhun()
    kexing_ruangu = people_label_shimei_ruangu()
    kexing_gu = people_label_shimei_gu()
    '''
    jinbiaozhun_ruangu[jinbiaozhun_ruangu > 0.1] = 1
    jinbiaozhun_gu[jinbiaozhun_gu > 0.1] = 1
    
    jinbiaozhun_ruangu[jinbiaozhun_ruangu < 0.1] = 0
    jinbiaozhun_gu[jinbiaozhun_gu < 0.1] = 0
    '''
    for i in range(kexing_gu.shape[0]):
        for j in range(jinbiaozhun_ruangu.shape[0]):
            if kexing_ruangu[i, 1] == jinbiaozhun_ruangu[j, 0] and kexing_ruangu[i, 2] == jinbiaozhun_ruangu[j, 1]:
                jishu = 0
                for i_2 in range(14):
                    if jinbiaozhun_ruangu[j, i_2+2] > 0.1:
                        jinbiaozhun_ruangu_temp = 1
                    else:
                        jinbiaozhun_ruangu_temp = 0
                    if kexing_ruangu[i, i_2+3] == jinbiaozhun_ruangu_temp:
                        jishu = jishu + 1
        '''
        if jishu == 13:
            print(kexing_ruangu[i, 1])
        '''

        for j in range(jinbiaozhun_gu.shape[0]):
            if kexing_gu[i, 1] == jinbiaozhun_gu[j, 0] and kexing_gu[i, 2] == jinbiaozhun_gu[j, 1]:
                jishu2 = 0
                jinbiaozhu_tongji = []
                for i_2 in range(15):
                    if jinbiaozhun_gu[j, i_2 + 4] + jinbiaozhun_gu[j, i_2 + 4+15] + jinbiaozhun_gu[j, i_2 + 4+30] > 0.1:
                        jinbiaozhun_ruangu_temp = 1
                    else:
                        jinbiaozhun_ruangu_temp = 0
                    jinbiaozhu_tongji.append(jinbiaozhun_ruangu_temp)
                    if kexing_gu[i, i_2 + 3] == jinbiaozhun_ruangu_temp:
                        jishu2 = jishu2 + 1
        if jishu >= 13 and jishu2 >= 14:
            print(kexing_ruangu[i, 1], jinbiaozhu_tongji)

if __name__ == '__main__':
    shimei_tongji()