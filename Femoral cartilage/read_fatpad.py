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


def people_label():
    excel_label_dir = '/media/ExtHDD02/csl/methods_data/biyelunwen/twentyninefenqu/dataset/fenlei29fenqu/cartilage.xlsx'
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

    sheets = np.zeros([sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0] + sheets3.shape[0], sheets0.shape[1]])
    sheets[0:sheets0.shape[0], :] = sheets0
    sheets[sheets0.shape[0]:sheets0.shape[0] + sheets1.shape[0], :] = sheets1
    sheets[sheets0.shape[0] + sheets1.shape[0]:sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0], :] = sheets2
    sheets[sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0]:sheets0.shape[0] + sheets1.shape[0] + sheets2.shape[0] + sheets3.shape[0], :] = sheets3
    return sheets



if __name__ == '__main__':
    sheets = people_label()