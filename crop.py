from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import os
import numpy as np
import h5py
from tqdm import tqdm
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 要裁剪图像的大小
img_w = 264
img_h = 264


# 读取路径下图片的名称
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L


image_sets = file_name('QB_train');  # 图片存贮路径
# gt_sets = file_name('GF_train');


# 添加噪声
def add_noise(img):
    drawObject = ImageDraw.Draw(img)
    for i in range(250):  # 添加点噪声
        temp_x = np.random.randint(0, img.size[0])
        temp_y = np.random.randint(0, img.size[1])
        drawObject.point((temp_x, temp_y), fill="white")  # 添加白色噪声点,噪声点颜色可变
    return img


# 色调增强
def random_color(img):
    img = ImageEnhance.Color(img)
    img = img.enhance(2)
    return img


def data_augment(src_roi,gt_roi):
    # 图像和标签同时进行90，180，270旋转
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(90)
        gt_roi = gt_roi.rotate(90)
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(180)
        gt_roi = gt_roi.rotate(180)
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(270)
        gt_roi = gt_roi.rotate(270)
    # 图像和标签同时进行竖直旋转
    if np.random.random() < 0.25:
        src_roi = src_roi.transpose(Image.FLIP_LEFT_RIGHT)
        gt_roi = gt_roi.transpose(Image.FLIP_LEFT_RIGHT)
    # 图像和标签同时进行水平旋转
    if np.random.random() < 0.25:
        src_roi = src_roi.transpose(Image.FLIP_TOP_BOTTOM)
        gt_roi = gt_roi.transpose(Image.FLIP_TOP_BOTTOM)
    # 图像进行高斯模糊
    # if np.random.random() < 0.25:
    #     src_roi = src_roi.filter(ImageFilter.GaussianBlur)
    # 图像进行色调增强
    if np.random.random() < 0.25:
        src_roi = random_color(src_roi)
    # 图像加入噪声
    # if np.random.random() < 0.2:
    #     src_roi = add_noise(src_roi)
    return src_roi, gt_roi
#
#
# # image_num：增广之后的图片数据
def creat_dataset(image_num=2, mode='original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 1
    # print(len(image_sets))

    for i in tqdm(range(1)):
        count = 0
        ##读入待裁剪原始图像
        src_img = Image.open('./QuickBird/pan/1.tif')
        gt_img = Image.open('./QuickBird/ms/1.tif')
        # 对图像进行随机裁剪，这里大小为256*256
        while count < 20000:
            width1 = random.randint(0, gt_img.size[0] - img_w)
            height1 = random.randint(0, gt_img.size[1] - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h
            width1_src = (width1-0)*4
            height1_src = (height1-0)*4
            width2_src = (width1-0)*4 + (img_w-0)*4
            height2_src = (height1-0)*4 + (img_h-0)*4

            src_roi = src_img.crop((width1_src, height1_src, width2_src, height2_src))
            gt_roi = gt_img.crop((width1, height1, width2, height2))

            if mode == 'augment':
                src_roi, gt_roi = data_augment(src_roi, gt_roi)

            if count == 0:
                pan = h5py.File('QB_train/pan.h5', 'w')
                pan.create_dataset('pan' + str(count), data=src_roi)
                pan.close()
                gt = h5py.File('QB_train/gt.h5', 'w')
                gt.create_dataset('gt' + str(count), data=gt_roi)
                gt.close()
            else:
                fpan = h5py.File('QB_train/pan.h5', 'a')
                fpan.create_dataset('pan'+str(count), data=src_roi)
                fpan.close()
                gt = h5py.File('QB_train/gt.h5', 'a')
                gt.create_dataset('gt' + str(count), data=gt_roi)
                gt.close()
            if count%100==0:
                print('count finished',count)
            count = count + 1
            g_count += 1
