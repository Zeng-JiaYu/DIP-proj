import cv2 as cv
import time
import numpy as np
from PyQt5.QtWidgets import QMessageBox


def warning(img):
    if img.shape[-1] != 3:
        msg_box = QMessageBox(QMessageBox.Warning, '警告', '请选择彩色图像！')
        msg_box.exec_()
        return -1


def color_deal(img, deal_Type):
    """根据用户的选择，对于图像做相应的处理"""
    new_img = np.zeros(img.shape, dtype=np.uint8)
    cv_img = np.zeros(img.shape, dtype=np.uint8)
    if img.shape[-1] != 3 and deal_Type != 1:
        pass
    if deal_Type == 1:
        # 手写图像取反色处理
        time1 = time.time()
        for row in range(img.shape[0]):  # 遍历每一行
            for col in range(img.shape[1]):  # 遍历每一列
                new_img[row][col] = 255 - img[row][col]
        time2 = time.time()
        print("\n数据检索遍历时间：", (time2 - time1) * 1000)
        # opencv求反色函数
        cv_img = cv.bitwise_not(img)
        time3 = time.time()
        print("opencv遍历时间：", (time3 - time2) * 1000)
    elif deal_Type == 2:
        if warning(img) != -1:
            cv_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif deal_Type == 3:
        if warning(img) != -1:
            cv_img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    elif deal_Type == 4:
        if warning(img) != -1:
            cv_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    elif deal_Type == 5:
        if warning(img) != -1:
            cv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    else:
        pass
    return new_img, cv_img


def img_sample(img, val):
    """根据传入的采样间隔参数，进行图像采样,
    *img:传入带采样图像; iv(interval):采样间隔参数"""
    # 获取图像的长，宽
    newImg = img[0:img.shape[0]:val, 0:img.shape[1]:val]
    return newImg


def img_quanty(img, q_Size):
    """根据传入的量化值，进行图像量化,
    *img:传入带采样图像; q_Size:量化范围"""
    # 获取图像的长，宽
    height = img.shape[0]
    width = img.shape[1]
    # 新图像分配内存
    newImg = np.zeros((height, width), np.uint8)
    # 均匀量化
    border = range(0, 255, q_Size)
    # 计算量化值
    for h in range(height):
        for w in range(width):
            for i in range(len(border)):
                if img[h, w] < border[i]:
                    newImg[h, w] = border[i]
                    break
    return newImg
