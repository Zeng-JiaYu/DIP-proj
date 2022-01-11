import cv2 as cv
import numpy as np
from skimage import morphology
import math as m
from numba import jit  # 转换为机器代码，加速运算
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Conv5_UI import Ui_Dialog as UI_Conv
import time


def morphy_binary(img, deal_Type):
    """根据用户的选择，对于图像做相应的二值形态学处理"""
    if img.shape[-1] == 3:
        pass
    cv_img = img
    q_dialog = QDialog()
    dlg = UI_Conv()
    dlg.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array(
            [[int(dlg.lineEdit1.text()), int(dlg.lineEdit2.text()), int(dlg.lineEdit3.text()),
              int(dlg.lineEdit4.text()), int(dlg.lineEdit5.text())],
             [int(dlg.lineEdit6.text()), int(dlg.lineEdit7.text()), int(dlg.lineEdit8.text()),
              int(dlg.lineEdit9.text()), int(dlg.lineEdit10.text())],
             [int(dlg.lineEdit11.text()), int(dlg.lineEdit12.text()), int(dlg.lineEdit13.text()),
              int(dlg.lineEdit14.text()), int(dlg.lineEdit15.text())],
             [int(dlg.lineEdit16.text()), int(dlg.lineEdit17.text()), int(dlg.lineEdit18.text()),
              int(dlg.lineEdit19.text()), int(dlg.lineEdit20.text())],
             [int(dlg.lineEdit21.text()), int(dlg.lineEdit22.text()), int(dlg.lineEdit23.text()),
              int(dlg.lineEdit24.text()), int(dlg.lineEdit25.text())]
             ])

        if deal_Type == 1:
            img, cv_img = erosion_binary(img, np_kernel)
        elif deal_Type == 2:
            img, cv_img = dilation_binary(img, np_kernel)
        elif deal_Type == 3:
            img, cv_img = open_binary(img, np_kernel)
        elif deal_Type == 4:
            img, cv_img = close_binary(img, np_kernel)
    return img, cv_img


def morphy_gray(img, deal_Type):
    """根据用户的选择，对于图像做相应的灰值形态学处理"""
    if img.shape[-1] == 3:
        pass
    q_dialog = QDialog()
    dlg = UI_Conv()
    dlg.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array(
            [[int(dlg.lineEdit1.text()), int(dlg.lineEdit2.text()), int(dlg.lineEdit3.text()),
              int(dlg.lineEdit4.text()), int(dlg.lineEdit5.text())],
             [int(dlg.lineEdit6.text()), int(dlg.lineEdit7.text()), int(dlg.lineEdit8.text()),
              int(dlg.lineEdit9.text()), int(dlg.lineEdit10.text())],
             [int(dlg.lineEdit11.text()), int(dlg.lineEdit12.text()), int(dlg.lineEdit13.text()),
              int(dlg.lineEdit14.text()), int(dlg.lineEdit15.text())],
             [int(dlg.lineEdit16.text()), int(dlg.lineEdit17.text()), int(dlg.lineEdit18.text()),
              int(dlg.lineEdit19.text()), int(dlg.lineEdit20.text())],
             [int(dlg.lineEdit21.text()), int(dlg.lineEdit22.text()), int(dlg.lineEdit23.text()),
              int(dlg.lineEdit24.text()), int(dlg.lineEdit25.text())]
             ])

        if deal_Type == 1:
            img = erosion_gray(img, np_kernel)
        elif deal_Type == 2:
            img = dilation_gray(img, np_kernel)
        elif deal_Type == 3:
            img = open_gray(img, np_kernel)
        elif deal_Type == 4:
            img = close_gray(img, np_kernel)
        elif deal_Type == 5:
            img = morphy_gray_edge(img, np_kernel)
    return img


def erosion_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值腐蚀，默认255像素点为目标
    *注意，传入的图像必须为二值图像, kernel为结构形状, 函数：XΘS={x|S+x∈X}"""
    time1 = time.time()  # 程序计时开始
    if img.shape[-1] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为二值图像
    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    # 手写二值图像腐蚀过程
    len = np_kernel.shape[0]
    half_len = int((len-1)/2)
    n = np.sum(np_kernel) * 255
    for y in range(half_len, rows - half_len):
        for x in range(half_len, cols - half_len):
            temp = np.sum(img[y - half_len:y + half_len + 1, x - half_len:x + half_len + 1] * np_kernel)
            if temp == n:
                new_img[y, x] = 255
            else:
                new_img[y, x] = 0
    time2 = time.time()  # 程序计时结束
    print("手写二值腐蚀运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像腐蚀过程
    np_kernel = np_kernel.astype(np.uint8)
    cv_img = cv.erode(img, np_kernel, iterations=1)
    time3 = time.time()  # 程序计时结束
    print("opencv二值腐蚀运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def dilation_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值膨胀，默认255像素点为目标
    *注意，传入的图像必须为二值图像, kernel为结构形状, 这里用的是腐蚀函数的对偶运算：
    X⊕S=∪{S+x|x∈X}=(X^c Θ S^v)^c"""
    time1 = time.time()  # 程序计时开始
    # 手写图像膨胀过程
    if img.shape[-1] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为二值图像
    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    len = np_kernel.shape[0]
    half_len = int((len - 1) / 2)
    for y in range(half_len, rows - half_len):
        for x in range(half_len, cols - half_len):
            temp = np.sum(img[y - half_len:y + half_len + 1, x - half_len:x + half_len + 1] * np_kernel)
            if temp >= 255:
                new_img[y, x] = 255
            else:
                new_img[y, x] = 0
    time2 = time.time()  # 程序计时结束
    print("手写二值膨胀运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像膨胀过程
    np_kernel = np_kernel.astype(np.uint8)
    cv_img = cv.dilate(img, np_kernel, iterations=1)
    time3 = time.time()  # 程序计时结束
    print("opencv二值膨胀运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def open_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值开，默认255像素点为目标(ww)
    *注意，传入的图像必须为二值图像，二值开函数：X○S=(XΘS)⊕S"""
    time1 = time.time()  # 程序计时开始
    # 手写图像二值开运算过程
    new_img, cv_img = erosion_binary(img, np_kernel)  # 先腐蚀
    new_img, cv_img = dilation_binary(new_img, np_kernel)  # 再膨胀
    time2 = time.time()  # 程序计时结束
    print("手写二值开运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像二值开运算过程
    cv_img = cv.morphologyEx(img, cv.MORPH_OPEN, np_kernel.astype(np.uint8))
    time3 = time.time()  # 程序计时结束
    print("opencv二值开运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def close_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值闭，默认255像素点为目标(ww)
        *注意，传入的图像必须为二值图像，二值闭函数：X·S=(X⊕S)ΘS"""
    time1 = time.time()  # 程序计时开始
    # 手写图像二值闭运算过程
    new_img, cv_img = dilation_binary(img, np_kernel)  # 先膨胀
    new_img, cv_img = erosion_binary(new_img, np_kernel)  # 再腐蚀
    time2 = time.time()  # 程序计时结束
    print("手写二值闭运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像二值闭运算过程
    cv_img = cv.morphologyEx(img, cv.MORPH_CLOSE, np_kernel.astype(np.uint8))
    time3 = time.time()  # 程序计时结束
    print("opencv二值闭运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def fast_thin(img, deal_Type):
    """*功能 : 根据传入的图像进行快速形态学细化，默认0像素点为背景"""
    time1 = time.time()  # 程序计时开始
    if img.shape[-1] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为二值图像
    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    new_img = np.zeros(img.shape, dtype=np.uint8)
    img[img == 255] = 1
    if deal_Type == 0:
        new_img = morphology.skeletonize(img)
        new_img = new_img.astype(np.uint8) * 255
    elif deal_Type == 1:
        skel, distance = morphology.medial_axis(img, return_distance=True)
        new_img = distance * skel
        new_img = new_img.astype(np.uint8) * 255
    time2 = time.time()  # 程序计时结束
    print("opencv细化算法算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


def erosion_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值腐蚀*注意，传入的图像必须为灰值图像,
       kernel为结构形状, 函数：(fΘb)(s,t)=min{f(s+x, t+y)}"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    if img.shape[-1] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为二值图像
    new_img = np.zeros(img.shape)  # 新建同原图大小一致的空图像
    # 手写灰值图像腐蚀过程
    len = np_kernel.shape[0]
    half_len = int((len - 1) / 2)
    for y in range(half_len, rows - half_len):
        for x in range(half_len, cols - half_len):
            temp = img[y - half_len:y + half_len + 1, x - half_len:x + half_len + 1] - np_kernel
            new_img[y, x] = np.min(temp)
    new_img = np.clip(new_img, 0, 255)  # 截取函数
    new_img = new_img.astype(np.uint8)  # 将数据类型转换为uint8格式
    time2 = time.time()  # 程序计时结束
    print("手写灰值腐蚀运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像腐蚀过程
    np_kernel = np_kernel.astype(np.uint8)
    cv_img = cv.erode(img, np_kernel, iterations=2)
    time3 = time.time()  # 程序计时结束
    print("opencv灰值腐蚀运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def dilation_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值腐蚀*注意，传入的图像必须为灰值图像,
       kernel为结构形状, 函数：(fΘb)(s,t)=min{f(s+x, t+y)}"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    if img.shape[-1] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为二值图像
    new_img = np.zeros(img.shape)  # 新建同原图大小一致的空图像
    # 手写灰值图像腐蚀过程
    len = np_kernel.shape[0]
    half_len = int((len - 1) / 2)
    for y in range(half_len, rows - half_len):
        for x in range(half_len, cols - half_len):
            temp = img[y - half_len:y + half_len + 1, x - half_len:x + half_len + 1] + np_kernel
            new_img[y, x] = np.max(temp)
    new_img = np.clip(new_img, 0, 255)  # 截取函数
    new_img = new_img.astype(np.uint8)  # 将数据类型转换为uint8格式
    time2 = time.time()  # 程序计时结束
    print("手写灰值膨胀运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像腐蚀过程
    np_kernel = np_kernel.astype(np.uint8)
    cv_img = cv.dilate(img, np_kernel, iterations=2)
    time3 = time.time()  # 程序计时结束
    print("opencv灰值膨胀运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def open_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值开，灰值开函数：X○S=(XΘS)⊕S"""
    time1 = time.time()  # 程序计时开始
    # 手写图像灰值开运算过程
    new_img = erosion_gray(img, np_kernel)  # 先腐蚀
    new_img = dilation_gray(new_img[0], np_kernel)  # 再膨胀
    time2 = time.time()  # 程序计时结束
    print("手写灰值开运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像灰值开运算过程
    cv_img = cv.morphologyEx(img, cv.MORPH_OPEN, np_kernel.astype(np.uint8), iterations=2)
    time3 = time.time()  # 程序计时结束
    print("opencv灰值开运算程序处理时间：%.3f毫秒" % ((time3 - time1) * 1000))
    return new_img[0], cv_img


def close_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值闭，灰值闭运算函数：X·S=(X⊕S)ΘS"""
    time1 = time.time()  # 程序计时开始
    # 手写图像灰值闭运算过程
    new_img = dilation_gray(img, np_kernel)  # 先膨胀
    new_img = erosion_gray(new_img[0], np_kernel)  # 再腐蚀
    time2 = time.time()  # 程序计时结束
    print("手写灰值闭运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像灰值闭运算过程
    cv_img = cv.morphologyEx(img, cv.MORPH_CLOSE, np_kernel.astype(np.uint8), iterations=2)
    time3 = time.time()  # 程序计时结束
    print("opencv灰值闭运算程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img[0], cv_img


def morphy_gray_edge(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值边缘求取，函数为：g(X)=(X⊕S)-(XΘS)"""
    time1 = time.time()  # 程序计时开始
    # 手写图像灰值形态学边缘求取过程
    ero_img, cv_ero_img = erosion_gray(img, np_kernel)  # 腐蚀
    dil_img, cv_dil_img = dilation_gray(img, np_kernel)  # 膨胀
    new_img = cv.absdiff(dil_img, ero_img)
    time2 = time.time()  # 程序计时结束
    print("手写灰值形态学边缘程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv图像灰值形态学边缘求取过程
    ero_img = cv.erode(img, np_kernel.astype(np.uint8), iterations=2)  # 腐蚀
    dil_img = cv.dilate(img, np_kernel.astype(np.uint8), iterations=2)  # 膨胀
    cv_img = cv.absdiff(dil_img, ero_img)
    time3 = time.time()  # 程序计时结束
    print("opencv灰值形态学边缘程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img
