import math as m
import time
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QDialog
from matplotlib import pyplot as plt
from numba import jit  # 转换为机器代码，加速运算
from Experiment_3.Conv_UI import Ui_Dialog as UI_Conv


def edge_detect(img, deal_Type):
    """根据用户的选择，对于图像做相应的边缘检测处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = prewitt(img)  # prewitt算子
    elif deal_Type == 2:
        img = sobel(img)  # sobel算子
    elif deal_Type == 3:
        img = log(img)  # log算子
    elif deal_Type == 4:
        img = canny(img)  # canny算子
    elif deal_Type == 5:
        img = custom(img)  # 自定义算子
    return img


def custom(img):
    """自定义算子"""
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Conv()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array(
            [[float(dialog.lineEdit1.text()), float(dialog.lineEdit2.text()), float(dialog.lineEdit3.text())],
             [float(dialog.lineEdit4.text()), float(dialog.lineEdit5.text()), float(dialog.lineEdit6.text())],
             [float(dialog.lineEdit7.text()), float(dialog.lineEdit8.text()), float(dialog.lineEdit9.text())]])
        time1 = time.time()  # 程序计时开始
        # 手写算法实现
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为二值图像
        new_img = convolution_2(img, np_kernel)  # 卷积
        new_img = np.clip(new_img, 0, 255)  # 截取函数
        new_img = new_img.astype(np.uint8)  # 将数据类型转换为uint8格式
        time2 = time.time()  # 程序计时结束
        temp = cv.filter2D(img, cv.CV_16S, np_kernel)
        cv_img = cv.convertScaleAbs(temp)  # 图像融合
        time3 = time.time()  # 程序计时结束
        print("手写自定义算子处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
        print("opencv自定义算子处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


@jit(nopython=True)
def convolution(img, core):
    """双方向卷积"""
    len = core.shape[1]
    half_len = (len - 1) / 2
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    rows, cols = img.shape[:2]  # 获取宽和高
    for row in range(half_len, rows - half_len - 1):
        for col in range(half_len, cols - half_len - 1):
            gx = np.multiply(img[row: row + len, col: col + len], core[0]).sum()  # 点对点相乘后进行累加
            gy = np.multiply(img[row: row + len, col: col + len], core[1]).sum()  # 点对点相乘后进行累加
            new_img[row][col] = np.abs(gx) + np.abs(gy)
    return new_img


def prewitt(img):
    """*功能 : 根据prewitt对应的卷积模板，对图像进行边缘检测
    *注意，这里只引入水平和竖直两个方向边缘检测卷积模板"""
    time1 = time.time()  # 程序计时开始
    prw_convs = np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                          [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]], dtype=int)
    # 图像遍历, 求取prewitt边缘
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_img = convolution(img, prw_convs)
    new_img = np.clip(new_img, 0, 255)
    time2 = time.time()  # 程序计时结束
    print("prewitt算子边缘检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    # cv程序编写
    x = cv.filter2D(img, cv.CV_16S, prw_convs[0])
    y = cv.filter2D(img, cv.CV_16S, prw_convs[1])
    # 图像融合
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    cv_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    time3 = time.time()  # 程序计时结束
    print("prewitt算子边缘检测CV程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def sobel(img):
    """*功能 : 根据sobel对应的卷积模板，对图像进行边缘检测
       *注意，这里只引入水平和竖直两个方向边缘检测卷积模板"""
    time1 = time.time()  # 程序计时开始
    prw_convs = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=int)
    # 图像遍历, 求取sobel边缘
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_img = convolution(img, prw_convs)
    new_img = np.clip(new_img, 0, 255)
    time2 = time.time()  # 程序计时结束
    print("sobel算子边缘检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    # cv程序编写
    x = cv.filter2D(img, cv.CV_16S, prw_convs[0])
    y = cv.filter2D(img, cv.CV_16S, prw_convs[1])
    # 图像融合
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    cv_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    time3 = time.time()  # 程序计时结束
    print("sobel算子边缘检测CV程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


@jit(nopython=True)
def convolution_2(img, core):
    len = core.shape[1]
    half_len = int((len - 1) / 2)
    new_img = np.zeros(img.shape)  # 新建同原图大小一致的空图像
    rows, cols = img.shape[:2]  # 获取宽和高
    for row in range(half_len, rows - len + 1):
        for col in range(half_len, cols - len + 1):
            img_temp = img[row-half_len: row + half_len+1, col-half_len: col + half_len+1]
            new_img[row][col] = np.multiply(img_temp, core).sum()
    return new_img


def log(img):
    """*功能 : 根据LOG算子对应的卷积模板，对图像进行边缘检测"""
    time1 = time.time()  # 程序计时开始
    prw_conv = np.array([[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4],
                         [-4, 0, 8, 0, -4], [-2, -4, -4, -4, -2]], dtype=int)
    # 图像遍历, 求取LOG边缘
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_img = convolution_2(img, prw_conv)
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype(np.uint8)
    time2 = time.time()  # 程序计时结束
    print("log算子边缘检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    # cv程序编写
    dst = cv.Laplacian(img, cv.CV_16S, ksize=5)
    cv_img = cv.convertScaleAbs(dst)
    time3 = time.time()  # 程序计时结束
    print("log算子边缘检测CV程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def canny(img):
    """*功能 : canny算子*"""
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    cv_img = cv.Canny(img, 100, 200)
    return new_img, cv_img


def thresh_otus(hist):
    max_devi = 0  # 最大方差 = 0
    for color in range(1, 256):
        w0 = np.sum(hist[:color])
        u0 = np.sum(np.multiply(np.arange(1, color + 1), hist[:color].transpose())) / w0  # 平均值
        w1 = 1 - w0
        u1 = np.sum(np.multiply(np.arange(color + 1, 257), hist[color:].transpose())) / w1
        devi = w0 * w1 * (u1 - u0) * (u1 - u0)
        if devi > max_devi:
            max_devi = devi
            max_T = color
    return max_T


def otsu(img, jug):
    """*功能：大津阈值分割，求取直方图数组，根据类间方差最大原理自动选择阈值，
    *注意：只处理灰度图像"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([img], [0], None, [256], [0, 255])  # 直接用cv函数计算直方图，hist: numpy格式
    hist = hist / (rows * cols)  # 归一化
    T = thresh_otus(hist)  # 计算阈值
    for row in range(rows):
        for col in range(cols):
            if img[row][col] > T:
                new_img[row][col] = 255
            else:
                new_img[row][col] = 0
    time2 = time.time()  # 程序计时结束
    print("大津阈值手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    # opencv大津阈值分割
    max_t, cv_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    time3 = time.time()  # 程序计时结束
    print("大津阈值cv程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    if jug:
        plt.plot(hist, color="r", label="otsu value in histogram")
        plt.xlim([0, 256])
        plt.axvline(max_t, color='green')  # 在直方图中绘制出阈值位置
        plt.legend()  # 用于给图像加图例，各种符号和颜色所代表内容与指标的说明
        plt.show()
    return new_img, cv_img


def hough_detect(img, deal_Type):
    """根据用户的选择，对于图像做相应的图像平滑处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = line_detect(img, 2)
    elif deal_Type == 2:
        img = circle_detect(img)
    return img


def hough_transform(img):
    """根据传入的图像，求取目标点对应的hough域，公式：ρ = x cos θ + y sin θ
    注：默认图像中255点为目标点"""
    rows, cols = img.shape[:2]  # 获取宽和高
    hg_rows, hg_cols = 180, int(m.sqrt(cols * cols + rows * rows))
    hough_img = np.zeros((181, hg_cols), dtype=np.int)  # 新建hough域，全为0值
    location = np.where(img == 0)
    for i in range(np.size(location, 1)):
        for rot in range(-90, 91):
            rot_temp = m.radians(rot)  # 将旋转角度从度转到弧度
            p = int(location[1][i] * m.cos(rot_temp) + location[0][i] * m.sin(rot_temp))
            hough_img[rot + 90][p] += 1
    return hough_img


def line_detect(img, num):
    """*功能 : 通过hough变换检测直线，num:需检测直线的条数"""
    time1 = time.time()  # 程序计时开始
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hough = hough_transform(gray_img)
    new_img = img.copy()
    rows, cols = img.shape[:2]  # 获取原图宽和高
    for i in range(num):
        location = np.where(hough == np.max(hough))
        rot, p = int(location[0]) - 90, int(location[1])
        pt_start = (0, int(p / m.sin(m.radians(rot))))  # 绘制直线起点
        pt_end = (cols, int((p - cols * m.cos(m.radians(rot))) / m.sin(m.radians(rot))))  # 绘制直线终点
        cv.line(new_img, pt_start, pt_end, (0, 0, 255), 1)
        hough[int(location[0]) - 1:int(location[0]) + 2, int(location[1]) - 1:int(location[1]) + 2] = 0
    time2 = time.time()  # 程序计时结束

    print("hough直线检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv函数检测直线
    edges = cv.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)  # 这里对最后一个参数使用了经验型的值
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    time3 = time.time()  # 程序计时结束
    print("hough直线检测opencv程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, img


def circle_detect(img):
    """*功能 : 直接利用opencv中的hough圆检测，检测出图像中的圆"""
    time1 = time.time()  # 程序计时开始
    # 霍夫变换圆检测
    cv_img = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=5, maxRadius=300)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 绘制外圆
        cv.circle(cv_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 绘制圆心
        cv.circle(cv_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    time2 = time.time()  # 程序计时开始
    print("hough圆检测opencv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return img, cv_img
