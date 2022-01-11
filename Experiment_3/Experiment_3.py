import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment_3.Linear_UI import Ui_Dialog as UI_Linear
from Experiment_3.Log_UI import Ui_Dialog as UI_Log
from Experiment_3.Exp_UI import Ui_Dialog as UI_Exp
from Experiment_3.Pow_UI import Ui_Dialog as UI_Pow
from Experiment_3.Conv_UI import Ui_Dialog as UI_Conv
from Experiment_3.Piecewise_UI import Ui_Dialog as UI_Piecewise
import time


# 灰度变化
def gray_deal(img, deal_Type):
    """根据用户的选择，对于图像做相应的灰度增强处理"""
    new_img = np.zeros(img.shape, dtype=np.uint8)
    cv_img = np.zeros(img.shape, dtype=np.uint8)
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        new_img = linear_strench(img)
    elif deal_Type == 2:
        new_img = log_strench(img)
    elif deal_Type == 3:
        new_img = exp_strench(img)
    elif deal_Type == 4:
        new_img = pow_strench(img)
    elif deal_Type == 5:
        new_img = piecewise_strench(img)
    return new_img, cv_img


# 分段线性变化
def piecewise_strench(img):
    a, b = np.min(img), np.max(img)
    q_dialog = QDialog()
    dialog = UI_Piecewise()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(a))  # 显示原图灰度范围
    dialog.lineEdit_b.setText(str(b))
    dialog.lineEdit_c.setText(str(a))  # 初始化变换后灰度范围
    dialog.lineEdit_d.setText(str(b))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互变换后灰度范围
        a = int(dialog.lineEdit_a.text())
        b = int(dialog.lineEdit_b.text())
        c = int(dialog.lineEdit_c.text())
        d = int(dialog.lineEdit_d.text())
        time1 = time.time()  # 程序计时开始
        L = 256
        new_img = np.select([img < a, img > b, True],
                            [c * img / a, (L - 1 - d) / (L - 1 - b) * (img - b) + d, (d - c) / (b - a) * (img - a) + c])
        new_img = new_img.astype(np.uint8)
        new_img = np.clip(new_img, 0, 255)
        time2 = time.time()  # 程序计时结束
        print("灰度增强程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
        return new_img


# 线性变化 c:0 d:255
def linear_strench(img):
    """*功能 : 根据传入的图像及给定的c,d两个灰值区间参数值，进行线性拉伸
    *注意，只对灰度图像拉伸，函数：g(x,y)=(d-c)/(b-a)*[f(x,y)-a]+c=k*[f(x,y)-a]+c"""
    a, b = np.min(img), np.max(img)
    q_dialog = QDialog()
    dialog = UI_Linear()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(a))  # 显示原图灰度范围
    dialog.lineEdit_b.setText(str(b))
    dialog.lineEdit_c.setText(str(a))  # 初始化变换后灰度范围
    dialog.lineEdit_d.setText(str(b))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互变换后灰度范围
        c = int(dialog.lineEdit_c.text())
        d = int(dialog.lineEdit_d.text())
        time1 = time.time()  # 程序计时开始
        k = (d - c) / (b - a)
        new_img = k * (img - a) + c
        new_img = new_img.astype(np.uint8)
        new_img = np.clip(new_img, 0, 255)
        time2 = time.time()  # 程序计时结束
        print("灰度增强程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
        return new_img


# 对数变化 a=-425 b=2 c=0.012
def log_strench(img):
    """*功能 : 根根据传入的图像及给定的a,b,c三个参数值，进行对数非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=a+lg[f(x,y)+1]/(c*lgb)"""
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Log()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(0.0))  # 初始化对数变换参数
    dialog.lineEdit_b.setText(str(2.0))
    dialog.lineEdit_c.setText(str(0.03))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        a, b, c = float(dialog.lineEdit_a.text()), float(dialog.lineEdit_b.text()), float(dialog.lineEdit_c.text())
        if c == 0 or b <= 0 or b == 1:  # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img
        time1 = time.time()  # 程序计时开始
        new_img = a + np.log10(img + 1) / (c * np.log10(b))
        new_img = np.clip(new_img, 0, 255)
        new_img = new_img.astype(np.uint8)
        time2 = time.time()  # 程序计时结束
        print("对数变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


# 指数变化 a=0 b=1.05 c=0.55
def exp_strench(img):
    """*功能 : 根根据传入的图像及给定的a,b,c三个参数值，进行指数非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=b^c[f(x,y)-a]-1"""
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Exp()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(150))  # 初始化对数变换参数
    dialog.lineEdit_b.setText(str(1.5))
    dialog.lineEdit_c.setText(str(0.6))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        a, b, c = float(dialog.lineEdit_a.text()), float(dialog.lineEdit_b.text()), float(dialog.lineEdit_c.text())
        if b <= 0 or b == 1:  # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img
        time1 = time.time()  # 程序计时开始
        new_img = b ** (c * (img - a)) - 1
        new_img = np.clip(new_img, 0, 255)
        new_img = new_img.astype(np.uint8)
        time2 = time.time()  # 程序计时结束
        print("指数变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


# 幂律变化 c=0.1 r=1.5
def pow_strench(img):
    """*功能 : 根根据传入的图像及给定的c,r两个参数值，进行幂律非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=c[f(x,y)]^r"""
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Pow()
    dialog.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_c.setText(str(1))  # 初始化对数变换参数
    dialog.lineEdit_r.setText(str(1.5))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        c, r = float(dialog.lineEdit_c.text()), float(dialog.lineEdit_r.text())
        if r <= 0 or c <= 0:  # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img
        time1 = time.time()  # 程序计时开始
        new_img = c * (img ** r)
        new_img = np.clip(new_img, 0, 255)
        new_img = new_img.astype(np.uint8)
        time2 = time.time()  # 程序计时结束
        print("幂律变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img


# 直方图均衡化
def hist_equalization(img, jug):
    """*功能 : 直方图均衡化算法, jug判断返回是图像/直方图"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    time1 = time.time()  # 程序计时开始
    hist = creat_histogram(img)
    hist_rate = np.array(hist) / (rows * cols)
    hist_rate = np.cumsum(hist_rate)
    result = np.round(np.array(hist_rate) * (len(hist_rate) - 1))
    for i in range(rows):
        for j in range(cols):
            new_img[i, j] = result[img[i, j]]
    time2 = time.time()  # 程序计时结束
    equ = cv.equalizeHist(img)
    time3 = time.time()  # 程序计时结束
    print("图像均衡算法程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    print("图像均衡算法程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    if jug:
        imgs = [img, new_img]
        colors = ("b", "r")
        texts = ("original histogram", "histogram after equalization")
        for i in range(2):
            hist = cv.calcHist([imgs[i]], [0], None, [256], [0, 255])
            plt.plot(hist, color=colors[i], label=texts[i])
        plt.xlim([0, 256])
        plt.legend()
        plt.show()
    return new_img, equ


# 创建直方图
def creat_histogram(img):
    """*功能 : 计算传入图像直方图，若是彩色图像，计算各颜色分量直方图并返回"""
    rows, cols = img.shape[:2]  # 获取宽和高
    hist = []
    if img.ndim == 2:  # 灰度图像统计直方图
        hist = [0] * 256  # 建立灰度图像直方图
        # 图像遍历
        for row in range(rows):
            for col in range(cols):
                hist[img[row][col]] += 1
    elif img.ndim == 3:  # 彩色图像统计直方图
        hist = [[0] * 256, [0] * 256, [0] * 256]  # 建立彩色图像直方图
        # 图像遍历
        for row in range(rows):
            for col in range(cols):
                hist[0][img[row][col][0]] += 1
                hist[1][img[row][col][1]] += 1
                hist[2][img[row][col][2]] += 1
    return hist


def gray_smooth(img, deal_Type):
    """根据用户的选择，对于图像做相应的图像平滑处理"""
    new_img = np.zeros(img.shape, dtype=np.uint8)
    cv_img = np.zeros(img.shape, dtype=np.uint8)
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        new_img, cv_img = neighbor_average(img)
    elif deal_Type == 2:
        new_img, cv_img = median_filter(img)
    return new_img, cv_img


# 均值滤波
def neighbor_average(img):
    """*功能 : 用户交互卷积模板，获取卷积系数进行邻域平滑，只对灰度图像处理"""
    rows, cols = img.shape[:2]  # 获取宽和高
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
        np_kernel = np_kernel / np_kernel.sum()  # 正则化
        time1 = time.time()  # 程序计时开始
        # 手写算法实现
        img = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        for i in range(rows):
            for j in range(cols):
                new_img[i, j] = np.multiply(img[i: i + 3, j: j + 3], np_kernel).sum()  # 点对点相乘后进行累加
        time2 = time.time()  # 程序计时结束
        # opencv实现
        cv_img = cv.GaussianBlur(img, (3, 3), 0)
        time3 = time.time()  # 程序计时结束
        print("手写平均平滑程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
        print("opencv邻域平均平滑程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


# 中值滤波
def median_filter(img):
    """*功能 : 中值滤波"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    cv_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像

    time1 = time.time()  # 程序计时开始
    # 手写算法实现
    len = 3  # 定义中值滤波模板3×3
    new_img_2 = np.zeros((rows - 2, cols - 2, 9))
    for i in range(len):
        for j in range(len):
            new_img_2[:, :, i * len + j] = img[i:rows + i - len + 1, j:cols + j - len + 1]
    new_img[1:rows - 1, 1:cols - 1] = np.median(new_img_2, axis=2)
    time2 = time.time()

    # opencv实现
    cv_img = cv.medianBlur(img, 3)
    time3 = time.time()  # 程序计时结束
    print("中值滤波程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    print("opencv滤波程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img
