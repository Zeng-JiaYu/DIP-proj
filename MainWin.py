# Created by: ww 2020/8/11
# 界面与逻辑分离，主窗口逻辑代码

import os
import cv2 as cv
import numpy as np
import sys
from MainDetect_UI import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from Experiment_1 import *
from Experiment_2 import *
from Experiment_3.Experiment_3 import *
from Experiment_4 import *
from Experiment_5 import *


class Main(QMainWindow, Ui_MainWindow):
    """重写主窗体类"""

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self)  # 初始化窗体显示
        self.timer = QTimer(self)  # 初始化定时器
        # 设置在label中自适应显示图片
        self.label_PrePicShow.setScaledContents(True)
        self.label_PrePicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # 初始黑化图像显示区域
        self.label_AftPicShow.setScaledContents(True)
        self.label_AftPicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")
        self.label_AftPicShow_2.setScaledContents(True)
        self.label_AftPicShow_2.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")
        self.img = None

    def onbuttonclick_selectDataType(self, index):
        """选择输入数据类型(图像,视频)"""
        if index == 1:
            filename, _ = QFileDialog.getOpenFileName(self, "选择图像", os.getcwd(), "Images (*.bmp *.jpg *.png);;All (*)")
            self.img = cv.imread(filename, -1)  # 参数-1，原通道，1，强制三通道(彩图)，0强制一通道(灰度图)
            self.img_show(self.label_PrePicShow, self.img)
        elif index == 2:
            filename, _ = QFileDialog.getOpenFileName(self, "选择视频", os.getcwd(), "Videos (*.avi *.mp4);;All (*)")
            self.capture = cv.VideoCapture(filename)
            self.fps = self.capture.get(cv.CAP_PROP_FPS)  # 获得视频帧率
            self.timer.timeout.connect(self.slot_video_display)
            flag, self.img = self.capture.read()  # 显示视频第一帧
            if flag:
                self.img_show(self.label_PrePicShow, self.img)

    def onbuttonclick_videodisplay(self):
        """显示视频控制函数, 用于连接定时器超时触发槽函数"""
        if self.pushButton_VideoDisplay.text() == "检测":
            self.timer.start(1000 / self.fps)
            self.pushButton_VideoDisplay.setText("暂停")
        else:
            self.timer.stop()
            self.pushButton_VideoDisplay.setText("检测")

    def slot_video_display(self):
        """定时器超时触发槽函数, 在label上显示每帧视频, 防止卡顿"""
        flag, self.img = self.capture.read()
        if flag:
            self.img_show(self.label_PrePicShow, self.img)
        else:
            self.capture.release()
            self.timer.stop()

    def oncombox_selectColorType(self, index):
        """选择图像色彩处理方式"""
        imgDeal, cv_img = color_deal(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)
        self.img_show(self.label_AftPicShow_2, cv_img)

    # 第一章相关控件
    def onslide_imgSample(self):
        """滚动条选择采样间隔"""
        iv = self.slider_ImgSample.value()
        self.label_sample.setText(str(iv))
        result = img_sample(self.img, iv)
        self.img_show(self.label_AftPicShow, result)

    def onslide_imgQuanty(self):
        """滚动条选择量化范围"""
        if self.img.shape[-1] == 3:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请选择灰度图像！')
            msg_box.exec_()
            return
        q_Size = self.slider_ImgQuanty.value()
        self.label_quanty.setText('0-' + str(q_Size))
        result = img_quanty(self.img, q_Size)
        self.img_show(self.label_AftPicShow, result)

    # 第二章相关控件
    def onslide_imgZoom(self):
        """控制滚动条实现图像缩放"""
        zm = self.slider_ImgZoom.value()
        self.label_zoom.setText(str(zm + 1) + 'tms') if zm > -1 else self.label_zoom.setText(
            '1/' + str(-zm + 1) + 'tms')
        zm = zm + 1 if zm > -1 else 1.0 / (-zm + 1)
        result, cvResult = img_zoom(self.img, zm)
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def onslide_imgTranslation(self):
        """控制滚动条实现图像左右平移"""
        trans = self.slider_ImgTranslation.value()
        self.label_tanslation.setText(str(trans) + 'pix')
        result, cvResult = img_translation(self.img, trans)
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def onslide_imgRotation(self):
        """控制滚动条进行图像旋转"""
        rot = self.slider_ImgRotate.value()
        self.label_rotate.setText(str(rot) + '°')
        result, cvResult = img_rotation(self.img, rot)
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def onbuttonclick_imgMirror(self):
        """点击按钮做图像镜面"""
        result, cvResult = img_imgMirror(self.img)
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def onbuttonclick_cardCrrection(self):
        """点击按钮做名片矫正"""
        result = card_correction(self.img)
        self.img_show(self.label_AftPicShow, result)

    # 第三章相关控件
    def oncombox_selectGrayDeal(self, index):
        """选择灰度增强处理方式"""
        result, cv_result = gray_deal(self.img, index)
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cv_result)

    def onbutttonclick_histEqual(self):
        """点击做图像增强"""
        result, cvResult = hist_equalization(self.img, self.checkBox_hist.isChecked())
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def oncombox_selectConvType(self, index):
        """选择图像平滑方式"""
        result, cvResult = gray_smooth(self.img, index)
        self.img_show(self.label_AftPicShow, result)
        self.img_show(self.label_AftPicShow_2, cvResult)

    # 第四章相关控件
    def oncombox_selectEdgeDetecType(self, index):
        """选择图像边缘检测方式"""
        imgDeal, cvResult = edge_detect(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def oncombox_selectHoughDetecType(self, index):
        """选择直线/圆检测方式"""
        img, hough_img = hough_detect(self.img, index)
        self.img_show(self.label_AftPicShow, img)
        self.img_show(self.label_AftPicShow_2, hough_img)

    def onbuttonclick_Otsu(self):
        """点击做大津阈值"""
        imgOtsu, cvResult = otsu(self.img, self.checkBox_hist_otsu.isChecked())
        self.img_show(self.label_AftPicShow, imgOtsu)
        self.img_show(self.label_AftPicShow_2, cvResult)

    # 第五章相关控件
    def oncombox_selectMorphyBinary(self, index):
        """选择图像二值形态学运算方式"""
        imgDeal, cvResult = morphy_binary(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def oncombox_selectMorphyGray(self, index):
        """选择图像灰值形态学运算方式"""
        imgDeal, cvResult = morphy_gray(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)
        self.img_show(self.label_AftPicShow_2, cvResult)

    def onbuttonclick_MorphyThin(self):
        """点击做图像快速形态学细化算法(骨架提取)"""
        deal_type = 0 if self.radioButton_Skeleton.isChecked() else 1
        imgThin = fast_thin(self.img, deal_type)
        self.img_show(self.label_AftPicShow_2, imgThin)

    def img_show(self, label, img):
        """图片在对应label中显示"""
        if img.shape[-1] == 3:
            qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1] * 3,
                            QImage.Format_RGB888).rgbSwapped()
        else:
            qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1], QImage.Format_Indexed8)
        label.setPixmap(QPixmap.fromImage(qimage))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    print(app.exec_())
    sys.exit(app.exec_())
