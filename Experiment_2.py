import cv2 as cv
import numpy as np
import time
import math as m
from numba import jit


def interpolationDblLinear(img, zm):
    """双线性插值"""
    result_img = np.zeros((zm + 1, zm + 1, 3), dtype=np.uint8)
    channel = 3
    result_img[0, 0], result_img[0, -1], result_img[-1, 0], result_img[-1, -1] = img[0], img[1], img[2], img[3]
    xp = [0, zm]
    x = range(1, zm)
    for k in range(channel):
        fp1 = [img[0][k], img[1][k]]
        fp2 = [img[2][k], img[3][k]]
        result_img[0, 1:zm, k] = np.interp(x, xp, fp1)
        result_img[-1, 1:zm, k] = np.interp(x, xp, fp2)
        for i in range(zm + 1):
            fp = [result_img[0, i, k], result_img[-1, i, k]]
            result_img[1:zm, i, k] = np.interp(x, xp, fp)
    return result_img


# 缩放操作
def img_zoom(img, zm):
    """对于传入的图像进行缩放操作, *zm:缩放因子"""
    # 手写图像缩放代码
    time1 = time.time()
    rows, cols = img.shape[:2]  # 获取宽和高
    new_rows, new_cols = int(rows * zm), int(cols * zm)
    if img.shape[-1] == 3:
        channel = 3
    else:
        channel = 1
    new_img = np.zeros((new_rows, new_cols, channel), dtype=np.uint8)
    result_img = np.zeros(img.shape, dtype=np.uint8)

    if zm == 1:
        result_img = img
    elif zm < 1:
        new_img = img[0:img.shape[0]:int(1/zm), 0:img.shape[1]:int(1/zm)]
        result_img[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
    elif zm > 1:
        for y in range(new_rows):
            for x in range(new_cols):
                img_x, img_y = x / zm, y / zm
                if img_x.is_integer() and img_y.is_integer():
                    new_img[y, x] = img[int(img_y), int(img_x)]
                    continue
                if m.ceil(img_x) == cols or m.ceil(img_y) == rows:
                    new_img[y, x] = img[m.floor(img_y), m.floor(img_x)]
                    continue
                p11 = img[m.floor(img_y), m.floor(img_x)]
                p12 = img[m.floor(img_y), m.ceil(img_x)]
                p21 = img[m.ceil(img_y), m.floor(img_x)]
                p22 = img[m.ceil(img_y), m.ceil(img_x)]
                dx, dy = img_x-m.floor(img_x), img_y - m.floor(img_y)
                temp1 = dx * p11 + (1 - dx) * p12
                temp2 = dx * p21 + (1 - dx) * p22
                new_img[y, x] = dy * temp1 + (1 - dy) * temp2
                result_img = new_img[0:rows, 0:cols]
    time2 = time.time()
    # opencv图像缩放
    cv_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    img = cv.resize(img, (int(cols * zm), int(rows * zm)))
    if zm > 1:  # 原图像大小显示
        cv_img = img[0:rows, 0:cols]
    else:
        cv_img[0:img.shape[0], 0:img.shape[1]] = img
    time3 = time.time()
    print("手写缩放程序处理时间： %.3f毫秒" % ((time2 - time1) * 1000))
    print("opencv缩放程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return result_img, cv_img


# 平移操作
def img_translation(img, trans):
    """对于传入的图像进行左右, *trans:平移参数"""
    time1 = time.time()
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    if trans < 0:
        new_img[:, 0:cols + trans] = img[:, abs(trans):]
    elif trans > 0:
        new_img[:, trans:] = img[:, 0:cols - trans]
    else:
        new_img = img
    time2 = time.time()
    # opencv实现平移操作
    M = np.float32([[1, 0, trans], [0, 1, 0]])
    cv_img = cv.warpAffine(img, M, (cols, rows))
    time3 = time.time()
    print("手写图像平移程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    print("opencv缩放程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


# 镜面变换
def img_imgMirror(img):
    """对于传入的图像进行镜面变换"""
    time1 = time.time()
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    for col in range(cols):  # 遍历每一列
        new_img[:, col] = img[:, cols - 1 - col]
    time2 = time.time()
    cv_img = cv.flip(img, 1)
    time3 = time.time()
    print("手写镜面变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    print("opencv缩放程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


# 旋转操作
def img_rotation(img, rot):
    """对于传入的图像进行旋转，可以绕任一点旋转, *rot:旋转角度"""
    rot = m.radians(rot)    # 将旋转角度从度转到弧度
    rows, cols = img.shape[:2]  # 获取宽和高
    channel = img.shape[-1]
    time1 = time.time()
    fSrcX, fSrcY, fDstX, fDstY = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(5)
    cos, sin = m.cos(rot), m.sin(rot)
    fSrcX[0], fSrcY[0] = -(cols - 1) / 2, (rows - 1) / 2
    fSrcX[1], fSrcY[1] = (cols - 1) / 2, (rows - 1) / 2
    fSrcX[2], fSrcY[2] = -(cols - 1) / 2, -(rows - 1) / 2
    fSrcX[3], fSrcY[3] = (cols - 1) / 2, -(rows - 1) / 2
    for i in range(4):
        fDstX[i], fDstY[i] = cos * fSrcX[i] + sin * fSrcY[i], -sin * fSrcX[i] + cos * fSrcY[i]
    new_cols = m.ceil(max(abs(fDstX[3] - fDstX[0]), abs(fDstX[2] - fDstX[1])) + 0.5)
    new_rows = m.ceil(max(abs(fDstY[3] - fDstY[0]), abs(fDstY[2] - fDstY[1])) + 0.5)
    new_img = np.zeros((new_rows + 3, new_cols + 3, channel), dtype=np.uint8)
    f1 = -0.5 * (new_cols - 1) * cos + 0.5 * (new_rows - 1) * sin + 0.5 * (cols - 1)
    f2 = -0.5 * (new_cols - 1) * sin - 0.5 * (new_rows - 1) * cos + 0.5 * (rows - 1)
    for i in range(new_rows-1):
        for j in range(new_cols-1):
            coordinateX = j * cos - i * sin + f1
            coordinateY = j * sin + i * cos + f2
            iu = int(coordinateX)
            iv = int(coordinateY)
            array = []
            if 0 <= coordinateX < cols - 1 and 0 <= coordinateY < rows - 1:
                array.append(img[iv, iu])
                array.append(img[iv, iu + 1])
                array.append(img[iv + 1, iu])
                array.append(img[iv + 1, iu + 1])
                new_img[i:i + 3, j:j + 3] = interpolationDblLinear(array, 2)
    time2 = time.time()
    # opencv绕任一点旋转代码
    # 第一个参数是旋转中心，第二个参数是旋转角度，第三个因子是旋转后的缩放因子
    rot = m.degrees(rot)
    b, a = rows / 2, cols / 2  # 设置旋转点位置
    h, w = rows / 2, cols / 2  # 图像高宽的一半
    M = cv.getRotationMatrix2D((a, b), rot, 1)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_cols = int((rows * sin) + (cols * cos))
    new_rows = int((rows * cos) + (cols * sin))
    M[0, 2] += (new_cols / 2) - w
    M[1, 2] += (new_rows / 2) - h
    cv_img = cv.warpAffine(img, M, (new_cols, new_rows))  # 第三个参数是输出图像的尺寸中心，图像的宽和高
    time3 = time.time()
    print("手写旋转程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    print("opencv旋转程序处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img


def onmouse_pick_points(event, x, y, flags, l_ImgPot):
    """card_correction的鼠标回调函数, """
    if event == cv.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d' % (x, y))
        l_ImgPot[2].append((x, y))
        cv.drawMarker(l_ImgPot[1], (x, y), (0, 0, 255))
    if event == cv.EVENT_RBUTTONDOWN:
        l_ImgPot[1] = l_ImgPot[0].copy()  # 将没有画十字的原图重新赋值给显示图像
        if len(l_ImgPot[2]) != 0:
            l_ImgPot[2].pop()  # 将最后一次绘制的标记清除
            for i in range(len(l_ImgPot[2])):  # 重新绘制全部标记
                cv.drawMarker(l_ImgPot[1], l_ImgPot[2][i], (0, 0, 255))


def card_correction(img):
    """对于传入的图像进行鼠标交互，选择四个顶点进行名片矫正"""
    l_ImgPot = [img, img.copy(), []]  # 记录画标识的图像和标识点  [0]原图 [1]处理后图
    cv.namedWindow('card', cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback('card', onmouse_pick_points, l_ImgPot)
    while True:
        cv.imshow('card', l_ImgPot[1])
        key = cv.waitKey(30)
        if key == 27:  # ESC
            break
    cv.destroyAllWindows()
    # 手写算法实现透视变换
    time1 = time.time()
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    time2 = time.time()
    # opencv实现透视变换
    pts1 = np.float32(l_ImgPot[2])
    pts2 = np.float32([[0, 0], [540, 0], [0, 900], [540, 900]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    cv_img = cv.warpPerspective(img, M, (540, 900))
    time3 = time.time()
    print("opencv名片矫正处理时间：%.3f毫秒" % ((time3 - time2) * 1000))
    return new_img, cv_img
