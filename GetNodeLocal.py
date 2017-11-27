#!/usr/bin/env python
# coding:utf8

import cv2
import numpy as np
from PIL import Image
from PIL import  ImageOps
import sys
import re

import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
try:
    from pyocr import pyocr
    from PIL import Image
except ImportError:
    print('模块导入错误,请使用pip安装,pytesseract依赖以下库：')
    print('http://www.lfd.uci.edu/~gohlke/pythonlibs/#pil')
    print('http://code.google.com/p/tesseract-ocr/')
    raise SystemExit


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 图像二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))


    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 7. 存储中间图片
    # cv2.imwrite("binary.png", binary)
    # cv2.imwrite("dilation.png", dilation)
    # cv2.imwrite("erosion.png", erosion)
    # cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    img,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 1500):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #print(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2 or height <= 15):
            continue


        region.append(box)

    return region


def detect(img,im,path):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(img,cv2.COLOR_BAYER_RG2RGB)
    #gray = cv2.imread(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray.png",gray)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)

    # 4. 用绿线画出这些找到的轮廓
    #a = 0
    # 5.拼接一个json对象
    result= []
    for box in region:
        minleft = min(box[0][0],box[1][0],box[2][0],box[3][0])
        mintop = min(box[0][1],box[1][1],box[2][1],box[3][1])
        maxleft = max(box[0][0],box[1][0],box[2][0],box[3][0])
        maxtop = max(box[0][1],box[1][1],box[2][1],box[3][1])
        box1 = (minleft,mintop,maxleft,maxtop)

        if box[0][0] * box[1][1] * box[2][0] * box[0][1] != 0:
            png = im.crop(box1)
            # a = a + 1
            # png.save('/Users/martin/data/pic/'
            #          +str(a)+'.png')

            res = Imgprint(png)
            if len(res) != 0:
                a={}
                a['name'] = ''.join(res);
                a['top'] = str(box[1][1]);
                a['left'] = str(box[1][0]);
                result.append(a)


        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    print(result)
    #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #cv2.imshow("img", img)


    # 带轮廓的图片
    cv2.imwrite(path.replace("png","jpg",),img)

    #cv2.waitKey(0)
    cv2.destroyAllWindows()


def Imgprint(img):
    tools = pyocr.get_available_tools()[:]
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    #print("Using '%s'" % (tools[0].get_name()))
    #print(tools[0].image_to_string(Image.open('/Users/martin/data/9.png'), lang='chi_sim'))
    res = tools[0].image_to_string(img, lang='chi_sim')
    res = re.findall(r"[\u4e00-\u9fa5]", res, re.S)
    return res
if __name__ == '__main__':

    path = sys.argv[1]
    im = Image.open(path)
    img = cv2.imread(path)

detect(img,im,path)



