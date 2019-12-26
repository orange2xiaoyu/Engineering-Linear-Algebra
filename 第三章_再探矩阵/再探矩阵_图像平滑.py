# -*- coding:utf8 -*-

import numpy as np
import cv2
from PIL import Image
from scipy import signal

"""
方法一：调用PIL库，实现对图像的操作
"""
# # 读取一张图片（PIL），并将其灰度化
# filename = "./lena.jpeg"
# image_rgb = Image.open(filename)
# image_gray = image_rgb.convert('L')    # 彩色图片转为灰度图

# # 将灰度图转化为像素矩阵（PIL需要将图片转成像素矩阵进行处理，cv2不需要，因为读进来的图像已经是uint8的矩阵了）
# matrix = np.asarray(image_gray)
# print("matrix.shape = ", matrix.shape)    # 输出矩阵的尺寸

# # 定义卷积核(均值滤波器)
# filter_3 = np.array([[1/9, 1/9, 1/9],
#                         [1/9, 1/9, 1/9],
#                         [1/9, 1/9, 1/9]])

# filter_7 = np.ones((7,7))/(7*7)

# gaussian_filter_7 = np.array([  [ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067],                       
#                                   [ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
#                                   [ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],                        
#                                   [ 0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771],
#                                   [ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],                        
#                                   [ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
#                                   [ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067] ])  

# sobel = np.array([[ -1, -2, -1 ],                        
#                   [  0,  0,  0 ],
#                   [  1,  2,  1 ]], dtype=float) 

# # 开始卷积（图像平滑）
# result = signal.convolve2d(matrix, filter_3, mode = 'same')
# result1 = signal.convolve2d(matrix, filter_7, mode = 'same')  # mode参数表示生成图像与原始图像大小不变，即补0操作
# result2 = signal.convolve2d(matrix, gaussian_filter_7, mode = 'same')
# result3 = signal.convolve2d(matrix, sobel, mode = 'same')
# print("result.shape = ", result.shape)
   
# # 将像素矩阵转换成图像（PIL需要将矩阵转为图像，才可以显示）
# image_rlt = Image.fromarray(result)
# image_rlt1 = Image.fromarray(result1)
# image_rlt2 = Image.fromarray(result2)
# image_rlt3 = Image.fromarray(result3)
# image_rlt.show()
# image_rlt1.show()
# image_rlt2.show()
# image_rlt3.show()


"""
方法二：调用cv2库，实现对图像的操作（常用）
"""
# 读取一张图片（cv2），并将其灰度化
img = cv2.imread("./lena.jpeg",0)       # 1表示rgb，0表示灰度
cv2.imshow("img", img)

#定义卷积核
filter_7 = np.ones((7,7))/(7*7)
gaussian_filter_7 = np.array([  [ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067],                       
                                  [ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
                                  [ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],                        
                                  [ 0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771],
                                  [ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],                        
                                  [ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
                                  [ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067] ])  
sobel = np.array([[ -1, -2, -1 ],                        
                  [  0,  0,  0 ],
                  [  1,  2,  1 ]]) 

# 图像平滑
img_result1 = signal.convolve2d(img, filter_7, mode = 'same')
img_result2 = signal.convolve2d(img, gaussian_filter_7, mode = 'same')
img_result3 = signal.convolve2d(img, sobel, mode = 'same')

cv2.imwrite("img_result1.jpg",img_result1)
cv2.imwrite("img_result2.jpg",img_result2)
cv2.imwrite("img_result4.jpg",img_result3)

cv2.imshow("img_result1",img_result1.astype(np.uint8))    # 将卷积后的图像转换为uint8格式，并显示，图片名为img_result1
cv2.imshow("img_result2",img_result2.astype(np.uint8))    # 将卷积后的图像转换为uint8格式，并显示，图片名为img_result1
cv2.imshow("img_result3",img_result3.astype(np.uint8))    # 将卷积后的图像转换为uint8格式，并显示，图片名为img_result1
cv2.waitKey(0)
