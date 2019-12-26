# -*- coding:utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2

#定义变换矩阵
A1 = np.array([[-1,0],
            [0,1]])               # 绕y轴旋转，x值变负

A2 = np.array([[1,0],
            [0,-1]])              # 绕y轴旋转，x值变负

k = 0.5
A3 = np.array([[1,k],
            [0,1]])               # 水平剪切，y值不变，x值变化

A4 = np.array([[1,0],
            [k,1]])               # 竖直剪切，x值不变，y值变化

theta = -(3.14/4)                # 当theta>0时，顺时针旋转；当theta<0时，逆时针旋转
A5 = np.array([[np.cos(theta), np.sin(theta)],
            [-np.sin(theta),np.cos(theta)]])    # 旋转变换


#定义输入矩阵（即输入图形）
B = np.array([[0,1,1,0,0],
            [1,1,0,0,1]])

# 计算输出矩阵（矩阵乘法）
Y = np.dot(A5,B)

# 绘制图形
plt.axis([-3,3,-3,3])                #设置坐标轴的范围
plt.axvline(x = 0, color = 'gray')    # x=0的中心线
plt.axhline(y = 0, color = 'gray')    # y=0的中心线
plt.grid(True)                       # 绘制网格
plt.plot(B[0],B[1],'-yo',lw = 3)     #绘制输入图形，实线黄色圆圈，线宽3cm
plt.plot(Y[0],Y[1],'-go',lw = 3) 
plt.show()

