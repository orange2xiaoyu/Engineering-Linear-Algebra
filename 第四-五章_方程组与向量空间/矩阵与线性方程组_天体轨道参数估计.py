# -*- coding:utf8 -*-

import numpy as np 
import matplotlib.pyplot as plt 


data = np.array([[0.88, 3.0],
				[1.10, 2.3],
				[1.42, 1.65],
				[1.25, 1.77],
				[1.01, 2.14]])

# 构造系数矩阵
A = np.ones((data.shape[0],data.shape[1]))    #(5*2)
for i in range(0,data.shape[0]):
	theta, r = data[i][0],data[i][1]
	A[i][1] = r * np.cos(theta)

print('A = ', A.shape[0])

# 构造常数项矩阵
# b = data[:,1]
b = np.array([3.00, 2.30, 1.65, 1.25, 1.01])
print('b = ',b)

# 求解估计参数
x = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
print(x)

theta_ = np.linspace(0,2*np.pi, 100)
r_ = x[0]/(1-x[1]*np.cos(theta_))
graph = plt.subplot(111,polar = True)      # 1行1列中的第一个
data_ = data.T
graph.plot(data_[0,:],data_[1,:],'go')
graph.plot(theta_,r_,'r',linewidth = 1)
plt.show()