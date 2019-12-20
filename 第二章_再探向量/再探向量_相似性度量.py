import numpy as np 
import matplotlib.pyplot as plt

def calEuclideanDistance(vec1,vec2):
	return np.sqrt(np.sum(np.square(vec1-vec2)))

def calCosim(vec1,vec2):
	return np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))   # np.dot表示内积，np.linalg.norm表示向量的模长

print("------------------第一个例子-----------------")
a_1 = np.array([2,10])
b_1 = np.array([20,100])
c_1 = np.array([3,15])

print("dist(a_1,b_1)",calEuclideanDistance(a_1,b_1))
print("dist(a_1,c_1)",calEuclideanDistance(a_1,c_1))
print("dist(b_1,c_1)",calEuclideanDistance(b_1,c_1))

print("sim(a_1,b_1)",calCosim(a_1,b_1))
print("sim(a_1,c_1)",calCosim(a_1,c_1))
print("sim(b_1,c_1)",calCosim(b_1,c_1))

plt.xlim(0,30)    #绘制x坐标
plt.ylim(0,110)
plt.grid()       #添加网格
plt.plot(a_1[0],a_1[1],'ro')
plt.plot(b_1[0],b_1[1],'bo')
plt.plot(c_1[0],c_1[1],'go')
plt.show()

print("------------------第二个例子-----------------")
a_2 = np.array([1,1])
b_2 = np.array([5,5])
c_2 = np.array([5,0])

print("dist(a_2,b_2)",calEuclideanDistance(a_2,b_2))
print("dist(a_2,c_2)",calEuclideanDistance(a_2,c_2))
print("dist(b_2,c_2)",calEuclideanDistance(b_2,c_2))

print("sim(a_2,b_2)",calCosim(a_2,b_2))
print("sim(a_2,c_2)",calCosim(a_2,c_2))
print("sim(b_2,c_2)",calCosim(b_2,c_2))

plt.xlim(0,6)    #绘制x坐标
plt.ylim(0,6)
plt.grid()       #添加网格
plt.plot(a_2[0],a_2[1],'ro')
plt.plot(b_2[0],b_2[1],'bo')
plt.plot(c_2[0],c_2[1],'go')
plt.show()
