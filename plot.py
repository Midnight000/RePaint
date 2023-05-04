import numpy as np
import matplotlib.pyplot as plt

# 定义高斯分布的参数
mu = 0  # 均值
sigma = 1  # 标准差

# 生成在一条直线上的随机数
x1 = np.random.normal(mu, sigma, 1000)
x2 = np.random.normal(mu, sigma, 1000)
x3 = np.random.normal(mu, sigma, 1000)
# 生成对应的高斯分布函数值
y1 = 2 * x1
y2 = 1 * x2
y3 = np.random.normal(mu, sigma, 1000)
# 画出散点图
sizes = 10
alphas = 0.75
plt.scatter(x1, y1, s=sizes, alpha=alphas)
plt.scatter(x2, y2, s=sizes, alpha=alphas)
plt.scatter(x3, y3, s=sizes, alpha=alphas)
# plt.title('Gaussian Distribution on a Straight Line')
plt.xlabel('x')
plt.ylabel('y')
plt.show()