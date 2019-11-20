'''
Linear Regression with TensorFlow
'''
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape

# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data] # 在第一列多加一列，值为1


X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")

# reshape（行，列）可以根据指定的数值将数据转换为特定的行数和列数

y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")


# tf.transpose(X):将X进行转置，并且根据perm参数重新排列输出维度。这是对数据的维度的进行操作的形式。
XT = tf.transpose(X)

# tf.matrix_inverse:求矩阵的逆
# tf.matmul：矩阵乘法

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    # tf.Tensor.eval() 当默认的会话被指定之后可以通过其计算一个张量的取值。
    theta_value = theta.eval()
    print(theta_value)

# 非tensorflow实现线性回归
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)