# -*- coding:utf-8 -*-
# @author: jim-bin

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import skimage

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes=load_dataset()
print(train_set_x_orig.shape, train_set_y.shape, test_set_x_orig.shape, test_set_y.shape, classes)
'''
train_set_x_orig.shape = 209, 64, 64, 3   表示训练集共有209个样本，，每个像素64像素高，64像素宽，3层，每个图片的像素数是64×64×3

[ [ [ [ x, x, x ]...64 ],...64 ],...209 ]
x,x,x表示红绿蓝的像素值

# train_set_y = 1,209 1行，209列；对应209个标签，每个的值为1或者0
# print(train_set_y)

[[0 0 1 0 0 0 0 1 0 0 0 1 0 1 1 0 0...]]
'''

# 图像实例
# index = 2
# example = train_set_x_orig[index] # 第四个样本
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

'''
取出每个图片对应的标签，转为字符串类型，numpy可以直接操作train_set_y[:, index]
纯python代码则不行，必须train_set_y[i][j]
'''
print(classes[0])
print(classes[1])
'''
b'non-cat'
b'cat'

a = e.reshape(1,1,10)
a
array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])

np.squeeze(a)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

'''

m_train = train_set_x_orig.shape[0]
m_test  = test_set_x_orig.shape[0]
num_px  = train_set_x_orig.shape[2]

print ("Number of training examples: m_train = " + str(m_train)) # 训练样本集个数
print ("Number of testing examples: m_test = " + str(m_test)) # 测试样本集个数
print ("Height/Width of each image: num_px = "  + str(num_px)) # 每个图片的高度或者宽度
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3")

print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: "  + str(test_set_x_orig.shape))
print ("test_set_y shape: "  + str(test_set_y.shape))

# example, reshape the matrix
z = np.array([[[1, 2, 3, 4, 1],
          [5, 6, 7, 8, 1],
          [9, 10, 11, 12, 1],
          [13, 14, 15, 16, 1]]])
print(z.shape)
print(z)
zz = z.reshape(1,-1)
print(zz.shape)
print(zz)

a = np.array([[1,2,3]])
b = np.array([1,2,3])
print(b.shape)
print(a.shape)
print(a)
print(np.squeeze(a))

# reshape the training and testing exampes
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
'''
[[x,x], [x,x],...] 64*64*3, 209
'''
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0])) # 第一列的前五个数，组成数组

# 均一化数据集，每个像素值除以255
train_set_x = train_set_x_flatten/ 255.
test_set_x = test_set_x_flatten / 255.
print (train_set_x.shape)
print (test_set_x.shape)
print (len(train_set_x))
print (len(test_set_x))

# sigmoid function
def sigmoid(z):
    s = 1. / (1 + np.exp(-z))
    return s

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

# 初始化参数以0
def initialize_with_zeros(dim):
    # dim 是向量w的大小，此处是w的个数，w为dim行，1列矩阵，或者dim维列向量
    # 返回全零的w，shape是(dim,1)
    # 返回标量b，偏置
    w = np.zeros(shape=(dim,1), dtype=np.float32)
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))


# 前馈传播
def propagate(w, b, X, Y):
    '''

    :param w: 权重，大小为(num_px * num_px * 3, 1)的numpy数组
    :param b: 偏置，标量，单个数字
    :param X: 大小为(num_px * num_px * 3, number of example)的numpy数组
    :param Y: 大小为(1, number of examples)的标签向量，一行多列
    :return: cost是逻辑回归的负对数似然成本，dw是w的梯度，db是b的梯度
    '''
    m = X.shape[1]  # 样本集个数

    # 前馈传播，从X到cost
    A = sigmoid(np.dot(w.T, X) + b) # 大小为（1， number of examples）
    cost = (-1./m) * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)), axis=1) # 按列相加，把每列的元素加在一起

    # 反向传播
    dw = (1./m) * np.dot(X, (A-Y).T)
    db = (1./m) * np.sum(A - Y, axis=1)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    print (cost)
    cost = np.squeeze(cost)
    print (cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    return grads, cost

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
# w 2*1, X 2*2, Y 1*2
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

# 更新参数
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''
    :param w: 权重，大小为(num_px * num_px * 3, 1)的numpy数组
    :param b: 偏置，标量，单个数字
    :param X: 大小为(num_px * num_px * 3, number of example)的numpy数组
    :param Y: 大小为(1, number of examples)的标签向量，一行多列
    :param num_iterations: 迭代次数
    :param learning_rate: 学习速率
    :param print_cost: 每100步打印一次cost
    :return: 权重w和偏置b的字典，dw，db的字典， costs，所有代价函数值的列表
    '''

    costs = []
    for i in range(num_iterations):
        # 前馈传播，计算梯度和代价函数
        grads, cost = propagate(w=w, b=b, X=X, Y=Y)
        # 检索梯度值
        db = grads["db"]
        dw = grads["dw"]

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每迭代100步，打印一次代价函数
        if i % 100 == 0:
            costs.append(cost)

        # 每训练100个样本，打印一次代价函数
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {
        "w" : w,
        "b" : b
    }

    grads = {
        "dw" : dw,
        "db" : db
    }

    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

def predict(w, b, X):
    m = X.shape[1] # 样本个数
    Y_prediction = np.zeros((1, m)) # 初始化为1行m列的向量
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    [print(x) for x in A]
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert (Y_prediction.shape == (1, m))
    return Y_prediction
print ("predictions = " + str(predict(w, b, X)))


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# index = 1
# plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
# print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")


# 画出学习曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('costs')
plt.xlabel('iterations (per 100)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()


learning_rates = [0.01, 0.001, 0.0001]
models = {

}

for i in learning_rates:
    print("learning_rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)
    print("\n" + "----------------------------------------------------------------------" + "\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel("costs")
plt.xlabel("iterarions")

legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()


# 测试自己的图片
my_image = "cat2.jpg"
fnanme = "images/" + my_image
image = np.array(plt.imread(fnanme))
my_image = skimage.transform.resize(image, output_shape=(num_px, num_px,)).reshape((1, num_px*num_px*3)).T
my_predictied_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predictied_image)) + ", your alorithm predicts a \"" + classes[int(np.squeeze(my_predictied_image))].decode("utf-8") + "\" picture.")
plt.show()