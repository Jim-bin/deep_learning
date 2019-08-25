# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets

import operator
from functools import reduce


np.random.seed(1) # set a seed so that the results are consistent

X, Y = load_planar_dataset()
print("the X shape is: ", X.shape)
print("the Y shape is: ", Y.shape)

def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t

def dtanh(x):
    t = tanh(x)
    return 1 - t * t

def sigmoid(x):
    s = 1.0 / (1 + np.exp(-x))
    return s

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

class Network(object):
    def __init__(self, n_x, n_h, n_y):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

    def initializer(self):
        W1 = np.random.random([self.n_x, self.n_h]) * 0.01
        b1 = np.random.random([self.n_h, 1]) * 0.01
        # b1 = np.zeros([self.n_h, 1])
        W2 = np.random.random([self.n_h, self.n_y]) * 0.01
        b2 = np.random.random([self.n_y, 1]) * 0.01
        # b2 = np.zeros([self.n_y, 1])

        parameters = {
            "W1" : W1,
            "b1" : b1,
            "W2" : W2,
            "b2" : b2,
        }

        return parameters

    def propogate(self, X, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1.T, X) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2.T, A1) + b2
        A2 = sigmoid(Z2)

        cache = {
            "Z1" : Z1,
            "A1" : A1,
            "Z2" : Z2,
            "A2" : A2,
        }

        return A2, cache

    def compute_cost(self, A2, Y, parameters):
        m = Y.shape[1]
        cost = -1.0 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2), axis=1)
        cost = np.squeeze(cost)
        return cost

    def backward_propagate(self, X, Y, cache, parameters):
        m = X.shape[1]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        Z2 = cache["Z2"]
        A2 = cache["A2"]

        # print("Z1 shape is: ", Z1.shape)
        # print("A1 shape is: ", A1.shape)
        # print("Z2 shape is: ", Z2.shape)
        # print("A2 shape is: ", A2.shape)

        W2 = parameters["W2"]
        W1 = parameters["W1"]

        dZ2 = A2 - Y
        dW2 = 1.0 / m * np.dot(A1, dZ2.T)
        db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1.0 / m * np.dot(X, dZ1.T)
        db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

        # print("dZ2 shape is: ", dZ2.shape)
        # print("dZ1 shape is: ", dZ1.shape)
        # print("dW1 shape is: ", dW1.shape)
        # print("dW2 shape is: ", dW2.shape)
        # print("db1 shape is: ", db1.shape)
        # print("db2 shape is: ", db2.shape)

        grads = {
            "dW2" : dW2,
            "db2" : db2,
            "dW1" : dW1,
            "db1" : db1,
        }
        return grads

    def update(self, parameters, grads, learning_rate):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        b1 = b1 - learning_rate * db1
        b2 = b2 - learning_rate * db2

        parameters = {
            "W1" : W1,
            "b1" : b1,
            "W2" : W2,
            "b2" : b2,
        }
        return parameters

    def predict(self, parameters, X):
        A2, cache = self.propogate(X, parameters)
        predictions = np.array([1 if x > 0.5 else 0 for x in A2.reshape(-1, 1)]).reshape(A2.shape)
        # predictions = np.array([1 if x > 0.5 else 0 for x in A2.reshape(-1, 1)]).reshape(A2.shape)
        return predictions

    def train(self, X, Y, numbers_train, learning_rate):
        parameters = self.initializer()
        print("W1 shape is: ", parameters["W1"].shape)
        print("b1 shape is: ", parameters["b1"].shape)
        print("W2 shape is: ", parameters["W2"].shape)
        print("b2 shape is: ", parameters["b2"].shape)
        for i in range(numbers_train):
            # 正向传播，计算Z1,A1,Z2,A2,并返回
            A2, cache = self.propogate(X, parameters)
            # 利用A2和Y计算代价函数cost
            cost = self.compute_cost(A2, Y, parameters)
            # 反向传播计算W和b的梯度
            grads = self.backward_propagate(X, Y, cache,  parameters)
            # 更新参数W和b
            parameters = self.update(parameters, grads, learning_rate)

            if i % 1000 == 0:
                print("the cost is:", cost, " when the train nunmber is: ", i)

        # 返回训练结束时候的参数W和b
        return parameters




nn = Network(2,4,1)
last_parameters = nn.train(X, Y, 10000, 1.2)
# print(last_parameters)

# # Build a model with a n_h-dimensional hidden layer
# # parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# # Plot the decision boundary
# plot_decision_boundary(lambda x: nn.predict(last_parameters, x.T), X, reduce(operator.add, Y))
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()

# # Print accuracy
# predictions = nn.predict(last_parameters, X)
# print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# # This may take about 2 minutes to run

# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     nn = Network(2, n_h, 1)
#     parameters = nn.train(X, Y, 10000, 1.2)
#     plot_decision_boundary(lambda x: nn.predict(parameters, x.T), X, reduce(operator.add, Y))
#     predictions = nn.predict(parameters, X)
#     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
# plt.show()

# Plot the decision boundary
# plot_decision_boundary(lambda x: nn.predict(last_parameters, x.T), X, reduce(operator.add, Y))
# plt.title("Decision Boundary for hidden layer size " + str(4))

# print("fished")


# parameters = nn.initializer()
# A2 = nn.propogate(X, parameters)
# cost = nn.compute_cost(A2, Y, parameters)

# print(cost)

# print(parameters)
# print(nn.propogate(X, parameters))
