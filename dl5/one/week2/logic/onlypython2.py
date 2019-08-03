import numpy as np

# 训练集
X = np.array([
  [1, 1],  # Alice
  [-3, -2],   # Bob
  [3, 4],   # Charlie
  [-5, -6], # Diana
])

# 标签
Y = np.array([
  1, # Alice
  0, # Bob
  1, # Charlie
  0, # Diana
])

def sigmoid(x):
    s = 1.0 / (1 + np.exp(-x))
    return s

def deriv_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def initialize_parameters():
    w1 = np.random.random()
    w2 = np.random.random()
    w3 = np.random.random()
    w4 = np.random.random()
    w5 = np.random.random()
    w6 = np.random.random()

    b1 = np.random.random()
    b2 = np.random.random()
    b3 = np.random.random()

    params = {
        "w1" : w1,
        "w2" : w2,
        "w3" : w3,
        "w4" : w4,
        "w5" : w5,
        "w6" : w6,
        "b1" : b1,
        "b2" : b2,
        "b3" : b3,
    }

    return w1,w2,w3,w4,w5,w6,b1,b2,b3

def propagate(x, w1,w2,w3,w4,w5,w6,b1,b2,b3):
    z1 = x[0] * w1 + x[1] * w2 + b1
    a1 = sigmoid(z1)

    z2 = x[0] * w3 + x[1] * w4 + b2
    a2 = sigmoid(z2)

    z3 = a1 * w5 + a2 * w6 + b3
    y_pred = sigmoid(z3) 

    return y_pred

def train(X, Y,  w1,w2,w3,w4,w5,w6,b1,b2,b3):
    learning_rate = 0.05
    epochs = 20000
    for epoch in range(epochs):
        Y_preds = []
        for x, y in zip(X, Y):
            # 向前传播
            z1 = x[0] * w1 + x[1] * w2 + b1
            a1 = sigmoid(z1)

            z2 = x[0] * w3 + x[1] * w4 + b2
            a2 = sigmoid(z2)

            z3 = a1 * w5 + a2 * w6 + b3
            y_pred = sigmoid(z3)

            Y_preds.append(y_pred)

            # 计算偏导数
            dldy_pred = -2 * (y - y_pred)
            dy_pred_dz3 = deriv_sigmoid(z3)
            dz3_dw5 = a1
            dz3_dw6 = a2
            dz3_db3 = 1

            dz3_da1 = w5
            dz3_da2 = w6

            da1_dz1 = deriv_sigmoid(z1)
            dz1_dw1 = x[0]
            dz1_dw2 = x[1]
            dz1_db1 = 1

            da2_dz2 = deriv_sigmoid(z2)
            dz2_dw3 = x[0]
            dz2_dw4 = x[1]
            dz2_db2 = 1

            dl_dw5 = dldy_pred * dy_pred_dz3 * dz3_dw5
            dl_dw6 = dldy_pred * dy_pred_dz3 * dz3_dw6
            dl_db3 = dldy_pred * dy_pred_dz3 * dz3_db3

            dl_dw1 = dldy_pred * dy_pred_dz3 * dz3_da1 * da1_dz1 * dz1_dw1
            dl_dw2 = dldy_pred * dy_pred_dz3 * dz3_da1 * da1_dz1 * dz1_dw2
            dl_db1 = dldy_pred * dy_pred_dz3 * dz3_da1 * da1_dz1 * dz1_db1

            dl_dw3 = dldy_pred * dy_pred_dz3 * dz3_da2 * da2_dz2 * dz2_dw3
            dl_dw4 = dldy_pred * dy_pred_dz3 * dz3_da2 * da2_dz2 * dz2_dw4
            dl_db2 = dldy_pred * dy_pred_dz3 * dz3_da2 * da2_dz2 * dz2_db2
            
            # 更新参数
            w1 = w1 - learning_rate * dl_dw1
            w2 = w2 - learning_rate * dl_dw2
            w3 = w3 - learning_rate * dl_dw3
            w4 = w4 - learning_rate * dl_dw4
            w5 = w5 - learning_rate * dl_dw5
            w6 = w6 - learning_rate * dl_dw6
            b1 = b1 - learning_rate * dl_db1
            b2 = b2 - learning_rate * dl_db2
            b3 = b3 - learning_rate * dl_db3

        if epoch % 100 == 0:
            Y_preds = np.array(Y_preds)
            all_loss = ((Y - Y_preds)**2).mean()
            print("the epoch is: " + str(epoch) + " The all_loss is: " + str(all_loss))
    
    return w1,w2,w3,w4,w5,w6,b1,b2,b3

w1,w2,w3,w4,w5,w6,b1,b2,b3 = initialize_parameters()
ww1,ww2,ww3,ww4,ww5,ww6,bb1,bb2,bb3 = train(X, Y, w1,w2,w3,w4,w5,w6,b1,b2,b3)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([5, 2])  # 155 pounds, 68 inches
jim = np.array([0, 0])  # 155 pounds, 68 inches

y_pred1 = propagate(emily, ww1,ww2,ww3,ww4,ww5,ww6,bb1,bb2,bb3)
y_pred2 = propagate(frank, ww1,ww2,ww3,ww4,ww5,ww6,bb1,bb2,bb3)
y_pred3 = propagate(jim, ww1,ww2,ww3,ww4,ww5,ww6,bb1,bb2,bb3)

print("Emily: %.3f" % y_pred1) # 0.951 - F
print("Frank: %.3f" % y_pred2) # 0.039 - M
print("Frank: %.3f" % y_pred3) # 0.039 - M


