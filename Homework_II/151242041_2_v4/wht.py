import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import math


iter_num = 200


def p1(X, beta):
    ans = np.ones((X.shape[0], 1))
    for i in range(X.shape[0]):
        try:
            tmp = np.exp(np.dot(X[i, :], beta))
            res = tmp / (tmp + 1)
            ans[i, 0] = res
        except Exception as e:
            ans[i, 0] = 1
        if math.isnan(ans[i, 0]) or math.isinf(ans[i, 0]):
            ans[i, 0] = 1
    return ans


def d_beta(X, beta, p1_val, y):
    res = -np.dot(X.transpose(), y - p1_val)
    # res = np.zeros((X.shape[1], 1))
    # for i in range(X.shape[0]):
    #     res -= X[i, :].reshape(-1, 1) * (y[i, 0] - p1_val[i, 0])
    return res


def H_beta(X, beta, p1_val):
    res = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        res += np.dot(X[i, :].reshape(-1, 1), X[i, :].reshape(1, -1))\
            * (p1_val[i, 0] * (1 - p1_val[i, 0]))
    return res


def accur(X, beta, y):
    hit = 0
    for i in range(X.shape[0]):
        y_hat = 0
        if np.dot(X[i, :].reshape(1, -1), beta.reshape(-1, 1)) > 0:
            y_hat = 1
        if y_hat == y[i, 0]:
            hit += 1
    return 1.0 * hit / X.shape[0]


def train(X, y):
    beta = np.zeros(X.shape[1]).reshape(-1, 1)
    for i in range(iter_num):
        p1_val = p1(X, beta)
        hess = H_beta(X, beta, p1_val)
        if np.max(hess) < 1e-5:
            break
        beta_bak = beta
        beta = beta - np.dot(np.linalg.inv(hess), d_beta(X, beta, p1_val, y))
        if math.isnan(np.sum(beta)) or math.isinf(np.sum(beta)):
            beta = beta_bak
            break
    return beta


def predict(X, beta):
    y = np.zeros((X.shape[0], 1), dtype=int)
    for i in range(X.shape[0]):
        if np.dot(X[i, :].reshape(1, -1), beta.reshape(-1, 1)) > 0:
            y[i, 0] = 1
    return y


y = np.genfromtxt('targets.csv', delimiter=',').reshape(-1, 1)
X = np.genfromtxt('data.csv', delimiter=',')
X = normalize(X)
X = np.hstack((X, np.ones(X.shape[0]).reshape((-1, 1))))
kf = KFold(n_splits=10)
kf.get_n_splits(X)


cc = 0

for train_index, test_index in kf.split(X):
    beta = train(X[train_index, :], y[train_index, :])
    y_hat = predict(X[test_index, :], beta)
    with open('fold%d.csv' % (cc + 1), 'w') as file:
        for i in range(test_index.size):
            file.write(str(cc * test_index.size + i + 1) + "," + str(y_hat[i, 0]) + "\r\n")
    cc += 1
