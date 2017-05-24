import numpy as np


def distance_func(x_i, model):
    gamma = model.get_params()['gamma']
    res = model.intercept_[0]
    # print(model.support_)
    for ii, i in enumerate(model.support_):
        # print(y[i])
        x = model.support_vectors_[ii, :]
        # print(x - x_i)
        res += model.dual_coef_[0][ii]\
            * np.exp(-gamma * np.dot(x - x_i, x - x_i))
        # print(res)
    return res


def predict(X, model):
    res = []
    for i in range(X.shape[0]):
        res.append(distance_func(X[i, :], model))
    return np.array(res) > 0
