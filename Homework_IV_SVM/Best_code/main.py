# -*- coding: UTF-8 -*-


def predict(x_test, m):
    """predict y using x and a pre-trained binary SVC model.  
    只需考虑二分类使用RBF kernel的情形
    """
    import numpy as np
    from sklearn.metrics import euclidean_distances
    
    # get parameters from fitted model
    intercept = m.intercept_[0]
    x_support = m.support_vectors_
    dual_coef = m.dual_coef_
    gamma = m._gamma

    # calculate pairwise distance using kernel
    kernel_res = np.exp(-gamma * euclidean_distances(x_test, x_support, squared=True))
    
    # calculate decision_function(x)
    f_arr = dual_coef * kernel_res # 2D broadcast
    f_arr = f_arr.sum(axis=1) + intercept
    
    return f_arr > 0