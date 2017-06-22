
# coding: utf-8

# In[3]:

import numpy as np
from numpy import seterr
from numpy import transpose as trans
import sklearn, sys
from sklearn import linear_model
from sklearn.model_selection import KFold
np.seterr(over='raise', under='ignore', divide='raise')


# In[4]:

data = np.genfromtxt('data.csv', delimiter = ',')
data = sklearn.preprocessing.normalize(data)
target = np.genfromtxt('targets.csv', delimiter = ',')
data_hat = np.append(data, np.ones([data.shape[0], 1]), 1)


# In[5]:

def sigmoid(x):
    max_elem = max(-x) 
#     print(max_elem)
#     print(np.exp(0 - max_elem))
#     print(np.exp(- x - max_elem))
#     print(min(np.exp(0 - max_elem) + np.exp(- x - max_elem)))
    try: 
        res = -(np.log(np.exp(0 - max_elem) + np.exp(- x - max_elem)) + max_elem ) # Only underflow could happen
        res = np.exp(res)
    except Exception as e:
        res = 0
    return res


# In[6]:

def hypo(x_hat, beta):
    return sigmoid(np.matmul(x_hat, beta))


# In[7]:

def prob_eq_one(x_hat, beta):
    max_elem = max(np.matmul(x_hat, beta))
    try:
        log =  np.matmul(x_hat, beta) - (np.log(np.exp(0 - max_elem) + np.exp(np.matmul(x_hat, beta) - max_elem)) + max_elem)
        res = trans(np.exp(log))
    except:
        res =  trans(np.zeros([x_hat.shape[0]]))
    # res_2 = np.exp(np.matmul(x_hat, beta)) / (1 + np.exp(np.matmul(x_hat, beta))) # Might Overflow
    # print(res - res_2)
    return res


# In[8]:

def error(X, beta, y):
    return np.count_nonzero(abs(np.round(hypo(X, beta)) - y))


# In[9]:

def grad(x_hat, beta, y, speed=1):
    p1 = prob_eq_one(x_hat, beta)
    # print(1+np.exp(np.matmul(x_hat, beta)))
    # print(np.exp(np.matmul(x_hat, beta)))
    # print(np.exp(np.matmul(x_hat, beta)) / (1 + np.exp(np.matmul(x_hat, beta))))
    ans = - np.matmul(trans(y) - p1,  x_hat)
    return ans*speed


# In[10]:

def hessian(x_hat, beta):
    m, d = x_hat.shape[0], x_hat.shape[1]

    p1 = prob_eq_one(x_hat, beta)
    p1_mult = p1 * (1 - p1)
    res = np.zeros([d, d])
    for i in range(m):
        #print(x_hat[i, :].size)
        #print(trans(x_hat[i, :]).size)
        #print(np.outer(x_hat[i, :], trans(x_hat[i, :])))
        res += p1_mult[i] * np.outer(x_hat[i, :], x_hat[i, :])
    return res


# In[36]:

def my_train(X, y, iter, beta):
    for i in range(iter):
        p1 = prob_eq_one(X, beta)
        hess = hessian(X, beta)
        beta_save = beta
        try:
            inv = np.linalg.inv(hess)
            beta -= np.matmul(inv, grad(X, beta, y))
        except Exception as e:
            if(np.max(hess) < np.exp(-100)):
                break
            else:
                beta = beta_save - grad(X, beta, y)
    print(error(data_hat, beta, target))
    return beta


# In[37]:

# Train the model with K-Fold
kf = KFold(n_splits=10)
kf.get_n_splits(data)


# In[38]:

for k, (train, test) in enumerate(kf.split(data, target)):
    m = data_hat[test].shape[0]
    beta = np.zeros(data_hat.shape[1], dtype=float)
    
    iter_beta = my_train(data_hat[train], target[train], 100,  beta)
    
    to_save = np.round(hypo(data_hat[test], iter_beta)).reshape(m, 1)
    to_save = np.dstack((test + 1 , to_save[:,0]))[0]
    np.savetxt('fold%d.csv'%(k+1), to_save, "%d,%d", delimiter=',')


# In[40]:

#Test on all dataset
# beta = np.zeros(data_hat.shape[1], dtype=float)
# beta = my_train(data_hat, target, 100, beta)
