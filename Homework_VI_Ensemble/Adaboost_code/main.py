
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression


# In[2]:

data = np.genfromtxt('data.csv',delimiter=',') # (476, 166)
target = np.genfromtxt('targets.csv') # (476,)
target[target == 0] = -1 # transform


# In[3]:

kf = KFold(n_splits=10)
kf.get_n_splits(data)


# In[20]:

for learner_num in [1, 5, 10, 100]:
    for k, (train_idx, test_idx) in enumerate(kf.split(data, target)):
        X, y = data[train_idx, :], target[train_idx]
        Xt, yt = data[test_idx, :], target[test_idx]

        sample_weight = np.ones(y.shape) / len(y)

        base_classifiers = []
        alpha = []
        for i in range(learner_num):
            base_classifiers.append(LogisticRegression(C=1000).fit(X, y, sample_weight))
            score = base_classifiers[i].score(X, y, sample_weight)
            if score <= 0.5:
                print("Breaked in base_num = %d, fold = %d"%(learner_num, i))
                break
            alpha.append(np.log((score) / (1 - score)) / 2)
            h = base_classifiers[i].predict(X)
            sample_weight *= np.exp(-alpha[i] * h * y)
            sample_weight /= np.sum(sample_weight)
        y_pred = np.zeros(yt.shape)
        for i in range(len(alpha)):
            y_pred += alpha[i] * base_classifiers[i].predict(Xt)
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
        to_save = np.dstack((test_idx + 1 , y_pred))[0]
        np.savetxt('experiments/base%d_fold%d.csv'%(learner_num , k + 1), to_save, '%d, %d', delimiter=',')
