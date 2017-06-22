
# coding: utf-8

# In[1]:

import numpy as np
import sys, time, math

# In[3]:

import pickle
X=pickle.load(open('train_data.pkl','rb')).todense() # unsupported in Python 2
y=pickle.load(open('train_targets.pkl','rb'))  # 652
Xt=pickle.load(open('test_data.pkl','rb')).todense() # 488, 5000, np.matrix
yt=pickle.load(open('test_targets.pkl','rb')) # 488,

def separateByClass(dataset, data_y):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (y[i] not in separated):
            separated[data_y[i]] = np.array(vector)
        else:
            separated[data_y[i]] = np.append(separated[data_y[i]], vector, axis=0)
    return separated


# In[6]:

sepa = separateByClass(X, y) # dict

# In[7]:

alpha = 1
cval_feature_prob = {}
default_feature_prob = {}
for cval in list(sepa):
    count_same_5k = {}
    count_default = {}
    for i in range(X.shape[1]):
        feature = sepa[cval][:,i] # 127, 1
        uni_all = np.unique([feature])
        uni_count = uni_all.shape[0]
        count_same = {}
        for j in range(uni_count):
            count_same[uni_all[j]] = np.log((len(feature) - np.count_nonzero(feature - uni_all[j] ) + alpha) / (len(feature) + alpha * uni_count) )
        count_default[i] = np.log( alpha / (len(feature) + alpha * uni_count) )
        count_same_5k[i] = count_same
    cval_feature_prob[cval] = count_same_5k # should be 5000 * perfeature_unique_num * feature_unique_count
    default_feature_prob[cval] = count_default
# len(cval_feature_prob) # 5 * 5000 * perfeature_unique_num * feature_unique_count
# default_feature_prob

mean_all = np.zeros([len(sepa), X.shape[1]])  # 5 * 5000
stdev_all = np.zeros([len(sepa), X.shape[1]]) # 5 * 5000
for cval in list(sepa):
    mean_all[cval] = np.mean(sepa[cval], axis=0)
    stdev_all[cval] = sepa[cval].std(axis=0)


# In[21]:

len_all = [len(sepa[i]) for i in range(len(sepa))]


# In[14]:
def logGaussianProb(x, mu, stdev):
    logExp = -(np.power(x - mu, 2) / (2 * np.power(stdev, 2)))
    return -np.log((np.sqrt(2 * np.pi) * stdev)) + logExp

start_time = time.time()

CONTINUOUS_RATIO = 1 / 2852555


import code
def predict(data, sepa):
    pred = np.zeros(data.shape[0])
    print(data.shape[0])
    # cont_disc_ratio = np.zeros([data.shape[0], len(list(sepa))]) 
    for ins in range(0, data.shape[0]):
        cprob = np.zeros(len(list(sepa)))
        best_label = 0
        for cval in list(sepa):             # classvalue
            cmat = sepa[cval]   
            discrete_prob_sum = 0          
            continuous_prob_sum = 0        

            for i in range(data.shape[1]):  
                if i < 2500:
                    add = cval_feature_prob[cval][i].get(data[ins, i])
                    if(None == add):
                        add = default_feature_prob[cval][i]
                    discrete_prob_sum += add
                else:  
                    mean, var = mean_all[cval][i], stdev_all[cval][i]
                    if(var < 1e-9) :
                        var = 1e-3      # adjust to a reasonable value. accuracy jumps from 40 to 68
                    continuous_prob_sum += logGaussianProb(data[ins, i], mean, var)
            # cont_disc_ratio[ins, cval] = continuous_prob_sum / discrete_prob_sum

            cprob[cval] = np.log(cmat.shape[0] / data.shape[0]) + discrete_prob_sum + continuous_prob_sum * CONTINUOUS_RATIO

        pred[ins] = np.argmax(cprob)
    # code.interact(local=locals())
    return pred

pred = predict(Xt, sepa)
print('Cost time = ', time.time() - start_time)
np.savetxt('test_predictions.csv', pred, "%d", delimiter=',')


