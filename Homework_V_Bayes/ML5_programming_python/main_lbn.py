import pickle
import numpy as np
import math
from collections import Counter
import time


start_time = time.time()

X = pickle.load(open('train_data.pkl', 'rb')).todense()  # unsupported in Python 2
y = pickle.load(open('train_targets.pkl', 'rb'))
Xt = pickle.load(open('test_data.pkl', 'rb')).todense()

"""print(type(X))
print(X.shape)
print(y.shape)
print(Xt.shape)"""

# convert data from np.matrix to np.ndarray
X = np.asarray(X)
y = np.asarray(y)
Xt = np.asarray(Xt)
"""for i in range(5000):
    print(i, Counter(X[:, i]))"""

#--------------------------------------------------------------
# contants and function definitions

N_DISCRETE = 2500
N_DISCRETE_VALUES = 2
LAPLACE_CORRECTION_ALPHA = 1.  # laplace correction coefficient!
STD_RATIO = 1e-4  # deal with zero std!


def log_norm_pdf_vectorize(x_arr, means, stds):
    assert len(x_arr) == len(means) and len(x_arr) == len(stds)
    return (- np.power(x_arr - means, 2) / (2 * np.power(stds, 2))
            - math.log(2 * math.pi) / 2
            - np.log(stds))


def train_naive_bayes(xtrain, ytrain):
    n_total = xtrain.shape[0]
    class_data = dict()
    y_values = []

    # store data from different classes in a dictionary
    for c, n in Counter(ytrain).items():
        y_values.append(c)
        class_data[c] = {'n': n, 'index': (ytrain == c), 'prior': n / n_total, 'discrete_prob': None, 'pdf_arr': None}
    class_data['y_values'] = y_values
    class_data['n_total'] = n_total

    for c in class_data['y_values']:
        mask = class_data[c]['index']
        n_samples = class_data[c]['n']

        discrete_features, continuous_features = xtrain[mask, : N_DISCRETE], xtrain[mask, N_DISCRETE:]

        # for each discrete feature, calculate P(x = 0 | y) and P(x = 1 | y)
        class_data[c]['discrete_prob'] = np.empty((N_DISCRETE_VALUES, N_DISCRETE))
        for i in range(N_DISCRETE_VALUES):
            class_data[c]['discrete_prob'][i, :] = ((np.sum(discrete_features == i, axis=0) + LAPLACE_CORRECTION_ALPHA)
                                                    / (n_samples + LAPLACE_CORRECTION_ALPHA * N_DISCRETE_VALUES))  # with lapace smoothing

        # for each continuous feature, calculate mean and std
        means, stds = continuous_features.mean(axis=0), continuous_features.std(axis=0)
        stds += 1e-3
        class_data[c]['means'], class_data[c]['stds'] = means, stds

    return class_data


def calc_posterior(x, oneclass_data):
    """return posterior probability of sample x.

    Parameters
    -----------
    oneclass_data : dict. oneclass_data = class_data[c], with c from y_values
    x : np.ndarray. 1D sample features.

    Returns
    --------
    posterior : float.

    """
    x_discrete, x_continuous = x[: N_DISCRETE].astype(int), x[N_DISCRETE:]  # convert to int, in order to be index

    # calculate discrete likelihood
    probabilities_1 = oneclass_data['discrete_prob'][x_discrete, np.arange(N_DISCRETE)]  # advanced slicing!
    log_likelihood_discrete = np.sum(np.log(probabilities_1))

    # calculate continuous likelihood
    probabilities_2 = log_norm_pdf_vectorize(x_continuous, oneclass_data['means'], oneclass_data['stds'])
    log_likelihood_continuous = np.sum(probabilities_2)

    # calculate P(y | x) = P(x | y) * P(y)
    posterior = (math.log(oneclass_data['prior'])
                 + log_likelihood_discrete
                 + log_likelihood_continuous  # * 1e-5 # prevent this from being dominant
                 )

    return posterior

#--------------------------------------------------------------
# training

class_data = train_naive_bayes(X, y)
print("Training completed.")

#--------------------------------------------------------------
# testing (predicting)

# load test data, loop for each data for each class
posterior_matrix = np.empty((Xt.shape[0], len(class_data['y_values'])))
for i, x in enumerate(Xt):
    for c in class_data['y_values']:
        posterior_matrix[i, c] = calc_posterior(x, class_data[c])

# compare and classify
y_pred = np.argmax(posterior_matrix, axis=1).reshape(-1, 1)

#--------------------------------------------------------------
# save data
np.savetxt('test_predictions.csv', y_pred, fmt='%s', delimiter=',', newline='\n')

time_cost = time.time() - start_time
print("\nPrediction and output completed. \nTotal time cost: {:6.3f} seconds.".format(time_cost))
