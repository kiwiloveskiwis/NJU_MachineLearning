import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression

eps = 1e-6
def transform(target):
    new_y = np.array(target)
    new_y[new_y == 0] = -1
    return new_y

target = np.genfromtxt('targets.csv', delimiter=',').flatten()
data = np.genfromtxt('data.csv', delimiter=',')
data = normalize(data)
kf = KFold(n_splits=10)
kf.get_n_splits(data)
ensemble_size = [1, 5, 10, 100]


for e_size in ensemble_size:
    cc = 0
    for train_index, test_index in kf.split(data):
        cc += 1
        X = data[train_index, :]
        y = target[train_index]
        with open('experiments/base%d_fold%d.csv' % (e_size, cc), 'w') as file:
            sample_weight = np.ones_like(y)
            sample_weight /= sample_weight.sum()

            logistic_classifier = []
            alpha = []
            for i in range(e_size):
                logistic_classifier.append(LogisticRegression(C=1000))
                logistic_classifier[i] = logistic_classifier[i].fit(X, y, sample_weight)
                score = logistic_classifier[i].score(X, y, sample_weight)
                # print(score)
                if score <= 0.5:
                    break
                alpha.append(np.log((score + eps) / (1 - score + eps)) / 2)
                h = logistic_classifier[i].predict(X)
                sample_weight *= np.exp(-alpha[i] * transform(h) * transform(y))
                sample_weight /= np.sum(sample_weight)
            y_pred = np.zeros_like(target[test_index])
            X_test = data[test_index, :]
            # print(len(alpha))
            for i in range(len(alpha)):
                y_pred_cur = logistic_classifier[i].predict(X_test)
                y_pred += alpha[i] * transform(y_pred_cur)
            y_pred[y_pred > 0] = 1
            y_pred[y_pred <= 0] = 0
            for i in range(test_index.size):
                file.write(str(test_index[i] + 1) + "," + str(int(y_pred[i])) + "\r\n")
            file.close()
