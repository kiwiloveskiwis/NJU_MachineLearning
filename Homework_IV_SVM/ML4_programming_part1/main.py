# Reference: http://scikit-learn.org/stable/modules/svm.html#multi-class-classification
# The decision function section
import numpy as np
from sklearn.svm import SVC

def predict(Xt, model):
	b = model.intercept_
	# print('model.intercept_ = ', b)
	alpha = model.dual_coef_[0]
	num_sv = alpha.shape[0] 

	sv = model.support_vectors_
	gamma = model.gamma
	
	(l, d) = Xt.shape
	pred = np.zeros(l)

	for k in range(0, l):
		pred[k] += b
		for i in range(0, num_sv):
			#print(np.dot((sv[i]-Xt[k]),(sv[i]-Xt[k])))
			pred[k] = pred[k] + alpha[i] * np.exp(-gamma * np.dot((sv[i] - Xt[k]), (sv[i] - Xt[k])))

	return pred >= 0