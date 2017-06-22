import numpy as np
import pickle
yt=pickle.load(open('test_targets.pkl','rb'))
y_pred=np.genfromtxt('test_predictions.csv')
print(sum(yt==y_pred)/yt.shape[0])


