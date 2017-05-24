import numpy as np
target=np.genfromtxt('test_targets.csv')
prediction1=np.genfromtxt('test_predictions.csv')
prediction2=np.genfromtxt('test_predictions_library.csv')
print("naive =", sum(target==prediction1)/prediction1.shape[0])
print("lib   = ",sum(target==prediction2)/prediction2.shape[0])
