import numpy as np
from sklearn.svm import SVC

X=np.genfromtxt('train_data.csv',delimiter=',')
y=np.genfromtxt('train_targets.csv')
Xt=np.genfromtxt('test_data.csv',delimiter=',')
yt=np.genfromtxt('test_targets.csv')

passed=0
gamma_list=[0.1,1,10]

for gamma in gamma_list:
    model=SVC(kernel="rbf", C=1, gamma=gamma) 
    model.fit(X,y)
    print('X.shape = ', X.shape)
    # print('coef_ =', model.coef_)
    # http://scikit-learn.org/stable/modules/svm.html
    print('model.classes_ = ' ,model.classes_)
    print('model.n_support_ = ',model.n_support_) # represents the number of support vectors per class
    print('dual_coef_ =', model.dual_coef_) # coef_ is only available when using a linear kernel
    support_indices = np.cumsum(model.n_support_)
    print('support_indices[0] = ', support_indices[0])
    print('------> ', model.dual_coef_[0:1])
    print('------> ', model.dual_coef_.shape )
    y_pred_true=model.predict(Xt)

    model.predict=None

    ##################################################################
    # you should implement the "predict" function in your main.py code
    # we will import and use it as following:
    from main import predict
    y_pred=predict(Xt,model)
    ##################################################################

    passed+=sum(y_pred==y_pred_true)/y_pred_true.shape[0]
    
if passed==len(gamma_list):
    print('passed')
else:
    print('failed')



