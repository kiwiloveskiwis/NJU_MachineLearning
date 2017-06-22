import numpy as np
target=np.genfromtxt('targets.csv')
acc=[]
indices=[]
for i in range(1,11):
    fold=np.genfromtxt('fold%d.csv'%i,delimiter=',',dtype=np.int)
    indices.extend(list(fold[:,0]))
    accuracy=sum(target[fold[:,0]-1]==fold[:,1])/fold.shape[0]
    acc.append(accuracy)

if len(set(indices))==len(indices) and len(indices)==target.shape[0]:
    print(np.array(acc).mean())
else:
    print('not a valid cross-validation')
