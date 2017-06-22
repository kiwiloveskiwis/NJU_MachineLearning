
# coding: utf-8


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[11]:

data = np.genfromtxt('train_data.csv', delimiter = ',')
#data = sklearn.preprocessing.normalize(data)
target = np.genfromtxt('train_targets.csv', delimiter = ',') # (3000, )
data_hat = np.append(data, np.ones([data.shape[0], 1]), 1)

test_data = np.genfromtxt('test_data.csv', delimiter = ',')
#test_data = sklearn.preprocessing.normalize(test_data)
test_target = np.genfromtxt('test_targets.csv', delimiter = ',')

(m, d) = data.shape
m_test = test_data.shape[0]
k = 10
test_y = np.zeros([m_test, k])
y = np.zeros([m, k])
for i in range(m):
    y[i, int(target[i])] = 1
    if(i < m_test):
        test_y[i, int(test_target[i])] = 1


# Train the model
model = Sequential()
model.add(Dense(units=m, input_dim=d))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(data, y, nb_epoch=15, batch_size=10)

scores = model.evaluate(test_data, test_y)
pred = model.predict(test_data)
res = pred.argmax(axis=1)

np.savetxt('test_predictions_library.csv', res, "%d", delimiter=',')

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))