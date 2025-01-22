from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D

import matplotlib.pyplot as plt
import numpy as np
from time import time

from keras.api.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0).astype(np.uint32)
X_test = np.where(X_test > 75, 1, 0).astype(np.uint32)

Y_train_org = Y_train.astype(np.uint32)
Y_test_org = Y_test.astype(np.uint32)
Y_train = np.zeros((Y_train_org.shape[0], 10), dtype=np.uint32)
for i in range(Y_train_org.shape[0]):
	Y_train[i, Y_train_org[i]] = 1

Y_test = np.zeros((Y_test_org.shape[0], 10), dtype=np.uint32)
for i in range(Y_test_org.shape[0]):
	Y_test[i, Y_test_org[i]] = 1


X_train_combine = np.zeros(X_train.shape, dtype=np.uint32)

for e in range(X_train.shape[0]):
    random_ind = np.random.randint(0, X_train.shape[0])
    Y_train[e, Y_train_org[random_ind]] = 1

    print(f'{Y_train[e]=}')
    plt.figure()
    X_train_combine[e] = (X_train_combine[e] + X_train[random_ind])
    plt.imshow(X_train_combine[e])
    plt.show()

# X_train_combine = np.where(X_train_combine > 0, 1, 0).astype(np.uint32)


X_test_combine = np.zeros(X_test.shape, dtype=np.uint32)

for e in range(X_test.shape[0]):
    random_ind = np.random.randint(0, X_test.shape[0])
    Y_test[random_ind, Y_test_org[random_ind]] = 1
    X_test_combine[e] = (X_test_combine[e] + X_test[random_ind])

# X_test_combine = np.where(X_test_combine > 0, 1, 0).astype(np.uint32)



tm = MultiOutputConvolutionalTsetlinMachine2D(2500, 3125, 10, (28, 28, 1), (10, 10))

for i in range(5):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(X_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))





