from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
import cv2
from time import time

from keras.api.datasets import fashion_mnist

if __name__ == "__main__":
	(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

	X_train = np.copy(X_train)
	X_test = np.copy(X_test)

	for i in range(X_train.shape[0]):
		X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	for i in range(X_test.shape[0]):
		X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	X_train = X_train.reshape((X_train.shape[0], -1))
	X_test = X_test.reshape((X_test.shape[0], -1))


	tm = MultiClassConvolutionalTsetlinMachine2D(
		number_of_clauses=40000,
		T=15000,
		s=10,
		dim=(28, 28, 1),
		patch_dim=(10, 10),
	)

	for i in range(30):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result_test = 100 * (tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		result_train = 100 * (tm.predict(X_train) == Y_train).mean()

		print(
			f"Epoch {i + 1} | Train Time: {stop_training - start_training:.2f}s, Test Time: {stop_testing - start_testing:.2f}s | Train Accuracy: {result_train:.4f}, Test Accuracy: {result_test:.4f}"
		)
