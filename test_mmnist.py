import argparse
import os
import cv2

import pickle
from time import time

from keras.api.datasets import cifar10
import numpy as np
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
from GraphTsetlinMachine.graphs import Graphs
import matplotlib.pyplot as plt


def preprocess_mmnist_data(resolution):
	"""
	Preprocess CIFAR-10 images and labels.

	Parameters:
	- resolution: The number of bins to quantize the pixel values into.
	- animals: An array of CIFAR-10 label indices to be considered as positive samples (1),
	    with all others as negative samples (0).

	Returns:
	- X_train, Y_train: Processed training images and labels.
	- X_test, Y_test: Processed testing images and labels.
	"""
	# (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
	train_dir = r"/home/mayurks/work/Datasets/MMNIST/train/m0/"
	train_img_list = os.listdir(train_dir)
	X_train_org = []
	Y_train = []
	for path in train_img_list:
		Y_train.append(int(path[-5]))
		img = Image.open(train_dir + path).convert("RGB").resize((28, 28))
		X_train_org.append(np.array(img))
	X_train_org = np.array(X_train_org)
	Y_train = np.array(Y_train, dtype=int)

	test_dir = r"/home/mayurks/work/Datasets/MMNIST/test/m0/"
	test_img_list = os.listdir(test_dir)
	X_test_org = []
	Y_test = []
	for path in test_img_list:
		Y_test.append(int(path[-5]))
		img = Image.open(test_dir + path).convert("RGB").resize((28, 28))
		X_test_org.append(np.array(img))
	X_test_org = np.array(X_test_org)
	Y_test = np.array(Y_test, dtype=int)

	# Flatten Y arrays
	Y_train, Y_test = Y_train.reshape(-1), Y_test.reshape(-1)

	# Initialize empty arrays for quantized images
	X_train = np.empty(
		(X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
		dtype=np.uint8,
	)

	X_test = np.empty(
		(X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution), dtype=np.uint8
	)

	# Quantize pixel values
	for z in range(resolution):
		threshold = (z + 1) * 255 / (resolution + 1)
		X_train[:, :, :, :, z] = X_train_org >= threshold
		X_test[:, :, :, :, z] = X_test_org >= threshold

	# Reshape quantized images and convert to uint32
	X_train = X_train.reshape(
		(X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3 * resolution)
	).astype(np.uint32)
	X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3 * resolution)).astype(
		np.uint32
	)

	return X_train, Y_train, X_test, Y_test


def f(b):
	ret = np.empty(3)
	t = len(b) // 3
	for i in range(3):
		nz = np.argwhere(b[i * t : (i + 1) * t]).ravel()
		ret[i] = nz[-1] if len(nz) > 0 else 0
	return ret


def unbinarize(a):
	return np.apply_along_axis(f, -1, a)


if __name__ == "__main__":
	X_train, Y_train, X_test, Y_test = preprocess_mmnist_data(8)

	img = X_train[0]
	unbin = unbinarize(img) // 8

	plt.imshow(unbin)
	plt.show()
