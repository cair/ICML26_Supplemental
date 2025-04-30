# Written by Ylva 
from time import time

import cv2
import numpy as np
from keras.api.datasets import cifar10
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from sklearn.metrics import accuracy_score

clauses = 160000
T = 30000
s = 20.0
patch_size = 8
resolution = 8
epochs = 30


def horizontal_flip(image):
    return cv2.flip(image, 1)


augmented_images = []
augmented_labels = []

labels = [b"airplane", b"automobile", b"bird", b"cat", b"deer", b"dog", b"frog", b"horse", b"ship", b"truck"]

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

for i in range(len(X_train_org)):
    image = X_train_org[i]
    label = Y_train[i]

    # Original image and label
    augmented_images.append(image)
    augmented_labels.append(label)

    augmented_images.append(horizontal_flip(image))
    augmented_labels.append(label)

X_train_aug = np.array(augmented_images)
Y_train = np.array(augmented_labels).reshape(-1, 1)

X_train = np.empty(
    (
        X_train_aug.shape[0],
        X_train_aug.shape[1],
        X_train_aug.shape[2],
        X_train_aug.shape[3],
        resolution,
    ),
    dtype=np.uint8,
)
for z in range(resolution):
    X_train[:, :, :, :, z] = X_train_aug[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_test = np.empty(
    (
        X_test_org.shape[0],
        X_test_org.shape[1],
        X_test_org.shape[2],
        X_test_org.shape[3],
        resolution,
    ),
    dtype=np.uint8,
)
for z in range(resolution):
    X_test[:, :, :, :, z] = X_test_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_train = X_train.reshape(
    (
        X_train_aug.shape[0],
        X_train_aug.shape[1],
        X_train_aug.shape[2],
        3 * resolution,
    )
)
X_test = X_test.reshape(
    (
        X_test_org.shape[0],
        X_test_org.shape[1],
        X_test_org.shape[2],
        3 * resolution,
    )
)

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

dim = (X_train.shape[1], X_train.shape[2], 3 * resolution)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

f = open("./CIFAR10/cifar10_%.1f_%d_%d_%d.txt" % (s, clauses, T, patch_size), "w+")


tm = MultiClassConvolutionalTsetlinMachine2D(
    clauses,
    T,
    s,
    dim,
    (patch_size, patch_size),
)

for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100 * accuracy_score(Y_test, tm.predict(X_test))
    stop_testing = time()

    result_train = 100 * accuracy_score(Y_train, tm.predict(X_train))

    print(
        "%d %.2f %.2f %.2f %.2f"
        % (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
    )
    print(
        "%d %.2f %.2f %.2f %.2f"
        % (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing),
        file=f,
    )
    f.flush()

f.close()
