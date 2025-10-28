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
	train_dir = "~/work/Datasets/MMNIST/train/m0"
	train_img_list = os.listdir(train_dir)
	X_train_org = []
	Y_train = []
	for path in train_img_list:
		Y_train.append(int(path[-5]))
		img = Image.open(path).convert("RGB").resize((28, 28))
		X_train_org.append(np.array(img))
	X_train_org = np.array(X_train_org)
	Y_train = np.array(Y_train, dtype=int)

	test_dir = "~/work/Datasets/MMNIST/test/m0"
	test_img_list = os.listdir(test_dir)
	X_test_org = []
	Y_test = []
	for path in test_img_list:
		Y_test.append(int(path[-5]))
		img = Image.open(path).convert("RGB").resize((28, 28))
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


def create_graphs(X, graph_args, init_with=None):
	graphs = Graphs(number_of_graphs=X.shape[0], symbols=symbols, init_with=init_with, **graph_args)

	for graph_id in range(X.shape[0]):
		graphs.set_number_of_graph_nodes(graph_id, number_of_nodes)

	graphs.prepare_node_configuration()

	for graph_id in range(X.shape[0]):
		for node_id in range(graphs.number_of_graph_nodes[graph_id]):
			graphs.add_graph_node(graph_id, node_id, 0)

	graphs.prepare_edge_configuration()

	for graph_id in range(X.shape[0]):
		windows = view_as_windows(X[graph_id, :, :, :], (patch_size, patch_size, channel_size))
		for q in range(dim):
			for r in range(dim):
				node_id = q * dim + r

				patch = windows[q, r, 0].reshape(-1).astype(np.uint32)
				for k in patch.nonzero()[0]:
					graphs.add_graph_node_property(graph_id, node_id, k)

				graphs.add_graph_node_property(graph_id, node_id, f"X{q}")
				graphs.add_graph_node_property(graph_id, node_id, f"Y{r}")

	graphs.encode()

	return graphs


# Read the data
resolution = 8
X_train, Y_train, X_test, Y_test = preprocess_cifar10_data(resolution)
print(f"X-shape:{X_train.shape}")


# Parameters
# ==========
train_folder = "train_larger"
test_folder = "test_larger"

graph_args = {
	"hypervector_size": 1024,
	"hypervector_bits": 8,
	"double_hashing": False,
}

batch_size = 1000

patch_size = 5
dim = X_train.shape[1] - patch_size + 1
channel_size = X_train.shape[-1]
number_of_nodes = dim * dim

# Get all possible patch symbols encoded as strings
# symbols = np.array(list(product([0, 1], [0, 1], repeat=5)))
# symbols = symbols[symbols[:, -1] == 0, :-1]
# symbols = [''.join([str(x) for x in s]) for s in symbols]

# Add pixelvalues as symbols
symbols: list[int | str] = [i for i in range(patch_size * patch_size * channel_size)]

# Add rows and columns as symbols
for d in range(dim):
	symbols.append(f"X{d}")
	symbols.append(f"Y{d}")

# Create train batches
skf = StratifiedKFold(n_splits=int(X_train.shape[0] / batch_size))

print("Create training graphs... ")
for i, (_, train_index) in enumerate(skf.split(X_train, Y_train)):
	print(f"Creating fold {i} [{(i+1)*train_index.shape[0]}/{X_train.shape[0]}]:")
	X_batch, Y_batch = X_train[train_index], Y_train[train_index]
	if i == 0:
		graphs_batch_0 = create_graphs(X_batch, graph_args, init_with=None)
		# Save batch
		with open(f"{train_folder}/batch_0.pickle", "wb") as f:
			pickle.dump((graphs_batch_0, Y_batch), f)
	else:
		graphs_batch = create_graphs(X_batch, graph_args, init_with=graphs_batch_0)
		graphs_batch.init_with = None
		# Save batch
		with open(f"{train_folder}/batch_{i}.pickle", "wb") as f:
			pickle.dump((graphs_batch, Y_batch), f)

print("Train graphs created")


# Create test batches
skf = StratifiedKFold(n_splits=int(X_test.shape[0] / batch_size))

print("Create test graphs... ")

for i, (_, test_index) in enumerate(skf.split(X_test, Y_test)):
	print(f"Creating fold {i} [{(i+1)*test_index.shape[0]}/{X_test.shape[0]}]:")
	X_batch, Y_batch = X_test[test_index], Y_test[test_index]
	graphs_batch = create_graphs(X_batch, graph_args, init_with=graphs_batch_0)
	graphs_batch.init_with = None
	# Save batch
	with open(f"{test_folder}/batch_{i}.pickle", "wb") as f:
		pickle.dump((graphs_batch, Y_batch), f)

print("Test graphs created")

print("Done")
