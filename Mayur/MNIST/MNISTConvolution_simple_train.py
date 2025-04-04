import argparse
import pickle
from lzma import LZMAFile
from time import time

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from keras.api.datasets import mnist
from skimage.util import view_as_windows
from tqdm import tqdm


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=2500, type=int)
    parser.add_argument("--T", default=3125, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--save_path", default="./MNIST/", type=str)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


def generate_graphs(X, graph_args):
    graphs = Graphs(**graph_args)

    num_graphs = X.shape[0]
    dim = X.shape[1]
    num_nodes = dim * dim

    for id in range(num_graphs):
        graphs.set_number_of_graph_nodes(id, num_nodes)

    graphs.prepare_node_configuration()

    for graph_id in tqdm(
        range(X_train.shape[0]), desc="Adding graph nodes", leave=False
    ):
        for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
            graphs_train.add_graph_node(graph_id, node_id, 0)

    graphs_train.prepare_edge_configuration()

    for graph_id in tqdm(
        range(X_train.shape[0]), desc="Adding node symbols", leave=False
    ):
        windows = view_as_windows(X_train[graph_id, :, :], (patch_size, patch_size))
        for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = q * dim + r

                patch = windows[q, r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_train.add_graph_node_property(graph_id, node_id, k)

                graphs_train.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
                graphs_train.add_graph_node_property(graph_id, node_id, "R:%d" % (r))

    graphs.encode()

    return graphs


if __name__ == "__main__":
    args = default_args()

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    N_samples = 60000
    X_train = X_train[:N_samples]
    Y_train = Y_train[:N_samples]
    X_test = X_test[:N_samples]
    Y_test = Y_test[:N_samples]

    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)
    Y_train = Y_train.astype(np.uint32)
    Y_test = Y_test.astype(np.uint32)

    patch_size = 10
    dim = 28 - patch_size + 1
    number_of_nodes = dim * dim
    symbols = []

    # Column and row symbols
    for i in range(dim):
        symbols.append("C:%d" % (i))
        symbols.append("R:%d" % (i))

    # Patch pixel symbols
    for i in range(patch_size * patch_size):
        symbols.append(i)

    graphs_train = generate_graphs(
        X_train,
        dict(
            number_of_graphs=X_train.shape[0],
            symbols=symbols,
            hypervector_size=args.hypervector_size,
            hypervector_bits=args.hypervector_bits,
            double_hashing=args.double_hashing,
        ),
    )

    print("Training data produced")

    graphs_test = generate_graphs(
        X_test,
        dict(
            number_of_graphs=X_test.shape[0],
            init_with=graphs_train,
        ),
    )

    print("Testing data produced")

    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        depth=1,
        message_size=args.message_size,
        message_bits=args.message_bits,
        double_hashing=False,
    )

    for i in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

        print(
            "%d %.2f %.2f %.2f %.2f"
            % (
                i,
                result_train,
                result_test,
                stop_training - start_training,
                stop_testing - start_testing,
            )
        )

        print(
            f"Epoch {i} | Train Acc: {result_train:.4f}, Test Acc: {result_test:.4f} | Train Time: {stop_training - start_training:.2f}, Test Time: {stop_testing - start_testing:.2f}"
        )

    model = tm.save()

    with LZMAFile(f"{args.save_path}/mnist_model.tm", "wb") as f:
        pickle.dump(model, f)
