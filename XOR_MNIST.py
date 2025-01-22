import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.graphs import Graphs
from time import time
import argparse
import random
from keras.api.datasets import mnist
from tqdm import tqdm



def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--number-of-clauses", default=5000, type=int)
    parser.add_argument("--T", default=8000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=1024, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=2048, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument(
        "--double-hashing", dest="double_hashing", default=False, action="store_true"
    )
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--number-of-values", default=10, type=int)
    # parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()

values = [i for i in range(args.number_of_values)]
# 784 white pixel symbols
symbols = [f"{i}" for i in range(28 * 28)]
# for k in range(28 * 28):
#     # symbols.append("W%d,%d" % (k // 28, k % 28))
#     symbols.append(k)


def create_dataset(X_mnist, Y_mnist, number_of_values, noise, graph_params):
    graphs = Graphs(**graph_params)
    N = graphs.number_of_graphs
    for graph_id in range(N):
        graphs.set_number_of_graph_nodes(graph_id, 2)

    graphs.prepare_node_configuration()

    for graph_id in range(N):
        graphs.add_graph_node(graph_id, "N1", 1)
        graphs.add_graph_node(graph_id, "N2", 1)

    graphs.prepare_edge_configuration()

    for graph_id in range(N):
        graphs.add_graph_node_edge(graph_id, "N1", "N2", "Plain")
        graphs.add_graph_node_edge(graph_id, "N2", "N1", "Plain")

    X = np.empty((args.number_of_examples, 2), dtype=np.uint32)
    Y = np.empty(args.number_of_examples, dtype=np.uint32)

    ii = [np.argwhere(Y_mnist == i).ravel() for i in range(number_of_values)]

    for graph_id in tqdm(range(N), desc="Preparing", dynamic_ncols=True):
        n1 = random.randint(0, number_of_values - 1)
        n2 = random.randint(0, number_of_values - 1)

        i1 = np.random.choice(ii[n1])
        i2 = np.random.choice(ii[n2])

        for k in X_mnist[i1].nonzero()[0]:
            graphs.add_graph_node_property(graph_id, "N1", f"{k}")

        for k in X_mnist[i2].nonzero()[0]:
            graphs.add_graph_node_property(graph_id, "N2", f"{k}")

        X[graph_id, 0] = n1
        X[graph_id, 1] = n2

        if (n1 % 2) == (n2 % 2):
            Y[graph_id] = 0
        else:
            Y[graph_id] = 1

        if np.random.rand() <= noise:
            Y[graph_id] = 1 - Y[graph_id]

    graphs.encode()
    return graphs, X, Y


if __name__ == "__main__":
    (X_train_mnist, Y_train_mnist), (X_test_mnist, Y_test_mnist) = mnist.load_data()

    X_train_mnist = (
        np.where(X_train_mnist > 75, 1, 0)
        .reshape(X_train_mnist.shape[0], -1)
        .astype(np.uint32)
    )
    X_test_mnist = (
        np.where(X_test_mnist > 75, 1, 0)
        .reshape(X_test_mnist.shape[0], -1)
        .astype(np.uint32)
    )
    Y_train_mnist = Y_train_mnist.astype(np.uint32)
    Y_test_mnist = Y_test_mnist.astype(np.uint32)

    # Create train data
    print("Creating training data")
    graphs_train, X_train, Y_train = create_dataset(
        X_train_mnist,
        Y_train_mnist,
        args.number_of_values,
        args.noise,
        graph_params=dict(
            number_of_graphs=args.number_of_examples,
            symbols=symbols,
            hypervector_size=args.hypervector_size,
            hypervector_bits=args.hypervector_bits,
            double_hashing=args.double_hashing,
        ),
    )

    # Create test data
    print("Creating testing data")
    graphs_test, X_test, Y_test = create_dataset(
        X_test_mnist,
        Y_test_mnist,
        args.number_of_values,
        args.noise,
        graph_params=dict(
            number_of_graphs=args.number_of_examples,
            init_with=graphs_train,
        ),
    )

    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        double_hashing=args.double_hashing,
        # max_included_literals=args.max_included_literals,
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
