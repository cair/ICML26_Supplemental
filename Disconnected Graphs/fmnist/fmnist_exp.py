import os
import numpy as np
from keras.datasets import fashion_mnist
from graphtm_exp.graph import Graphs
from tqdm import tqdm
import pickle
from datetime import datetime
from graphtm_exp.benchmark import Benchmark


resolution = 8

def graph_generator(X, graph_args, patch_size):
    graphs = Graphs(**graph_args)

    num_graphs = X.shape[0]
    dim = X.shape[1] - patch_size + 1
    num_nodes = dim * dim

    for id in range(num_graphs):
        graphs.set_number_of_graph_nodes(id, num_nodes)

    graphs.prepare_node_configuration()

    for graph_id in tqdm(range(X.shape[0]), desc="Adding graph nodes", leave=False):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            graphs.add_graph_node(graph_id, node_id, 0)

    graphs.prepare_edge_configuration()

    for graph_id in tqdm(range(X.shape[0]), desc="Adding node symbols", leave=False):
        for node_id in range(num_nodes):
            x, y = node_id // dim, node_id % dim
            patch = X[graph_id, x : x + patch_size, y : y + patch_size].flatten()

            graphs.add_graph_node_property(graph_id, node_id, "R:%d" % (x))
            graphs.add_graph_node_property(graph_id, node_id, "C:%d" % (y))

            for p in patch.nonzero()[0]:
                graphs.add_graph_node_property(graph_id, node_id, p)

    graphs.encode()

    return graphs


def load_or_create_graphs(ds_dir):
    filename = f"{ds_dir}/fmnist_data_processed.pkl"
    try:
        with open(f"{filename}", "rb") as f:
            dat = pickle.load(f)
        print("Loaded existing graphs from disk.")

    except FileNotFoundError:
        print("Graphs not found on disk. Generating new graphs...")
        (x_train_org, y_train), (x_test_org, y_test) = fashion_mnist.load_data()

        x_train = np.empty((*x_train_org.shape, resolution), dtype=np.uint8)
        x_test = np.empty((*x_test_org.shape, resolution), dtype=np.uint8)

        # Quantize pixel values
        for z in range(resolution):
            threshold = (z + 1) * 255 / (resolution + 1)
            x_train[..., z] = (x_train_org >= threshold) & 1
            x_test[..., z] = (x_test_org >= threshold) & 1

        y_train = y_train.astype(np.uint32)
        y_test = y_test.astype(np.uint32)

        symbols = []
        patch_size = 3

        # Column and row symbols
        for i in range(28 - patch_size + 1):
            symbols.append("C:%d" % (i))
            symbols.append("R:%d" % (i))

        # Patch pixel symbols
        for i in range(patch_size * patch_size * resolution):
            symbols.append(i)

        graph_args = dict(
            number_of_graphs=x_train.shape[0],
            symbols=symbols,
            hypervector_size=256,
            hypervector_bits=2,
            double_hashing=False,
        )

        graphs_train = graph_generator(x_train, graph_args, patch_size)
        graphs_test = graph_generator(
            x_test,
            dict(
                number_of_graphs=x_test.shape[0],
                init_with=graphs_train,
            ),
            patch_size,
        )
        print("Graphs generated.")
        dat = {
            "graphs_args": graph_args,
            "patch_size": patch_size,
            "symbols": symbols,
            "x_train": x_train,
            "y_train": y_train,
            "graphs_train": graphs_train,
            "x_test": x_test,
            "y_test": y_test,
            "graphs_test": graphs_test,
        }
        with open(f"{filename}", "wb") as f:
            pickle.dump(dat, f)

    xtrain, ytrain, graphstrain = dat["x_train"], dat["y_train"], dat["graphs_train"]
    xtest, ytest, graphstest = dat["x_test"], dat["y_test"], dat["graphs_test"]
    xtrain = xtrain.reshape(xtrain.shape[0], -1).astype(np.uint32)
    xtest = xtest.reshape(xtest.shape[0], -1).astype(np.uint32)
    return xtrain, ytrain, graphstrain, xtest, ytest, graphstest


if __name__ == "__main__":
    x_train, y_train, graphs_train, x_test, y_test, graphs_test = load_or_create_graphs("./Mayur/fmnist/data")

    save_dir = f"./Mayur/fmnist/results/{datetime.now().strftime('%a_%d_%b_%Y_%I_%M_%S_%p')}"
    os.makedirs(save_dir, exist_ok=True)
    name = "mnist"

    xgb_args = {}
    gtm_args = {
        "number_of_clauses": 40000,
        "T": 15000,
        "s": 10.0,
        "depth": 1,
    }
    cotm_args = {
        "number_of_clauses": 40000,
        "T": 15000,
        "s": 10.0,
        "dim": (28, 28, resolution),
        "patch_dim": (3, 3),
    }

    bm = Benchmark(
        X=x_train,
        Y=y_train,
        graphs=graphs_train,
        save_dir=save_dir,
        name=name,
        gtm_args=gtm_args,
        xgb_args=xgb_args,
        cotm_args=cotm_args,
        X_test=x_test,
        Y_test=y_test,
        graphs_test=graphs_test,
    )
    bm.run()

    print(f"Results saved in {save_dir}")

