import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

random.seed(42)


def generate_graphs(symbols, noise, graph_args: dict):
    graphs = Graphs(**graph_args)
    number_of_examples = graph_args["number_of_graphs"]

    for graph_id in range(number_of_examples):
        graphs.set_number_of_graph_nodes(graph_id, 2)

    graphs.prepare_node_configuration()

    for graph_id in range(number_of_examples):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            number_of_edges = 1
            graphs.add_graph_node(graph_id, node_id, number_of_edges)

    graphs.prepare_edge_configuration()

    X = np.empty((number_of_examples, 2))
    Y = np.empty(number_of_examples, dtype=np.uint32)

    for graph_id in range(number_of_examples):
        edge_type = "Plain"
        source_node_id = 0
        destination_node_id = 1
        graphs.add_graph_node_edge(
            graph_id, source_node_id, destination_node_id, edge_type
        )

        source_node_id = 1
        destination_node_id = 0
        graphs.add_graph_node_edge(
            graph_id, source_node_id, destination_node_id, edge_type
        )

        x1 = random.choice(symbols)
        x2 = random.choice(symbols)
        X[graph_id] = np.array([x1, x2])
        if (x1 % 2) == (x2 % 2):
            Y[graph_id] = 0
        else:
            Y[graph_id] = 1

        graphs.add_graph_node_property(graph_id, 0, x1)
        graphs.add_graph_node_property(graph_id, 1, x2)

        if np.random.rand() <= noise:
            Y[graph_id] = 1 - Y[graph_id]

    graphs.encode()

    return graphs, X, Y


def train(tm, graphs_train, Y_train, graphs_test, Y_test, epochs):
    history = []
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1, end=" ")

        start_training = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        preds = tm.predict(graphs_test)
        preds_train = tm.predict(graphs_train)

        acc = np.mean(preds == Y_test)
        acc_train = np.mean(preds_train == Y_train)

        print(
            f"Train time: {stop_training - start_training:.2f}s, Train acc: {acc_train:.4f}, Test acc: {acc:.4f}"
        )
        history.append(acc)

    return history


if __name__ == "__main__":
    epochs = 50
    noises = [0.01, 0.05, 0.1, 0.2]
    num_values = [50, 100, 200, 400, 800]

    tm_params = {
        "number_of_clauses": 1000,
        "T": 2000,
        "s": 1,
        "message_size": 2048,
        "message_bits": 2,
        "double_hashing": True,
        "depth": 2,
        "grid": (16 * 13, 1, 1),
        "block": (128, 1, 1),
    }

    bests = np.zeros((len(noises), len(num_values)))

    for i, noise in enumerate(noises):
        for j, num_value in enumerate(reversed(num_values)):
            print(
                f"----------Running for {num_value} values and noise {noise}----------"
            )

            print("Generating data...")
            symbols = [i for i in range(num_value)]
            graph_params = {
                "number_of_graphs": 50000,
                "hypervector_size": 2048,
                "hypervector_bits": 2,
                "double_hashing": True,
                "symbols": symbols,
            }
            graphs_train, X_train, Y_train = generate_graphs(
                symbols, noise, graph_params
            )

            graphs_test, X_test, Y_test = generate_graphs(
                symbols,
                0.0,
                {
                    "number_of_graphs": 2000,
                    "init_with": graphs_train,
                },
            )

            trial_history = []
            for trial in range(5):
                print(f"Running trial {trial + 1}/{5}")
                tm = MultiClassGraphTsetlinMachine(**tm_params)
                accs = train(tm, graphs_train, Y_train, graphs_test, Y_test, epochs)
                trial_history.append(accs)

            df = pd.DataFrame(trial_history).T
            df.to_csv(
                f"./MV-NoisyXOR/results/noisy_xor_{num_value}_{noise}.csv", index=False
            )
            print(
                f"Results saved to ./MV-NoisyXOR/results/noisy_xor_{num_value}_{noise}.csv"
            )

            best = df.max(axis=1).min(axis=0)
            bests[i, j] = best
            print(f"Best accuracy: {best}")

            print("---------------------------------------------------")

    fig, ax = plt.subplots(layout="compressed")
    sns.heatmap(bests, annot=True, cbar=True, ax=ax)
    fig.savefig("./MV-NoisyXOR/results/heatmap.png")
