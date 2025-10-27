from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from graphtm_exp.graph import Graphs as Graphs
from tqdm import tqdm


def is_valid_sequence(seq):
    valid_bases = {"A", "C", "G", "T"}
    return all(base in valid_bases for base in seq)


# Read data and filter out rows with illegal characters
df = pd.read_csv("./test/Sequence.csv")
filtered_data = df[df["sequence"].apply(is_valid_sequence)]

# Select 5 of the classes
labels_to_sample = [
    "SARS-CoV-2",
    "Influenza virus",
    "Dengue virus",
    "Zika virus",
    "Rotavirus",
]
df_filtered = filtered_data[filtered_data["label"].isin(labels_to_sample)]

# select balanced subset using smalles number of samples for one class
n_samples_per_class = df_filtered.groupby("label").count()["sequence"].min()
sampled_df = df_filtered.groupby("label").apply(lambda x: x.sample(n=n_samples_per_class, random_state=42))
sampled_df = sampled_df.reset_index(drop=True)

# Split intor train and test
train_df, test_df = train_test_split(sampled_df, stratify=sampled_df["label"], test_size=0.2, random_state=42)


label_mapping = {"SARS-CoV-2": 0, "Influenza virus": 1, "Dengue virus": 2, "Rotavirus": 3, "Zika virus": 4}


Y_train = np.empty(len(train_df), dtype=np.uint32)
Y_test = np.empty(len(test_df), dtype=np.uint32)

for graph_id in range(len(train_df)):
    label = train_df.iloc[graph_id]["label"]
    Y_train[graph_id] = label_mapping.get(label, -1)

for graph_id in range(len(test_df)):
    label = test_df.iloc[graph_id]["label"]
    Y_test[graph_id] = label_mapping.get(label, -1)


number_of_examples_train = len(train_df)
number_of_examples_test = len(test_df)
number_of_classes = 5
max_sequence_length = 500
hypervector_size = 512
hypervector_bits = 2
number_of_state_bits = 8

symbols = [
    "GGA",
    "ACA",
    "AAA",
    "TAT",
    "GCC",
    "AAC",
    "GTG",
    "CCC",
    "TCG",
    "TCT",
    "GAA",
    "GAC",
    "GGG",
    "TTG",
    "GAT",
    "TGA",
    "GGT",
    "TAG",
    "TGC",
    "GGC",
    "CGC",
    "CGA",
    "TTT",
    "CTT",
    "GAG",
    "CCA",
    "TAA",
    "AGC",
    "GCA",
    "CCT",
    "ATT",
    "TAC",
    "CGT",
    "CTG",
    "GTC",
    "AAG",
    "AGA",
    "TTA",
    "GCT",
    "CAA",
    "GTA",
    "CAT",
    "ACC",
    "AAT",
    "CTC",
    "GTT",
    "CAC",
    "ATC",
    "TCA",
    "TGG",
    "TCC",
    "AGT",
    "CAG",
    "CCG",
    "ACT",
    "ATG",
    "TGT",
    "TTC",
    "GCG",
    "AGG",
    "ACG",
    "CTA",
    "ATA",
    "CGG",
]


print("Creating training data")
graphs_train = Graphs(
    number_of_examples_train,
    symbols=symbols,
    hypervector_size=hypervector_size,
    hypervector_bits=hypervector_bits,
    double_hashing=False,
)

for graph_id in range(number_of_examples_train):
    sequence = train_df.iloc[graph_id]["sequence"]
    sequence_length = len(sequence)
    less_seq = min(sequence_length, max_sequence_length)
    num_nodes = max(0, less_seq - 2)
    graphs_train.set_number_of_graph_nodes(graph_id, num_nodes)
graphs_train.prepare_node_configuration()

for graph_id in range(number_of_examples_train):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_train.number_of_graph_nodes[graph_id] - 1 else 1
        graphs_train.add_graph_node(graph_id, node_id, number_of_edges)

graphs_train.prepare_edge_configuration()

Y_train = np.empty(number_of_examples_train, dtype=np.uint32)

for graph_id in range(number_of_examples_train):
    label = train_df.iloc[graph_id]["label"]

    Y_train[graph_id] = label_mapping.get(label, -1)

    sequence = train_df.iloc[graph_id]["sequence"]

    sequence_length = len(sequence)
    less_seq = min(sequence_length, max_sequence_length)
    num_nodes = max(0, less_seq - 2)

    for node_id in range(num_nodes):
        if node_id > 0:
            destination_node_id = node_id - 1
            edge_type = "Left"
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

        if node_id < num_nodes - 1:
            destination_node_id = node_id + 1
            edge_type = "Right"
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

    for node_id in range(num_nodes):
        symbol = sequence[node_id : node_id + 3]  # Extract 3-character symbol
        graphs_train.add_graph_node_property(graph_id, node_id, symbol)

graphs_train.encode()
print("TRAIN DATA CREATED")


print("Creating test data")
graphs_test = Graphs(number_of_examples_test, init_with=graphs_train)

for graph_id in range(number_of_examples_test):
    sequence = test_df.iloc[graph_id]["sequence"]
    sequence_length = len(sequence)
    less_seq = min(sequence_length, max_sequence_length)
    num_nodes = max(0, less_seq - 2)
    graphs_test.set_number_of_graph_nodes(graph_id, num_nodes)
graphs_test.prepare_node_configuration()

for graph_id in range(number_of_examples_test):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_test.number_of_graph_nodes[graph_id] - 1 else 1
        graphs_test.add_graph_node(graph_id, node_id, number_of_edges)

graphs_test.prepare_edge_configuration()

Y_test = np.empty(number_of_examples_test, dtype=np.uint32)

for graph_id in range(number_of_examples_test):
    label = test_df.iloc[graph_id]["label"]
    Y_test[graph_id] = label_mapping.get(label, -1)

    sequence = test_df.iloc[graph_id]["sequence"]
    sequence_length = len(sequence)
    less_seq = min(sequence_length, max_sequence_length)
    num_nodes = max(0, less_seq - 2)
    for node_id in range(num_nodes):
        if node_id > 0:
            destination_node_id = node_id - 1
            edge_type = "Left"
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

        if node_id < num_nodes - 1:
            destination_node_id = node_id + 1
            edge_type = "Right"
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

    for node_id in range(num_nodes):
        symbol = sequence[node_id : node_id + 3]
        graphs_test.add_graph_node_property(graph_id, node_id, symbol)

graphs_test.encode()
print("TEST DATA CREATED")

epochs = 50

print("=================Normal Graphs====================")
tm = MultiClassGraphTsetlinMachine(
    T=2000,
    s=1.0,
    depth=2,
    number_of_clauses=2000,
    max_included_literals=200,
    message_size=512,
    message_bits=2,
    one_hot_encoding=False,
    grid=(16 * 13, 1, 1),
    block=(128, 1, 1),
)

for i in range(epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

    print(
        "%d %.2f %.2f %.2f %.2f"
        % (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
    )


print("=================Split Graphs====================")
tm2 = MultiClassGraphTsetlinMachine(
    T=2000,
    s=1.0,
    depth=2,
    number_of_clauses=2000,
    max_included_literals=200,
    message_size=512,
    message_bits=2,
    one_hot_encoding=False,
    grid=(16 * 13, 1, 1),
    block=(128, 1, 1),
)

for i in range(epochs):
    fit_time = 0.0
    for b in tqdm(range(0, Y_train.shape[0], 100), desc="Training batches", dynamic_ncols=True, leave=False):
        gsub = graphs_train[b : b + 100]
        ysub = Y_train[b : b + 100]
        start_training = time()
        tm2.fit(gsub, ysub, epochs=1, incremental=True)
        stop_training = time()
        fit_time += stop_training - start_training

    start_testing = time()
    result_test = 100 * (tm2.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100 * (tm2.predict(graphs_train) == Y_train).mean()

    print(
        "%d %.2f %.2f %.2f %.2f"
        % (i, result_train, result_test, fit_time, stop_testing - start_testing)
    )
