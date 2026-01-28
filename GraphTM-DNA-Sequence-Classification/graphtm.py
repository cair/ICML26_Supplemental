########### Imports ###########
from graphtm_exp.benchmark import Benchmark
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.utils import resample
from GraphTsetlinMachine.graphs import Graphs

def is_valid_sequence(seq):
    if not isinstance(seq, str) or len(seq) < 3:
        return False
    valid_bases = {'A', 'C', 'G', 'T'}
    return all(base in valid_bases for base in seq)

if __name__ == "__main__":

    # -------------------------
    # Load dataset
    # -------------------------
    samples = pd.read_csv("/workspace/GraphTM-DNA-Sequence-Classification/Dataset/Sequence_Dataset.csv")
    max_length = 500
    balance_training_samples = 2000
    label_mapping = {
        "SARS-CoV-2": 0,
        "Influenza virus": 1,
        "Dengue virus": 2,
        "Zika virus": 3,
        "Rotavirus": 4,
        
    }
  

    # -------------------------
    # Remove Labels
    # -------------------------
    """viruses_to_remove = ["Rotavirus"]

    samples = samples[~samples['label'].isin(viruses_to_remove)]"""
    
    # -------------------------
    # Filter invalid sequences
    # -------------------------
    filtered_data = samples[samples['sequence'].apply(is_valid_sequence)]

    print(filtered_data['label'].unique())
    print(filtered_data['label'].value_counts())
    # -------------------------
    # Train / Test split
    # -------------------------
    train_df, test_df = train_test_split(
        filtered_data,
        test_size=0.2,
        random_state=42,
        stratify=filtered_data['label']
    )

    # -------------------------
    # Balance training data
    # -------------------------
    label_counts = train_df['label'].value_counts()
    resampled_data = []

    for label, count in label_counts.items():
        label_data = train_df[train_df['label'] == label]

        if count < balance_training_samples:
            label_data_resampled = resample(
                label_data,
                replace=True,
                n_samples=balance_training_samples,
                random_state=42
            )
        else:
            label_data_resampled = label_data.sample(
                n=balance_training_samples,
                random_state=42
            )

        resampled_data.append(label_data_resampled)

    train_df = (
        pd.concat(resampled_data)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    enc_dict = {'A': np.array([0,0,0,1]),
            'C': np.array([0,0,1,0]),
            'G': np.array([0,1,0,0]),
            'T': np.array([1,0,0,0])}
    
    

    def encode_sequence_oh(seq, max_length=max_length):
        enc_seq = np.zeros((max_length, 4), dtype=int)
        for i, letter in enumerate(seq[:max_length]):
            enc_seq[i, :] = enc_dict[letter]
        return enc_seq

    X_train = np.array([encode_sequence_oh(train_df.iloc[i]['sequence']) for i in range(train_df.shape[0])], dtype=np.uint32)
    X_test = np.array([encode_sequence_oh(test_df.iloc[i]['sequence']) for i in range(test_df.shape[0])], dtype=np.uint32)
    # =========================
    # Arguments
    # =========================

    class Args:
        number_of_examples_train = len(train_df)
        number_of_examples_test = len(test_df)
        number_of_classes = 5
        max_sequence_length = max_length
        hypervector_size = 512
        hypervector_bits = 2
        double_hashing = False
        message_size = 2048
        message_bits = 2
        number_of_state_bits = 8
        max_included_literals = 200

    args = Args()

    # =========================
    # Codon symbols (64)
    # =========================

    symbols = [
        'AAA','AAC','AAG','AAT','ACA','ACC','ACG','ACT',
        'AGA','AGC','AGG','AGT','ATA','ATC','ATG','ATT',
        'CAA','CAC','CAG','CAT','CCA','CCC','CCG','CCT',
        'CGA','CGC','CGG','CGT','CTA','CTC','CTG','CTT',
        'GAA','GAC','GAG','GAT','GCA','GCC','GCG','GCT',
        'GGA','GGC','GGG','GGT','GTA','GTC','GTG','GTT',
        'TAA','TAC','TAG','TAT','TCA','TCC','TCG','TCT',
        'TGA','TGC','TGG','TGT','TTA','TTC','TTG','TTT'
    ]

    # =========================
    # Create Training Graphs
    # =========================

    print("Creating training data")

    graphs_train = Graphs(
        args.number_of_examples_train,
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing
    )

    # Set number of nodes
    for graph_id in range(args.number_of_examples_train):
        seq = train_df.iloc[graph_id]['sequence']
        less_seq = min(len(seq), args.max_sequence_length)
        num_nodes = max(0, less_seq - 2)
        graphs_train.set_number_of_graph_nodes(graph_id, num_nodes)

    graphs_train.prepare_node_configuration()

    # Add node degrees
    for graph_id in range(args.number_of_examples_train):
        n_nodes = graphs_train.number_of_graph_nodes[graph_id]
        for node_id in range(n_nodes):
            edges = 2 if 0 < node_id < n_nodes - 1 else 1
            graphs_train.add_graph_node(graph_id, node_id, edges)

    graphs_train.prepare_edge_configuration()

    # Labels + edges + node symbols
    Y_train = np.empty(args.number_of_examples_train, dtype=np.uint32)

    for graph_id in range(args.number_of_examples_train):
        row = train_df.iloc[graph_id]
        seq = row['sequence']

        Y_train[graph_id] = label_mapping[row['label']]

        less_seq = min(len(seq), args.max_sequence_length)
        num_nodes = max(0, less_seq - 2)

        for node_id in range(num_nodes):
            if node_id > 0:
                graphs_train.add_graph_node_edge(
                    graph_id, node_id, node_id - 1, "Left"
                )
            if node_id < num_nodes - 1:
                graphs_train.add_graph_node_edge(
                    graph_id, node_id, node_id + 1, "Right"
                )

        for node_id in range(num_nodes):
            graphs_train.add_graph_node_property(
                graph_id, node_id, seq[node_id:node_id + 3]
            )

    graphs_train.encode()
    print("TRAIN DATA CREATED")

    # =========================
    # Create Test Graphs
    # =========================

    print("Creating test data")

    graphs_test = Graphs(
        args.number_of_examples_test,
        init_with=graphs_train
    )

    for graph_id in range(args.number_of_examples_test):
        seq = test_df.iloc[graph_id]['sequence']
        less_seq = min(len(seq), args.max_sequence_length)
        num_nodes = max(0, less_seq - 2)
        graphs_test.set_number_of_graph_nodes(graph_id, num_nodes)

    graphs_test.prepare_node_configuration()

    for graph_id in range(args.number_of_examples_test):
        n_nodes = graphs_test.number_of_graph_nodes[graph_id]
        for node_id in range(n_nodes):
            edges = 2 if 0 < node_id < n_nodes - 1 else 1
            graphs_test.add_graph_node(graph_id, node_id, edges)

    graphs_test.prepare_edge_configuration()

    Y_test = np.empty(args.number_of_examples_test, dtype=np.uint32)

    for graph_id in range(args.number_of_examples_test):
        row = test_df.iloc[graph_id]
        seq = row['sequence']

        Y_test[graph_id] = label_mapping[row['label']]

        less_seq = min(len(seq), args.max_sequence_length)
        num_nodes = max(0, less_seq - 2)

        for node_id in range(num_nodes):
            if node_id > 0:
                graphs_test.add_graph_node_edge(
                    graph_id, node_id, node_id - 1, "Left"
                )
            if node_id < num_nodes - 1:
                graphs_test.add_graph_node_edge(
                    graph_id, node_id, node_id + 1, "Right"
                )

        for node_id in range(num_nodes):
            graphs_test.add_graph_node_property(
                graph_id, node_id, seq[node_id:node_id + 3]
            )

    graphs_test.encode()
    print("TEST DATA CREATED")

        ################ Benchmark Parameters ################
    save_dir = "./GraphTM-DNA-Sequence-Classification/"  # Directory to save results
    name = "Sequence_classification"  # Name of the experiment
    gtm_args = {"number_of_clauses": 2000,
        "T": 2000,
        "s": 1.0,
        "depth": 2,
        "max_included_literals": args.max_included_literals
        
        }  # Graph TM parameters as dictionary, or None to skip GTM
    xgb_args = {None}  # XGBoost parameters as dictionary, or None to skip XGBoost
    vanilla_tm_args = {None}  # Vanilla TM parameters as dictionary, or None to skip Vanilla TM
    cotm_args = {None}  # CoTM parameters as dictionary, or None to skip CoTM


    # Create Benchmark
    bm = Benchmark(
        X_train,
        Y_train,
        graphs_train,
        save_dir,
        name=name,
        gtm_args=gtm_args,
        #xgb_args=xgb_args,
        #vanilla_tm_args=vanilla_tm_args,
        #cotm_args=cotm_args,
        X_test=X_test,
        Y_test=Y_test,
        graphs_test=graphs_test,
    )

    # Run Benchmark
    bm.run()