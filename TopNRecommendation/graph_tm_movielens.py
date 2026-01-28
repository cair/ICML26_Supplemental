import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MultiLabelBinarizer
import logging
#import wandb
import pickle

from data_loader_movielens import load_train_test_datasets, augment_movielens_dataset
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiOutputGraphTsetlinMachine

# TODO: See if we can find the source of numba logging being set to DEBUG
logging.getLogger('numba').setLevel(logging.WARNING) # Suppresses a large numba debug dump.

NUMBER_OF_NODES = 3
K_VALUES = [1, 5, 10, 50]
BATCH_SIZE = 1 * 10**4
RUN_FOLDER = "implementation/runs"
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)

def build_user_interaction_dict(train_df: pd.DataFrame) -> dict:
    user_interaction_dict = {x : [] for x in train_df["x"].explode().unique()}

    for _, row in train_df.iterrows():
        user_id = row["user_id"]
        interacted_items = row["x"]
        for item in interacted_items:
            user_interaction_dict[item].append(user_id)

    return user_interaction_dict

def get_user_symbol(user_id):
    return "U_" + str(user_id)

def get_item_symbol(title):
    return "I_" + str(title)

def get_category_symbol(category):
    return "C_" + str(category)

def initialize_graphs(df, symbols, args, batch_range=None, graphs_train=None, user_interactions=None, num_edges_per_node=None):
    if not batch_range is None:
        df = df.iloc[batch_range[0]:batch_range[1]]

    if graphs_train is None:
        graphs = Graphs(
            df.shape[0],
            symbols=symbols,
            hypervector_size=args.hypervector_size,
            hypervector_bits=args.hypervector_bits,
            double_hashing = args.double_hashing,
        )
    else:
        graphs = Graphs(
            df.shape[0],
            init_with=graphs_train # If test set, initialize with the same parameters as train set
        )

    for graph_id in range(df.shape[0]):
        graphs.set_number_of_graph_nodes(graph_id, len(df.iloc[graph_id]["titles_x"]))

    graphs.prepare_node_configuration()

    for graph_id in range(df.shape[0]):
        graph_number_of_nodes = graphs.number_of_graph_nodes[graph_id]
        for node_id in range(graph_number_of_nodes):
            number_of_edges = graph_number_of_nodes - 1 if not num_edges_per_node else num_edges_per_node
            graphs.add_graph_node(graph_id, node_id, number_of_edges)

    graphs.prepare_edge_configuration()

    for graph_id in range(df.shape[0]):

        titles = df.iloc[graph_id]["titles_x"]
        genres = df.iloc[graph_id]["genres_x"]
        graph_number_of_nodes = graphs.number_of_graph_nodes[graph_id]
        for node_id in range(graph_number_of_nodes):
            # Add an edge to all other nodes except self unless num_edges_per_node is specified
            total_nodes = graph_number_of_nodes - 1
            limit = min(total_nodes, num_edges_per_node) if num_edges_per_node else total_nodes
            limit = limit + 1 if node_id < limit else limit # adjust limit to account for skipping self-loop
            for destination_node_id in range(limit):
                if destination_node_id == node_id:
                    continue
                edge_type = "interacted"
                graphs.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)
            
            graphs.add_graph_node_property(graph_id, node_id, get_item_symbol(titles[node_id]))
            for genre in genres[node_id]: # Each movie can have multiple genres
                graphs.add_graph_node_property(graph_id, node_id, get_category_symbol(genre))
            if user_interactions is not None:
                interacted_users = user_interactions.get(titles[node_id], [])
                for user_id in interacted_users:
                    graphs.add_graph_node_property(graph_id, node_id, get_user_symbol(user_id))
            
    graphs.encode()

    return graphs

RECOMMENDATIONS_PATH = "recs/movielens/tms/"
if not os.path.exists(RECOMMENDATIONS_PATH):
    os.makedirs(RECOMMENDATIONS_PATH)
PLACEHOLDER_TIMESTAMP = 999999
def save_recommendations_as_tsv(df_test, graphs_test, tm, y_mlb, top_k=10, recs_file_name="vanilla_tm_recs.tsv"):
    y_proba = tm.score(graphs_test)
        
    sorted_indexes = y_proba.argsort(axis=1)[:, ::-1]
    top_k_predictions = sorted_indexes[:, :top_k]

    recommendations_strings = [] # Structure recommendations as a string for compatibility with ELLIOT
    for i, (predicted_indices, user_id) in enumerate(zip(top_k_predictions, df_test["user_id"].values)):
        predicted_labels = [y_mlb.classes_[idx] for idx in predicted_indices]
        predicted_scores = y_proba[i][predicted_indices]
        user_string = ""
        for label, score in zip(predicted_labels, predicted_scores):
            if label in df_test.iloc[i]["x"]:
                continue # Since repeat interactions are impossible in Movielens, do not recommend items the user has already interacted with.
                # This is also the default behavior in ELLIOT.
            user_string += f"{user_id}\t{label}\t{score:.4f}\t{PLACEHOLDER_TIMESTAMP}\n"
        recommendations_strings.append(user_string.strip())

    recommendations_path = os.path.join(RECOMMENDATIONS_PATH, recs_file_name)
    with open(recommendations_path, "w") as f:
        f.write("\n".join(recommendations_strings))
    print(f"Recommendations saved to {recommendations_path}")

RELEVANT_COLUMNS = [
    'user_id', 'x', 'rating_x', 'timestamp_x', 'y', 'rating_y',
    'timestamp_y', 'gender', 'age', 'occupation', 'titles_x', 'genres_x',
    'titles_y', 'genres_y'
]
USER_COLS = ['user_id', 'gender', 'age', 'occupation']
X_BUNDLE_COLS = ['x', 'rating_x', 'timestamp_x', 'titles_x', 'genres_x']
Y_BUNDLE_COLS = ['y', 'rating_y', 'timestamp_y', 'titles_y', 'genres_y']

# Filter the dataframes by popular items in their labels
def filter_df_by_popular_items(df, popular_set):
    original_user_data = df[USER_COLS + X_BUNDLE_COLS].drop_duplicates(subset=['user_id'])
    
    exploded_df = df[USER_COLS + Y_BUNDLE_COLS].explode(Y_BUNDLE_COLS)
    
    filtered_exploded = exploded_df[exploded_df['y'].isin(popular_set)]
    
    agg_df = filtered_exploded.groupby(USER_COLS).agg(list).reset_index()
    
    final_df = pd.merge(original_user_data, agg_df, on=USER_COLS, how='left')
    final_df = final_df.dropna(subset=['x'])
    final_df = final_df.dropna(subset=['y'])
        
    return final_df

def binarize_labels(df, multilabelbinarizer=None):
    mlb = multilabelbinarizer
    if mlb is None:
        mlb = MultiLabelBinarizer()
        mlb.fit(df['y'])
    y_binarized = mlb.transform(df['y'])
    return y_binarized, mlb

def get_labels(df):
    return np.stack(df['y_mlb'].values)

# NOTE: These metrics are exclusively used for evaluation during training. The final reported metrics are calculated via ELLIOT
def calculate_ndcg(df, scores, y_mlb, k=10):
    ndcg_scores = []
    for i, (_, row) in enumerate(df.iterrows()): # Apparently iterrows does not necessarily 0-index the rows??
        true_labels = set(row['y'])
        predicted_label = []
        for idx in np.argsort(-scores[i]):
            label = y_mlb.classes_[idx]
            if label in row["x"]:
                continue
            predicted_label.append(label)
            if len(predicted_label) >= k:
                break
        #score_indices = np.argsort(-scores[i])[:k]
        #score_labels = [y_mlb.classes_[idx] for idx in score_indices]
        dcg = 0.0
        idcg = 0.0
        for rank, idx in enumerate(predicted_label):
            if idx in true_labels:
                dcg += 1.0 / np.log2(rank + 2) # + 2 since rank is 0-indexed
        for rank in range(min(len(true_labels), k)):
            idcg += 1.0 / np.log2(rank + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

# NOTE: These metrics are exclusively used for evaluation during training. The final reported metrics are calculated via ELLIOT
def calculate_hit_rate(df, scores, y_mlb, k=10):
    correct = 0
    total = df.shape[0]
    for i, (_, row) in enumerate(df.iterrows()):
        true_labels = set(row['y'])
        predicted_label = []
        for idx in np.argsort(-scores[i]):
            label = y_mlb.classes_[idx]
            if label in row["x"]:
                continue
            predicted_label.append(label)
            if len(predicted_label) >= k:
                break
        #predicted_indices = np.argsort(-scores[i])[:k]
        #predicted_label = [y_mlb.classes_[idx] for idx in predicted_indices]
        if any(label in true_labels for label in predicted_label):
            correct += 1
    return correct / total * 100.0

def main(args, use_wandb=False, wandb_name=None):  
    np.random.seed(42)
    start_time = time.time()
    train_df, test_df, _ = load_train_test_datasets()
    train_df, _, _ = augment_movielens_dataset(train_df, include_movie_features=True)
    test_df, _, _ = augment_movielens_dataset(test_df, include_movie_features=True)
    train_df, test_df = train_df[RELEVANT_COLUMNS], test_df[RELEVANT_COLUMNS]

    MAX_UNIQUE_ITEMS = None # Set to None to use all items
    if MAX_UNIQUE_ITEMS is not None:
        popular_items_set = np.unique(np.concatenate([train_df['y'].explode().values, test_df['y'].explode().values]))
        popular_items_set = set(popular_items_set[popular_items_set < MAX_UNIQUE_ITEMS])
        train_df = filter_df_by_popular_items(train_df, popular_items_set)
        test_df = filter_df_by_popular_items(test_df, popular_items_set)

    if use_wandb:
        log_args = vars(args)
        log_args["exp_id"] = args.exp_id
        # Removed for anonymity. Initialize wandb here:
        run = None
    else:
        run = None

    binzarized_y_train, mlb = binarize_labels(train_df)
    binzarized_y_test, _ = binarize_labels(test_df, multilabelbinarizer=mlb)
    train_df['y_mlb'] = list(binzarized_y_train)
    test_df['y_mlb'] = list(binzarized_y_test)
    
    users = np.concatenate([train_df['user_id'].unique(), test_df['user_id'].unique()])
    print("Users: ",len(users))

    items = np.unique(np.concatenate([train_df['titles_x'].explode().values, test_df['titles_x'].explode().values]))
    print("Items: ",len(items))

    categories = np.unique(np.concatenate([[genre for genres in train_df['genres_x'].explode().values for genre in genres], [genre for genres in test_df['genres_x'].explode().values for genre in genres]]))
    print("Genres: ",len(categories))

    # Initialize Graphs with symbols for GTM
    symbols = [get_item_symbol(i) for i in items] + [get_category_symbol(c) for c in categories]
    user_interactions = None # Optional parameter, includes a symbol for each user that has interacted with an item for each node.
    if args.include_user_interactions:
        symbols += [get_user_symbol(u) for u in users]
        user_interactions = build_user_interaction_dict(train_df)
    print("Symbols: ",len(symbols))
    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds")

    graphs_train = initialize_graphs(train_df, symbols, args, user_interactions=user_interactions, num_edges_per_node=args.num_edges_per_node)
    print(f"Initialized training graphs in {time.time() - start_time:.2f} seconds")
    graphs_test = initialize_graphs(test_df, symbols, args, graphs_train=graphs_train, user_interactions=user_interactions, num_edges_per_node=args.num_edges_per_node)
    print(f"Initialized testing graphs in {time.time() - start_time:.2f} seconds")

    start_time = time.time()

    tm = MultiOutputGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits = args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing = args.double_hashing,
        one_hot_encoding = args.one_hot_encoding,
        grid=(16*13*4,1,1), #h100 config (see gtm repo)
        block=(128,1,1) #h100 config
    )

    for epoch in tqdm(range(args.epochs)):
        epoch_dict = {}
        Y_train_ratings = get_labels(train_df)
        tm.fit(graphs_train, Y_train_ratings, epochs=1, incremental=True)

        train_predictions = tm.score(graphs_train)
        test_predictions = tm.score(graphs_test)

        for k in K_VALUES: # Evaluate multiple k values for wandb logging
            epoch_dict[f"Train_HR@{k}"] = calculate_hit_rate(train_df, train_predictions, mlb, k=k)
            epoch_dict[f"Train_NDCG@{k}"] = calculate_ndcg(train_df, train_predictions, mlb, k=k)
            epoch_dict[f"Test_HR@{k}"] = calculate_hit_rate(test_df, test_predictions, mlb, k=k)
            epoch_dict[f"Test_NDCG@{k}"] = calculate_ndcg(test_df, test_predictions, mlb, k=k)



        if use_wandb:
            run.log(epoch_dict)
        PRINT_K = 50
        # Removed timers to reduce dependencies.
        print(f"Epoch {epoch+1}/{args.epochs}\n\
        Test HR@{PRINT_K}: {epoch_dict[f'Test_HR@{PRINT_K}']:.2f}%, Train HR@{PRINT_K}: {epoch_dict[f'Train_HR@{PRINT_K}']:.2f}%, \n\
        Test NDCG@{PRINT_K}: {epoch_dict[f'Test_NDCG@{PRINT_K}']:.4f}, Train NDCG@{PRINT_K}: {epoch_dict[f'Train_NDCG@{PRINT_K}']:.4f}")

    # Save the model for interpretability analysis
    tm.save(os.path.join(RUN_FOLDER, f"graphtm_movielens_model_{args.exp_id}.gtm"))
    with open(os.path.join(RUN_FOLDER, f"graphtm_movielens_graphs_{args.exp_id}.pkl"), "wb") as f:
        pickle.dump({
            "graphs_train": graphs_train,
            "graphs_test": graphs_test,
            "args": args,
            "mlb": mlb
        }, f)

    save_recommendations_as_tsv(test_df, graphs_test, tm, mlb, top_k=1000, recs_file_name=f"graphtm_movielens_recs_{args.exp_id}.tsv")
    if use_wandb:
        run.finish()
    print(f"Trained and evaluated model in {benchmark_total.elapsed():.2f} seconds")
    

def default_args(**kwargs):
    parser = argparse.ArgumentParser() # Since the script is meant to be run in-debugger, set the default values to None to be explicit.
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--number-of-clauses", default=None, type=int)
    parser.add_argument("--T", default=None, type=int)
    parser.add_argument("--s", default=None, type=float)
    parser.add_argument("--number-of-state-bits", default=None, type=int)
    parser.add_argument("--depth", default=None, type=int)
    parser.add_argument("--hypervector-size", default=None, type=int)
    parser.add_argument("--hypervector-bits", default=None, type=int)
    parser.add_argument("--message-size", default=None, type=int)
    parser.add_argument("--message-bits", default=None, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=None, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=None, action='store_true')
    parser.add_argument('--include-user-interactions', dest='include_user_interactions', default=None, action='store_true')
    parser.add_argument("--max-included-literals", default=None, type=int)
    parser.add_argument("--exp_id", default=None, type=str)
    parser.add_argument("--num-edges-per-node", default=None, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

ARGS = {
    "epochs": 10,
    "number_of_clauses": 4096,
    "T": 500,
    "s": 5.0,
    "number_of_state_bits": 8,
    "depth": 1,
    "hypervector_size": 2096,
    "hypervector_bits": 64,
    "one_hot_encoding": False,
    "message_size": 256,
    "message_bits": 64,
    "double_hashing": False,
    "include_user_interactions": False,
    "max_included_literals": 256,
    "exp_id": "graphtm_movielens_v7",
    "num_edges_per_node": 2, # Limit the number of edges per node to reduce training time. Set to None to include all edges.
}

if __name__ == "__main__":
    main(default_args(**ARGS), use_wandb=False, wandb_name=None)