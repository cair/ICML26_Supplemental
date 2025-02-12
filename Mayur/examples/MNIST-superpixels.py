import numpy as np
from torch_geometric.datasets import MNISTSuperpixels
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import networkx as nx
import optuna
import os
import pickle
from sklearn.metrics import confusion_matrix
import pywt
from tqdm import tqdm

from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from seaborn.palettes import color_palette


# Argument parser for model parameters
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--number-of-clauses", default=2500, type=int)
    parser.add_argument("--T", default=3125, type=int)
    parser.add_argument("--s", default=10, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=1024, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=2048, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument(
        "--double-hashing", dest="double_hashing", default=True, action="store_true"
    )
    parser.add_argument("--max-included-literals", default=32, type=int)
    parser.add_argument("--output-file", default="optuna_results_dec.csv", type=str)
    parser.add_argument("--resolution-wavelet", default=0, type=int)
    parser.add_argument("--resolution", default=8, type=int)
    parser.add_argument(
        "--gpu", default=1, type=int, help="GPU device to use (default: 0)"
    )
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()

train_dataset = MNISTSuperpixels(root="data/MNISTSuperpixels", train=True)
test_dataset = MNISTSuperpixels(root="data/MNISTSuperpixels", train=False)

# train_dataset = train_dataset[:100]  # Use a subset for testing
# test_dataset = test_dataset[:100]


train_dataset = train_dataset  # Use a subset for testing
test_dataset = test_dataset

# slight increase:
blurRowandCol = True
blurRowandCol2 = True
blurRowandCol3 = False

edges = False
DirectionalEdges = True

Y_train = np.array([data.y.item() for data in train_dataset], dtype=np.uint32)
Y_test = np.array([data.y.item() for data in test_dataset], dtype=np.uint32)

# Initialize symbols for graph construction
symbols = [
    f"W{i},{j}" for i in range(28) for j in range(28)
]  # First-layer node symbols
symbols.extend(
    f"B{i},{j}" for i in range(28) for j in range(28)
)  # Background pixel symbols
symbols.append("W")  # White symbol
symbols.append("B")  # Black symbol

# Grey intensity levels
symbols.extend(f"G{i}" for i in range(35))  # 35 grey intensity levels
# Add symbols for wavelet coefficients
symbols.extend(
    f"WaveletL{i}G{j}" for i in range(5) for j in range(args.resolution_wavelet)
)


# Edge count symbols
symbols.extend(f"E{i}" for i in range(100))  # Symbols for edge counts (0 to 99)

# Row and column indicators
symbols.extend(f"R{i}" for i in range(28))  # Row indicators
symbols.extend(f"C{i}" for i in range(28))  # Column indicators

# Second-layer node identifiers
symbols.extend(
    f"SecondLayerNode{i}" for i in range(100)
)  # Symbols for second-layer nodes

# Second-layer node features
symbols.extend(f"Degree{i}" for i in range(100))  # Degree of connectivity
symbols.extend(
    f"GroupRange{i}-{j}" for i in range(100) for j in range(i + 1, 101)
)  # Group range indicators
symbols.extend(f"TypeAConnections{i}" for i in range(100))  # Type A edge counts
symbols.extend(f"TypeBConnections{i}" for i in range(100))  # Type B edge counts
symbols.extend(f"AvgPixelValue{i}" for i in range(25))  # Average pixel values (0-25)
symbols.extend(f"Connectivity{i}" for i in range(100))  # Connectivity levels
symbols.extend(
    f"EdgeTypeDiversity{i}" for i in range(100)
)  # Edge type diversity levels
symbols.extend(f"AvgDistance{i}" for i in range(100))  # Average distance levels
symbols.extend(f"LayerPos{i}" for i in range(100))  # Global positional indicators

new_nodes = 10
new_nodes = 0


def create_graphs_with_second_layer(
    dataset, new_nodes, distance_groups, resolution_wavelet, init_with=None,
):
    graphs = Graphs(
        len(dataset),
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        # double_hashing=args.double_hashing,
        init_with=init_with,
    )

    for graph_id, data in enumerate(dataset):
        num_nodes = data.num_nodes
        total_nodes = num_nodes + (new_nodes if new_nodes > 0 else 0)
        graphs.set_number_of_graph_nodes(graph_id, total_nodes)

    graphs.prepare_node_configuration()

    for graph_id, data in tqdm(enumerate(dataset)):
        num_nodes = data.num_nodes
        total_nodes = num_nodes + (new_nodes if new_nodes > 0 else 0)
        edge_counts = np.zeros(total_nodes, dtype=np.uint32)
        edge_index = data.edge_index.numpy()

        if DirectionalEdges:
            # Directional edge calculation
            for edge_id in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, edge_id]), int(edge_index[1, edge_id])
                row, col = data.pos[src].numpy().astype(int)
                dst_row, dst_col = data.pos[dst].numpy().astype(int)
                if row < dst_row:
                    edge_counts[src] += 1
                elif row > dst_row:
                    edge_counts[src] += 1
                if col < dst_col:
                    edge_counts[src] += 1
                elif col > dst_col:
                    edge_counts[src] += 1
        if edges:
            for edge_id in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, edge_id]), int(edge_index[1, edge_id])
                edge_counts[src] += 1

        # Second-layer connections (only if new_nodes > 0)
        if new_nodes > 0:
            group_size = max(1, num_nodes // new_nodes)
            for new_node_id in range(new_nodes):
                start = new_node_id * group_size
                end = min(start + group_size, num_nodes)

                for first_layer_node_id in range(start, end):
                    edge_counts[first_layer_node_id] += 1
                    edge_counts[num_nodes + new_node_id] += 1

        # Add nodes with final edge counts
        for node_id in range(num_nodes):
            graphs.add_graph_node(graph_id, node_id, edge_counts[node_id])

        if new_nodes > 0:
            for new_node_id in range(new_nodes):
                graphs.add_graph_node(
                    graph_id,
                    num_nodes + new_node_id,
                    edge_counts[num_nodes + new_node_id],
                )

    graphs.prepare_edge_configuration()

    for graph_id, data in enumerate(dataset):
        edge_index = data.edge_index.numpy()
        # Directional Edges (-2% vs all edges are the same)
        if DirectionalEdges:
            # Split into groups of "distance"
            max_distance = 9
            group_size = max_distance // distance_groups

            for edge_id in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, edge_id]), int(edge_index[1, edge_id])
                row, col = data.pos[src].numpy().astype(int)
                dst_row, dst_col = data.pos[dst].numpy().astype(int)

                # Calculate distances
                row_distance = abs(row - dst_row)
                col_distance = abs(col - dst_col)

                # Determine distance group
                row_group = min(row_distance // group_size, distance_groups - 1)
                col_group = min(col_distance // group_size, distance_groups - 1)

                # Add directional edges with distance group encoded
                if row < dst_row:
                    graphs.add_graph_node_edge(graph_id, src, dst, f"UP_{row_group}")
                elif row > dst_row:
                    graphs.add_graph_node_edge(graph_id, src, dst, f"DOWN_{row_group}")

                if col < dst_col:
                    graphs.add_graph_node_edge(graph_id, src, dst, f"LEFT_{col_group}")
                elif col > dst_col:
                    graphs.add_graph_node_edge(graph_id, src, dst, f"RIGHT_{col_group}")

        if edges:
            # Step 3: Add first-layer edges
            for edge_id in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, edge_id]), int(edge_index[1, edge_id])
                graphs.add_graph_node_edge(graph_id, src, dst, "FirstLayer")

        # Step 4: Add second-layer edges (only if new_nodes > 0)
        if new_nodes > 0:
            group_size = max(1, num_nodes // new_nodes)
            for new_node_id in range(new_nodes):
                start = new_node_id * group_size
                end = min(start + group_size, num_nodes)

                for first_layer_node_id in range(start, end):
                    graphs.add_graph_node_edge(
                        graph_id, first_layer_node_id, num_nodes + new_node_id, f"TypeA"
                    )
                    graphs.add_graph_node_edge(
                        graph_id, num_nodes + new_node_id, first_layer_node_id, f"TypeB"
                    )

        # Step 5: Add node properties
        for node_id in range(num_nodes):
            feature_value = data.x[node_id].item()
            row, col = data.pos[node_id].numpy().astype(int)
            resolution_wavelet = 0  # Wavelet seems to kill performance
            if resolution_wavelet != 0:
                # Wavelet Transform
                wavelet = "haar"
                level = 0
                coeffs = pywt.wavedec(
                    data.x.numpy().flatten(), wavelet=wavelet, level=level
                )

                # Encode coefficients as thermometer features
                for i, coeff in enumerate(coeffs):
                    min_coeff, max_coeff = np.min(coeff), np.max(coeff)
                    bucket_size = (max_coeff - min_coeff) / resolution_wavelet

                    for node_id, value in enumerate(coeff):
                        bucket = (
                            int((value - min_coeff) / bucket_size)
                            if bucket_size > 0
                            else 0
                        )
                        bucket = min(
                            bucket, resolution_wavelet - 1
                        )  # Ensure within range
                        graphs.add_graph_node_property(
                            graph_id, node_id, f"WaveletL{i}G{bucket}"
                        )

            # Threshold color encoding
            for z in range(args.resolution):
                threshold = (z + 1) * 255 / (args.resolution + 1)
                if feature_value >= threshold:
                    graphs.add_graph_node_property(graph_id, node_id, f"G{z}")

            # Row and column blur
            graphs.add_graph_node_property(graph_id, node_id, (f"R{col}"))
            if blurRowandCol:
                if col < 27:
                    graphs.add_graph_node_property(graph_id, node_id, (f"R{col + 1}"))
                if col > 0:
                    graphs.add_graph_node_property(graph_id, node_id, (f"R{col - 1}"))
            if blurRowandCol2:
                if col < 26:
                    graphs.add_graph_node_property(graph_id, node_id, (f"R{col + 2}"))
                if col > 1:
                    graphs.add_graph_node_property(graph_id, node_id, (f"R{col - 2}"))

            graphs.add_graph_node_property(graph_id, node_id, (f"C{row}"))
            if blurRowandCol:
                if row < 27:
                    graphs.add_graph_node_property(graph_id, node_id, (f"C{row + 1}"))
                if row > 0:
                    graphs.add_graph_node_property(graph_id, node_id, (f"C{row - 1}"))
            if blurRowandCol2:
                if row < 26:
                    graphs.add_graph_node_property(graph_id, node_id, (f"C{row + 2}"))
                if row > 1:
                    graphs.add_graph_node_property(graph_id, node_id, (f"C{row - 2}"))

            # Color data at specific row and col
            if feature_value == 0:
                symbol_name = f"B{row},{col}"
                graphs.add_graph_node_property(graph_id, node_id, symbol_name)
                graphs.add_graph_node_property(graph_id, node_id, "B")
            else:
                symbol_name = f"W{row},{col}"
                graphs.add_graph_node_property(graph_id, node_id, symbol_name)

            # Number of edges
            if not (node_id >= num_nodes):  # Exclude second-layer nodes
                graphs.add_graph_node_property(
                    graph_id, node_id, (f"E{edge_counts[node_id]}")
                )

        if new_nodes > 0:  # Second layer of nodes
            for new_node_id in range(new_nodes):
                second_layer_node_id = num_nodes + new_node_id

                # Add basic identifier property
                graphs.add_graph_node_property(
                    graph_id, second_layer_node_id, f"SecondLayerNode{new_node_id}"
                )

                # Add the number of connected first-layer nodes (degree)
                graphs.add_graph_node_property(
                    graph_id,
                    second_layer_node_id,
                    f"Degree{edge_counts[second_layer_node_id]}",
                )

                # Add positional encoding if applicable (e.g., region-based grouping)
                group_size = max(1, num_nodes // new_nodes)
                start = new_node_id * group_size
                end = min(start + group_size, num_nodes)
                graphs.add_graph_node_property(
                    graph_id, second_layer_node_id, f"GroupRange{start}-{end}"
                )

                # Average pixel intensity of connected first-layer nodes
                avg_pixel_value = 0
                connected_nodes = end - start
                if connected_nodes > 0:
                    avg_pixel_value = (
                        sum(
                            data.x[first_layer_node_id].item()
                            for first_layer_node_id in range(start, end)
                        )
                        / connected_nodes
                    )
                graphs.add_graph_node_property(
                    graph_id,
                    second_layer_node_id,
                    f"AvgPixelValue{int(avg_pixel_value)}",
                )

                # Add connectivity metrics
                graphs.add_graph_node_property(
                    graph_id, second_layer_node_id, f"Connectivity{connected_nodes}"
                )

                # Spatial clustering properties (e.g., average distance between connected nodes)
                avg_distance = 0
                distances = []
                for i in range(start, end):
                    for j in range(i + 1, end):
                        src_row, src_col = data.pos[i].numpy().astype(int)
                        dst_row, dst_col = data.pos[j].numpy().astype(int)
                        distances.append(
                            abs(src_row - dst_row) + abs(src_col - dst_col)
                        )
                if distances:
                    avg_distance = sum(distances) / len(distances)
                graphs.add_graph_node_property(
                    graph_id, second_layer_node_id, f"AvgDistance{int(avg_distance)}"
                )

                # Add global positional indicators
                graphs.add_graph_node_property(
                    graph_id, second_layer_node_id, f"LayerPos{new_node_id}"
                )

    graphs.encode()
    return graphs


def objective(trial):
    # resolution_wavelet = trial.suggest_int("resolution-wavelet", 0, 10, step=1)
    # print(resolution_wavelet)
    distance_groups = 9  # best value from Optuna
    new_nodes = 14  # best value from Optuna
    graphs_train = create_graphs_with_second_layer(
        train_dataset, new_nodes, distance_groups, resolution_wavelet
    )
    graphs_test = create_graphs_with_second_layer(
        test_dataset, new_nodes, distance_groups, resolution_wavelet
    )
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        T=args.T,
        s=args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing=args.double_hashing,
    )
    best_accuracy = 0.0
    patience = 10  # Number of epochs to wait for improvement
    wait = 0  # Counter for early stopping
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

        # Get predictions
        predicted_test = tm.predict(graphs_test)
        result_test = 100 * (predicted_test == Y_test).mean()

        # Save predictions and true labels to a pickle file
        with open("predictions_dec.pkl", "wb") as f:
            pickle.dump({"predictions": predicted_test, "true_labels": Y_test}, f)

        print(
            "%d %.2f %.2f %.2f %.2f %.2f"
            % (
                epoch,
                result_train,
                result_test,
                stop_training - start_training,
                stop_testing - start_testing,
                result_test - result_train,
            )
        )
        # Check if test accuracy has improved
        if result_test > best_accuracy:
            best_accuracy = result_test
            wait = 0  # Reset the counter
        else:
            wait += 1
            print(f"No improvement for {wait}/{patience} epochs, best {best_accuracy}")

        # Early stopping condition
        if wait >= patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. Best accuracy: {best_accuracy:.2f}%"
            )
            break
    accuracy = result_test
    # Save trial result to file
    with open(args.output_file, "a") as f:
        f.write(f"{trial.number},{new_nodes},{accuracy}\n")

    return accuracy


def draw_simple_graph(gt, graph_id):
    # colorslist = cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
    pastel = plt.get_cmap("Pastel1")
    colorslist = pastel(np.linspace(0, 1, len(gt.edge_type_id)))
    # colorslist = colors.LinearSegmentedColormap.from_list(
    #     "fullcmap", pastel(np.linspace(0, 1, 8))
    # )
    G = nx.MultiDiGraph()
    # pos = nx.spring_layout(G)
    # arc_rad = 0.2

    for node_id in range(0, gt.number_of_graph_nodes[graph_id]):
        for node_edge_num in range(
            0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]
        ):
            edge_index = (
                gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
            )
            G.add_edge(
                node_id, str(gt.edge[edge_index][0]), weight=gt.edge[edge_index][1]
            )

    dim=28
    def random_cons_layout():
        return {k: [int(k) // dim, int(k) % dim] for k in G.nodes.keys()}
    pos = random_cons_layout()
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

    legend_elements = []
    # print(f'{len(gt.edge_type_id)=}')
    for k in range(len(gt.edge_type_id)):
        eset = [(u, v) for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
        elabls = [d for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
        le = "Unknown"
        for ln in gt.edge_type_id.keys():
            if gt.edge_type_id[ln] == k:
                le = ln
                break
        legend_elements.append(
            Line2D([0], [0], marker="o", color=colorslist[k], label=le, lw=0)
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=eset,
            width=3,
            edge_color=colorslist[k],
            connectionstyle=f"arc3, rad=0.1",
            # label=elabls,
        )
    # print(legend_elements)
    plt.title("Graph " + str(graph_id))
    plt.legend(handles=legend_elements, loc="upper left")
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()



# def plot_superpixels(graph: Graphs, graph_id, img):
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(img)
#
#
#     breakpoint()
#
#     plt.show()

if __name__ == "__main__":
    # Initialize results file
    # if not os.path.exists(args.output_file):
    #     with open(args.output_file, "w") as f:
    #         f.write("trial,s,accuracy\n")
    #
    # # Create an Optuna study
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50)
    #
    # # Print the best parameters
    # print("Best trial:", study.best_trial)
    # print("Best parameters:", study.best_trial.params)
    # print("Best accuracy:", study.best_value)

    resolution_wavelet = args.resolution_wavelet
    distance_groups = 9  # best value from Optuna
    new_nodes = 14  # best value from Optuna
    graphs_train: Graphs = create_graphs_with_second_layer(
        train_dataset, new_nodes, distance_groups, resolution_wavelet
    )
    # draw_simple_graph(graphs_train, 0)
    graphs_test: Graphs = create_graphs_with_second_layer(
        test_dataset, new_nodes, distance_groups, resolution_wavelet, init_with=graphs_train,
    )
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        T=args.T,
        s=args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        # max_included_literals=args.max_included_literals,
    )
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

        # Get predictions
        predicted_test = tm.predict(graphs_test)
        result_test = 100 * (predicted_test == Y_test).mean()


    # ind = 0
    # clause_outputs, cs = tm.transform_nodewise(graphs_test)
    # weights = tm.get_state()[1].reshape(tm.number_of_outputs, tm.number_of_clauses)
    #
    # print(f'{graphs_test.hypervector_size=}')
    # print(f'{graphs_test.hypervectors=}')
    # print(f'{graphs_test.hypervectors.max()=}')
    # print(f'{graphs_test.hypervectors.min()=}')


    # clause_literals = tm.get_clause_literals(graphs_test.hypervectors)
    # num_symbols = len(graphs_test.symbol_id)
    # pred = np.argmax(clause_outputs[ind])
    # breakpoint()
    #
    # for c in tqdm(range(tm.number_of_clauses)):
    #     w = weights[pred, c]
    #
    #     if w < 0:
    #         continue




