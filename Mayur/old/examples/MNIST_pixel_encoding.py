import argparse
from time import time

from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from matplotlib import colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from keras.api.datasets import mnist
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from seaborn.palettes import color_palette
from tqdm import tqdm

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

dim = 3
X_train = X_train[:1000, :dim, :dim]
Y_train = Y_train[:1000]
X_test = X_test[:1000, :dim, :dim]
Y_test = Y_test[:1000]

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--number-of-clauses", default=2000, type=int)
    parser.add_argument("--T", default=3125, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--hypervector-size", default=8, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=2048, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument(
        "--double-hashing", dest="double_hashing", default=False, action="store_true"
    )
    parser.add_argument("--max-included-literals", default=None, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()

# patch_size = 10
# dim = 28 - patch_size + 1

number_of_nodes = dim * dim

symbols = ["0", "1"]

# Column and row symbols
# for i in range(dim):
#     symbols.append("C:%d" % (i))
#     symbols.append("R:%d" % (i))
#
# # Patch pixel symbols
# for i in range(patch_size * patch_size):
#     symbols.append(i)

graphs_train = Graphs(
    X_train.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing=args.double_hashing,
)

for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

for graph_id in range(X_train.shape[0]):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        xpos = node_id // dim
        ypos = node_id % dim

        if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
            graphs_train.add_graph_node(graph_id, node_id, 3)

        elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
            graphs_train.add_graph_node(graph_id, node_id, 5)

        else:
            graphs_train.add_graph_node(graph_id, node_id, 8)


graphs_train.prepare_edge_configuration()

for graph_id in tqdm(range(X_train.shape[0]), desc="Adding edges"):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        xpos = node_id // dim
        ypos = node_id % dim

        neigh, et = [], []
        if xpos + 1 < dim:
            neigh.append((xpos + 1) * dim + (ypos))
            et.append("bottom")
        if xpos - 1 >= 0:
            neigh.append((xpos - 1) * dim + (ypos))
            et.append("top")
        if ypos + 1 < dim:
            neigh.append(xpos * dim + (ypos + 1))
            et.append("right")
        if ypos - 1 >= 0:
            neigh.append((xpos) * dim + (ypos - 1))
            et.append("left")
        if xpos + 1 < dim and ypos + 1 < dim:
            neigh.append((xpos + 1) * dim + (ypos + 1))
            et.append("bottom-right")
        if xpos + 1 < dim and ypos - 1 >= 0:
            neigh.append((xpos + 1) * dim + (ypos - 1))
            et.append("bottom-left")
        if xpos - 1 >= 0 and ypos + 1 < dim:
            neigh.append((xpos - 1) * dim + (ypos + 1))
            et.append("top-right")
        if xpos - 1 >= 0 and ypos - 1 >= 0:
            neigh.append((xpos - 1) * dim + (ypos - 1))
            et.append("top-left")

        if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
            assert len(neigh) == 3, "should have 3 neightbours"
        elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
            assert len(neigh) == 5, "should have 5 neightbours"
        else:
            assert len(neigh) == 8, "should have 8 neightbours"

        for nei, e in zip(neigh, et):
            graphs_train.add_graph_node_edge(graph_id, node_id, nei, e)


for graph_id in tqdm(range(X_train.shape[0]), desc="Adding node symbol"):
    # if graph_id % 1000 == 0:
    #     print(graph_id, X_train.shape[0])

    img = X_train[graph_id]

    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        xpos = node_id // dim
        ypos = node_id % dim

        if img[xpos, ypos] == 0:
            graphs_train.add_graph_node_property(graph_id, node_id, "0")
        else:
            graphs_train.add_graph_node_property(graph_id, node_id, "1")


graphs_train.encode()

print("Training data produced")




graphs_test = Graphs(
    X_test.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing=args.double_hashing,
)

for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        xpos = node_id // dim
        ypos = node_id % dim

        if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
            graphs_test.add_graph_node(graph_id, node_id, 3)

        elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
            graphs_test.add_graph_node(graph_id, node_id, 5)

        else:
            graphs_test.add_graph_node(graph_id, node_id, 8)


graphs_test.prepare_edge_configuration()

for graph_id in tqdm(range(X_test.shape[0]), desc="Adding edges"):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        xpos = node_id // dim
        ypos = node_id % dim

        neigh, et = [], []
        if xpos + 1 < dim:
            neigh.append((xpos + 1) * dim + (ypos))
            et.append("bottom")
        if xpos - 1 >= 0:
            neigh.append((xpos - 1) * dim + (ypos))
            et.append("top")
        if ypos + 1 < dim:
            neigh.append(xpos * dim + (ypos + 1))
            et.append("right")
        if ypos - 1 >= 0:
            neigh.append((xpos) * dim + (ypos - 1))
            et.append("left")
        if xpos + 1 < dim and ypos + 1 < dim:
            neigh.append((xpos + 1) * dim + (ypos + 1))
            et.append("bottom-right")
        if xpos + 1 < dim and ypos - 1 >= 0:
            neigh.append((xpos + 1) * dim + (ypos - 1))
            et.append("bottom-left")
        if xpos - 1 >= 0 and ypos + 1 < dim:
            neigh.append((xpos - 1) * dim + (ypos + 1))
            et.append("top-right")
        if xpos - 1 >= 0 and ypos - 1 >= 0:
            neigh.append((xpos - 1) * dim + (ypos - 1))
            et.append("top-left")

        if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
            assert len(neigh) == 3, "should have 3 neightbours"
        elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
            assert len(neigh) == 5, "should have 5 neightbours"
        else:
            assert len(neigh) == 8, "should have 8 neightbours"

        for nei, e in zip(neigh, et):
            graphs_test.add_graph_node_edge(graph_id, node_id, nei, e)


for graph_id in tqdm(range(X_test.shape[0]), desc="Adding node symbol"):
    # if graph_id % 1000 == 0:
    #     print(graph_id, X_test.shape[0])

    img = X_test[graph_id]

    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        xpos = node_id // dim
        ypos = node_id % dim

        if img[xpos, ypos] == 0:
            graphs_test.add_graph_node_property(graph_id, node_id, "0")
        else:
            graphs_test.add_graph_node_property(graph_id, node_id, "1")


graphs_test.encode()

print("Testing data produced")





##draw a simple 2D graph using networkx. draw_simple_graph(obj<Graphs>*, int<graph_id>*, str<savefinename>)
def draw_simple_graph(gt, graph_id, filename="plotgraph.png"):
    # colorslist = cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
    pastel = plt.get_cmap("Pastel1")
    colorslist = pastel(np.linspace(0, 1, 8))
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

    def random_cons_layout():
        return {k: [int(k) // dim, int(k) % dim] for k in G.nodes.keys()}


    pos = random_cons_layout()
    # nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    legend_elements = []
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


# draw_simple_graph(graphs_train, 0)

def printing_stuff(tm):
    weights = tm.get_state()[1].reshape(tm.number_of_outputs, tm.number_of_clauses)
    clause_literals = tm.get_clause_literals(graphs_train.hypervectors)
    num_symbols = len(graphs_train.symbol_id)
    clause_outputs, class_sums = tm.transform_nodewise(graphs_test)
    e = 0
    for e in [0, 1, 2]:
        pred = np.argmax(class_sums[e])
        print(f"{Y_test[e]=}")
        print(f"{class_sums[e]=}")
        print(f"{pred=}")

        # clause_literals -> (num_clauses, 2*num_symbols)
        position_symbols = 38
        total_symbols = len(graphs_test.symbol_id)

        # clause_outputs -> (num_samples, num_clauses, num_nodes)
        co = clause_outputs[e]

        final_imgs = np.zeros((3, 28, 28))
        for c in tqdm(range(tm.number_of_clauses)):
            w = weights[pred, c]
            if w < 0:
                continue
            pos_literals = clause_literals[c, position_symbols:total_symbols]
            neg_literals = clause_literals[
                c, total_symbols + position_symbols : 2 * total_symbols
            ]

            eff_literals = pos_literals - neg_literals

            pos_literals = pos_literals.reshape((10, 10))
            neg_literals = neg_literals.reshape((10, 10))
            eff_literals = eff_literals.reshape((10, 10))

            for node_id in range(np.max(graphs_test.number_of_graph_nodes)):
                xpos, ypos = node_id // 19, node_id % 19

                if co[c, node_id] == 1:
                    final_imgs[0, xpos : xpos + 10, ypos : ypos + 10] += pos_literals * w
                    final_imgs[1, xpos : xpos + 10, ypos : ypos + 10] += neg_literals * w
                    final_imgs[2, xpos : xpos + 10, ypos : ypos + 10] += eff_literals * w

        # Matplotlib visualization shenanigans
        rocket = color_palette("rocket", as_cmap=True)
        fullcmap = colors.LinearSegmentedColormap.from_list(
            "fullcmap", rocket(np.linspace(0, 1, 100))
        )
        cmap = colors.LinearSegmentedColormap.from_list(
            "cmap", rocket(np.linspace(0.5, 1, 50))
        )

        fig, axs = plt.subplots(1, 4, figsize=(10, 5), layout="compressed", squeeze=False)
        axs[0, 0].imshow(X_test[e])
        axs[0, 1].imshow(final_imgs[0], cmap=cmap)
        axs[0, 2].imshow(final_imgs[1], cmap=cmap)
        axs[0, 3].imshow(final_imgs[2], cmap=fullcmap)

        axs[0, 0].set_title("Input image")
        axs[0, 1].set_title("Positive Literals")
        axs[0, 2].set_title("Negative Literals")
        axs[0, 3].set_title("Pos-Neg Literals")

        for ax in axs.ravel():
            ax.axis("off")

        # fig.savefig(f"figs/mnist_test_{pred}.png")
    plt.show()

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits=args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    # max_included_literals=args.max_included_literals,
    double_hashing=args.double_hashing,
)

for i in range(args.epochs):
    start_training = time()
    # breakpoint()
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


# def scale(X, x_min, x_max):
#     nom = (X - X.min()) * (x_max - x_min)
#     denom = X.max() - X.min()
#     denom = denom + (denom == 0)
#     return x_min + nom / denom
#
#
# def scale_image(img):
#     if len(img.shape) == 3:
#         for ch in range(3):
#             img[..., ch] = scale(img[..., ch], 0, 1)
#     else:
#         img = scale(img, 0, 1)
#
#     return img
#
#
# weights = tm.get_state()[1].reshape(tm.number_of_outputs, tm.number_of_clauses)
# clause_literals = tm.get_clause_literals(graphs_train.hypervectors)
# num_symbols = len(graphs_train.symbol_id)
# clause_outputs, class_sums = tm.transform_nodewise(graphs_test)
# e = 0
# for e in [0, 1, 2]:
#     pred = np.argmax(class_sums[e])
#     print(f"{Y_test[e]=}")
#     print(f"{class_sums[e]=}")
#     print(f"{pred=}")
#
#     # clause_literals -> (num_clauses, 2*num_symbols)
#     position_symbols = 38
#     total_symbols = len(graphs_test.symbol_id)
#
#     # clause_outputs -> (num_samples, num_clauses, num_nodes)
#     co = clause_outputs[e]
#
#     final_imgs = np.zeros((3, 28, 28))
#     for c in tqdm(range(tm.number_of_clauses)):
#         w = weights[pred, c]
#         if w < 0:
#             continue
#         pos_literals = clause_literals[c, position_symbols:total_symbols]
#         neg_literals = clause_literals[
#             c, total_symbols + position_symbols : 2 * total_symbols
#         ]
#
#         eff_literals = pos_literals - neg_literals
#
#         pos_literals = pos_literals.reshape((10, 10))
#         neg_literals = neg_literals.reshape((10, 10))
#         eff_literals = eff_literals.reshape((10, 10))
#
#         for node_id in range(np.max(graphs_test.number_of_graph_nodes)):
#             xpos, ypos = node_id // 19, node_id % 19
#
#             if co[c, node_id] == 1:
#                 final_imgs[0, xpos : xpos + 10, ypos : ypos + 10] += pos_literals * w
#                 final_imgs[1, xpos : xpos + 10, ypos : ypos + 10] += neg_literals * w
#                 final_imgs[2, xpos : xpos + 10, ypos : ypos + 10] += eff_literals * w
#
#     # Matplotlib visualization shenanigans
#     rocket = color_palette("rocket", as_cmap=True)
#     fullcmap = colors.LinearSegmentedColormap.from_list(
#         "fullcmap", rocket(np.linspace(0, 1, 100))
#     )
#     cmap = colors.LinearSegmentedColormap.from_list(
#         "cmap", rocket(np.linspace(0.5, 1, 50))
#     )
#
#     fig, axs = plt.subplots(1, 4, figsize=(10, 5), layout="compressed", squeeze=False)
#     axs[0, 0].imshow(X_test[e])
#     axs[0, 1].imshow(final_imgs[0], cmap=cmap)
#     axs[0, 2].imshow(final_imgs[1], cmap=cmap)
#     axs[0, 3].imshow(final_imgs[2], cmap=fullcmap)
#
#     axs[0, 0].set_title("Input image")
#     axs[0, 1].set_title("Positive Literals")
#     axs[0, 2].set_title("Negative Literals")
#     axs[0, 3].set_title("Pos-Neg Literals")
#
#     for ax in axs.ravel():
#         ax.axis("off")
#
#     fig.savefig(f"figs/mnist_test_{pred}.png")
#     plt.show()
