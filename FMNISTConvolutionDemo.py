import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from keras.api.datasets import fashion_mnist
from matplotlib import colors
from seaborn import color_palette
from tqdm import tqdm

(X_train_org, Y_train), (X_test_org, Y_test) = fashion_mnist.load_data()

label_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# X_train_org = X_train_org[:1000]
# Y_train = Y_train[:1000]
# X_test_org = X_test_org[:1000]
# Y_test = Y_test[:1000]

resolution = 8
patch_size = 3
dim = 28 - patch_size + 1

X_train = np.empty((*X_train_org.shape, resolution), dtype=np.uint8)
X_test = np.empty((*X_test_org.shape, resolution), dtype=np.uint8)

# Quantize pixel values
for z in range(resolution):
    threshold = (z + 1) * 255 / (resolution + 1)
    X_train[..., z] = (X_train_org >= threshold) & 1
    X_test[..., z] = (X_test_org >= threshold) & 1

Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)


def f(b):
    nz = np.argwhere(b).ravel()
    return nz[-1] if len(nz) > 0 else 0


def unbinarize(a):
    return np.apply_along_axis(f, -1, a)


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--number-of-clauses", default=40000, type=int)
    parser.add_argument("--T", default=15000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument(
        "--double-hashing", dest="double_hashing", default=False, action="store_true"
    )
    # parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()

number_of_nodes = dim * dim
symbols = []

# Column and row symbols
for i in range(dim):
    symbols.append("C:%d" % (i))
    symbols.append("R:%d" % (i))

# Patch pixel symbols
for i in range(patch_size * patch_size * resolution):
    symbols.append(i)

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
        graphs_train.add_graph_node(graph_id, node_id, 0)

graphs_train.prepare_edge_configuration()

for graph_id in tqdm(range(X_train.shape[0])):
    for node_id in range(number_of_nodes):
        x, y = node_id // dim, node_id % dim
        patch = X_train[graph_id, x : x + patch_size, y : y + patch_size].flatten()

        graphs_train.add_graph_node_property(graph_id, node_id, "R:%d" % (x))
        graphs_train.add_graph_node_property(graph_id, node_id, "C:%d" % (y))

        for p in patch.nonzero()[0]:
            graphs_train.add_graph_node_property(graph_id, node_id, p)

graphs_train.encode()

print("Training data produced")

graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        graphs_test.add_graph_node(graph_id, node_id, 0)

graphs_test.prepare_edge_configuration()

for graph_id in tqdm(range(X_test.shape[0])):
    for node_id in range(number_of_nodes):
        x, y = node_id // dim, node_id % dim
        patch = X_test[graph_id, x : x + patch_size, y : y + patch_size].flatten()

        graphs_test.add_graph_node_property(graph_id, node_id, "R:%d" % (x))
        graphs_test.add_graph_node_property(graph_id, node_id, "C:%d" % (y))

        for p in patch.nonzero()[0]:
            graphs_test.add_graph_node_property(graph_id, node_id, p)

graphs_test.encode()

print("Testing data produced")

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


def scale(X, x_min, x_max):
    nom = (X - X.min()) * (x_max - x_min)
    denom = X.max() - X.min()
    denom = denom + (denom == 0)
    return x_min + nom / denom


def scale_image(img):
    if len(img.shape) == 3:
        for ch in range(3):
            img[..., ch] = scale(img[..., ch], 0, 1)
    else:
        img = scale(img, 0, 1)

    return img


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
    position_symbols = 2 * dim
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
        pos_literals = unbinarize(
            pos_literals.reshape((patch_size, patch_size, resolution))
        )
        neg_literals = unbinarize(
            neg_literals.reshape((patch_size, patch_size, resolution))
        )

        eff_literals = pos_literals - neg_literals
        # eff_literals = eff_literals.reshape((patch_size, patch_size, resolution))

        for node_id in range(np.max(graphs_test.number_of_graph_nodes)):
            xpos, ypos = node_id // dim, node_id % dim

            if co[c, node_id] == 1:
                final_imgs[0, xpos : xpos + patch_size, ypos : ypos + patch_size] += (
                    pos_literals * w
                )
                final_imgs[1, xpos : xpos + patch_size, ypos : ypos + patch_size] += (
                    neg_literals * w
                )
                final_imgs[2, xpos : xpos + patch_size, ypos : ypos + patch_size] += (
                    eff_literals * w
                )

    # Matplotlib visualization shenanigans
    rocket = color_palette("rocket", as_cmap=True)
    fullcmap = colors.LinearSegmentedColormap.from_list(
        "fullcmap", rocket(np.linspace(0, 1, 100))
    )
    cmap = colors.LinearSegmentedColormap.from_list(
        "cmap", rocket(np.linspace(0.5, 1, 50))
    )

    fig, axs = plt.subplots(1, 4, figsize=(10, 5), layout="compressed", squeeze=False)
    axs[0, 0].imshow(unbinarize(X_test[e]))
    axs[0, 1].imshow(final_imgs[0], cmap=cmap)
    axs[0, 2].imshow(final_imgs[1], cmap=cmap)
    axs[0, 3].imshow(final_imgs[2], cmap=fullcmap)

    axs[0, 0].set_title("Input image")
    axs[0, 1].set_title("Positive Literals")
    axs[0, 2].set_title("Negative Literals")
    axs[0, 3].set_title("Pos-Neg Literals")

    for ax in axs.ravel():
        ax.axis("off")

    fig.savefig(f"figs/fmnist_test_{label_names[int(pred)]}.png")
