from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from keras.api.datasets import mnist


from matplotlib.lines import Line2D


def generate_graphs(X, Y, graph_args):
    graphs = Graphs(**graph_args)

    num_graphs = X.shape[0]
    dim = X.shape[1]
    num_nodes = dim * dim

    for id in range(num_graphs):
        graphs.set_number_of_graph_nodes(id, num_nodes)

    graphs.prepare_node_configuration()

    for graph_id in range(num_graphs):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            xpos = node_id // dim
            ypos = node_id % dim

            if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
                graphs.add_graph_node(graph_id, node_id, 3, str(node_id))

            elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
                graphs.add_graph_node(graph_id, node_id, 5, str(node_id))

            else:
                graphs.add_graph_node(graph_id, node_id, 8, str(node_id))

    graphs.prepare_edge_configuration()

    for graph_id in tqdm(range(num_graphs), desc="Adding edges"):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            d = {
                "up": node_id - dim,
                "down": node_id + dim,
                "left": node_id - 1,
                "right": node_id + 1,
                "up-left": node_id - (dim + 1),
                "up-right": node_id - (dim - 1),
                "down-left": node_id + (dim - 1),
                "down-right": node_id + (dim + 1),
            }

            n = 0
            breakpoint()
            for et, nei_id in d.items():
                if 0 <= nei_id < num_nodes:
                    n += 1
                    graphs.add_graph_node_edge(graph_id, node_id, nei_id, et)


            xpos = node_id // dim
            ypos = node_id % dim

            if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
                assert n == 3, "should have 3 neightbours"
            elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
                assert n == 5, "should have 5 neightbours"
            else:
                assert n == 8, "should have 8 neightbours"

            # neigh, et = [], []
            # if xpos + 1 < dim:
            #     neigh.append((xpos + 1) * dim + (ypos))
            #     et.append("bottom")
            # if xpos - 1 >= 0:
            #     neigh.append((xpos - 1) * dim + (ypos))
            #     et.append("top")
            # if ypos + 1 < dim:
            #     neigh.append(xpos * dim + (ypos + 1))
            #     et.append("right")
            # if ypos - 1 >= 0:
            #     neigh.append((xpos) * dim + (ypos - 1))
            #     et.append("left")
            # if xpos + 1 < dim and ypos + 1 < dim:
            #     neigh.append((xpos + 1) * dim + (ypos + 1))
            #     et.append("bottom-right")
            # if xpos + 1 < dim and ypos - 1 >= 0:
            #     neigh.append((xpos + 1) * dim + (ypos - 1))
            #     et.append("bottom-left")
            # if xpos - 1 >= 0 and ypos + 1 < dim:
            #     neigh.append((xpos - 1) * dim + (ypos + 1))
            #     et.append("top-right")
            # if xpos - 1 >= 0 and ypos - 1 >= 0:
            #     neigh.append((xpos - 1) * dim + (ypos - 1))
            #     et.append("top-left")
            #
            # if (xpos == 0 or xpos == dim - 1) and (ypos == 0 or ypos == dim - 1):
            #     assert len(neigh) == 3, "should have 3 neightbours"
            # elif xpos == 0 or xpos == dim - 1 or ypos == 0 or ypos == dim - 1:
            #     assert len(neigh) == 5, "should have 5 neightbours"
            # else:
            #     assert len(neigh) == 8, "should have 8 neightbours"

            # for nei, e in zip(neigh, et):
            #     graphs.add_graph_node_edge(graph_id, node_id, nei, e)

    for graph_id in tqdm(range(num_graphs), desc="Adding node symbol"):
        img = X[graph_id]

        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            xpos = node_id // dim
            ypos = node_id % dim

            if img[xpos, ypos] == 1:
                graphs.add_graph_node_property(graph_id, node_id, "1")
            else:
                graphs.add_graph_node_property(graph_id, node_id, "0")

    graphs.encode()

    return graphs


def plot_graph(gt: Graphs, graph_id: int):
    pastel = plt.get_cmap("Pastel1")
    colorslist = pastel(np.linspace(0, 1, 8))

    G = nx.MultiDiGraph()

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

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    legend_elements = []
    for k in range(len(gt.edge_type_id)):
        breakpoint()
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
            connectionstyle="arc3, rad=0.1",
            label=elabls,
        )
    print(legend_elements)

    plt.title("Graph " + str(graph_id))
    plt.legend(handles=legend_elements, loc="upper left")
    plt.show()


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)
    Y_train = Y_train.astype(np.uint32)
    Y_test = Y_test.astype(np.uint32)

    dim = 3
    X_train = X_train[:1000, :dim, :dim]
    Y_train = Y_train[:1000]
    X_test = X_test[:1000, :dim, :dim]
    Y_test = Y_test[:1000]

    graph_args = dict(
        number_of_graphs=X_train.shape[0],
        symbols=["0", "1"],
        hypervector_size=8,
        hypervector_bits=2,
        double_hashing=True,
    )

    graphs_train = generate_graphs(X_train, Y_train, graph_args)
    graphs_test = generate_graphs(
        X_test, Y_test, dict(number_of_graphs=X_test.shape[0], init_with=graphs_train)
    )

    plot_graph(graphs_train, 0)
