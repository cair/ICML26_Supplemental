import argparse
from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def get_states(tm):
    """
    Get the state values for each literal and each clause in the Graph Tsetlin Machine

    Input: A trained Graph Tsetlin Machine object
    
    Returns a [number_of_clauses x number_of_literals] numpy array with state values
    """
    tm.ta_action(depth, 0, 0)
    def get_state_val(tm, clause, ta):
        ta_state = tm.ta_state.reshape((tm.number_of_clauses, tm.number_of_ta_chunks, tm.number_of_state_bits))
        v = 0
        for i in range(tm.number_of_state_bits):
            v += 2**i * int(ta_state[clause, ta // 32, i] & (1 << (ta % 32)) > 0)
        return v

    state_vals = np.zeros([tm.number_of_clauses, tm.number_of_literals], dtype=int)
    for clause in range(tm.number_of_clauses):
        for ta in range(tm.number_of_literals):
            state_vals[clause, ta] = get_state_val(tm, clause, ta)
    return state_vals



def get_msg_states(tm, depth=1):
    tm.ta_action(depth, 0, 0) 
    def get_state_val(tm, clause, ta):
        #ta_state = tm.ta_state.reshape((tm.number_of_clauses, tm.number_of_ta_chunks, tm.number_of_state_bits))
        message_ta_state = tm.message_ta_state.reshape((tm.number_of_clauses, tm.number_of_message_chunks, tm.number_of_state_bits))
        v = 0
        for i in range(tm.number_of_state_bits):
            v += 2**i * int(message_ta_state[clause, ta // 32, i] & (1 << (ta % 32)) > 0)
        return v

    #print(get_state_val(tm, 0, 0))
    state_vals = np.zeros([tm.number_of_clauses, tm.number_of_message_literals], dtype=int)
    for clause in range(tm.number_of_clauses):
        for ta in range(tm.number_of_message_literals):
            state_vals[clause, ta] = get_state_val(tm, clause, ta)
    return state_vals


def create_data(number_of_samples, noise=0.0, init_with=None, seed=None):
    graphs = Graphs(
        number_of_samples,
        symbols=SYMBOLS,
        hypervector_size=HV_SIZE,
        hypervector_bits=HV_BITS,
        double_hashing=False,
        init_with=init_with,
    )

        # Prepare nodes
    for graph_id in range(number_of_samples):
        graphs.set_number_of_graph_nodes(
            graph_id, np.random.randint(N_CLASSES, MAX_LENGTH+1))
    graphs.prepare_node_configuration()
    
    # Prepare nodes

    # Add nodes
    for graph_id in range(number_of_samples):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            number_of_edges = 2 if node_id > 0 and node_id < graphs.number_of_graph_nodes[graph_id]-1 else 1
            graphs.add_graph_node(graph_id, node_id, number_of_edges)
    graphs.prepare_edge_configuration()
    
    # Add edges
    Y = np.zeros(number_of_samples, dtype=np.uint32)
    for graph_id in range(number_of_samples):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            if node_id > 0:
                destination_node_id = node_id - 1
                edge_type = "Left"
                graphs.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)
    
            if node_id < graphs.number_of_graph_nodes[graph_id]-1:
                destination_node_id = node_id + 1
                edge_type = "Right"
                graphs.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)
    
        Y[graph_id] = np.random.randint(N_CLASSES)
        node_id = np.random.randint(Y[graph_id], graphs.number_of_graph_nodes[graph_id])
        node_sym = np.zeros(graphs.number_of_graph_nodes[graph_id])
        node_sym[node_id - Y[graph_id]: node_id+1] = 1
        #for node_pos in range(graphs.number_of_graph_nodes[graph_id]):
        #    if node_sym[node_pos] == 1:
        #        graphs.add_graph_node_property(graph_id, node_pos, 'A')
        #    else:
        #        graphs.add_graph_node_property(graph_id, node_pos, 'X')
                
        for node_pos in range(Y[graph_id] + 1):
            graphs.add_graph_node_property(graph_id, node_id - node_pos, 'A')

        if np.random.rand() <= noise:
            Y[graph_id] = np.random.choice(np.setdiff1d(np.arange(N_CLASSES), [Y[graph_id]]))

    graphs.encode()
    return graphs, Y


def create_graph_from_array(sequence_array, y, init_with=None, seed=None):
    graph = Graphs(
        1,
        symbols=SYMBOLS,
        hypervector_size=HV_SIZE,
        hypervector_bits=HV_BITS,
        double_hashing=False,
        init_with=init_with,
    )

    graph.set_number_of_graph_nodes(0, len(sequence_array))
    graph.prepare_node_configuration()

    # Add nodes
    for node_id in range(graph.number_of_graph_nodes[0]):
        number_of_edges = 2 if node_id > 0 and node_id < graph.number_of_graph_nodes[0]-1 else 1
        graph.add_graph_node(0, node_id, number_of_edges)
    graph.prepare_edge_configuration()

    # Add edges
    for node_id in range(graph.number_of_graph_nodes[0]):
        if node_id > 0:
            destination_node_id = node_id - 1
            edge_type = "Left"
            graph.add_graph_node_edge(0, node_id, destination_node_id, edge_type)

        if node_id < graph.number_of_graph_nodes[0]-1:
            destination_node_id = node_id + 1
            edge_type = "Right"
            graph.add_graph_node_edge(0, node_id, destination_node_id, edge_type)

    node_id = np.where(sequence_array == 1)[0][-1]
    for node_pos in range(y+1):
        graph.add_graph_node_property(0, node_id - node_pos, 'A')

    graph.encode()
    return graph


MAX_LENGTH = 100
N_CLASSES = 5
HV_SIZE = 32
HV_BITS = 8
SEED = 1

SYMBOLS = ['A']

rng = np.random.seed(SEED)


graphs_train, Y_train = create_data(10000, noise=0.01)
graphs_test, Y_test = create_data(1000, init_with=graphs_train)
graphs_sample, Y_sample = create_data(10, init_with=graphs_train)

y_sample_1 = np.zeros(20, dtype=int)
y_sample_2 = np.zeros(100, dtype=int)
y_sample_1[6:10] = 1
y_sample_2[6:10] = 1

g1 = create_graph_from_array(y_sample_1, 3, init_with=graphs_train)
g2 = create_graph_from_array(y_sample_2, 3, init_with=graphs_train)

#breakpoint()

# TM Parameter 

number_of_clauses = 30
T = 300
s = 2.5
number_of_state_bits = 8
depth = 3
message_size = 512
message_bits = 8
max_included_literals = None
double_hashing = False
epochs = 20
q=1.0


tm = MultiClassGraphTsetlinMachine(
    number_of_clauses,
    T=T,
    s=s,
    number_of_state_bits=number_of_state_bits,
    depth=depth,
    message_size=message_size,
    message_bits=message_bits,
    max_included_literals=max_included_literals,
    double_hashing=double_hashing,
    q=q,
    grid=(16*13,1,1),
    block=(128,1,1),
)

train_accuracy = []
test_accuracy = []
               
for i in range(epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()
    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    train_accuracy.append(result_train)
    test_accuracy.append(result_test)
    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print('avg. train accuracy', np.mean(train_accuracy[-10:]))
print('avg. test accuracy', np.mean(test_accuracy[-10:]))


np.set_printoptions(threshold=30000, linewidth=300)
the_graph = graphs_sample
print("-----", "Transform nodewise", "-----")
for sample_id in range(10):
    print(tm.transform_nodewise(the_graph)[0][sample_id,:,:])

print("/n Repeat/n")
np.set_printoptions(threshold=30000, linewidth=300)
the_graph = graphs_sample
print("-----", "Transform nodewise", "-----")
for sample_id in range(10):
    print(tm.transform_nodewise(the_graph)[0][sample_id,:,:])
    
#breakpoint()

print()
print("-----", "Transform", "-----")
for sample_id in range(10):
    print(tm.transform(the_graph)[0][sample_id])
