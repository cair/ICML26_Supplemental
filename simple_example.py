import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

# tmu
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


def clause_sum_to_probability(clause_sum, T):
	return 0.5 * (1 + clause_sum / T)


# Created functions
def create_samples(pattern, number_of_samples, prob):
	"""
	Create n samples with a particular pattern.
	The parameter 'prob' is the wanted proportion of true labels for that pattern.
	"""
	X = np.repeat([pattern], repeats=number_of_samples, axis=0)
	y = np.zeros(number_of_samples, dtype=np.uint32)
	y[: int(number_of_samples * prob)] = 1
	return X, y


def make_dataset(patterns, probabilities, number_of_samples_per_pattern, random_state=None):
	"""
	Creates a dataset from lists of patterns, probabilities
	and number of samples with each pattern.
	The 'random_state' is relevant for only shuffling the samples.
	"""
	rng = np.random.default_rng(random_state)
	n_samples_total = sum(number_of_samples_per_pattern)
	X = np.empty([n_samples_total, len(patterns[0])], dtype=np.uint32)
	y = np.empty(n_samples_total, dtype=np.uint32)
	start = 0
	for i, pattern in enumerate(patterns):
		X_sub, y_sub = create_samples(pattern, number_of_samples_per_pattern[i], probabilities[i])
		X[start : start + number_of_samples_per_pattern[i], :] = X_sub
		y[start : start + number_of_samples_per_pattern[i]] = y_sub
		start += number_of_samples_per_pattern[i]
	ind = np.arange(X.shape[0])
	rng.shuffle(ind)
	X, y = X[ind], y[ind]
	return X, y


# Create the graph
patterns = [
	[1, 1, 1],
	[1, 1, 0],
	[1, 0, 1],
	[1, 0, 0],
	[0, 1, 1],
	[0, 1, 0],
	[0, 0, 1],
	[0, 0, 0],
]
probabilities = [0.95, 0.90, 0.85, 0.60, 0.50, 0.25, 0.10, 0.05]
samples_per_pattern = [100 for _ in range(8)]

X_train, y_train = make_dataset(patterns, probabilities, samples_per_pattern, random_state=42)


graph_args = {
	"hypervector_size": 4,
	"hypervector_bits": 2,
	"double_hashing": False,
	"symbols": ["X"],
}

node_names = ["A", "B", "C"]


def create_graph(X, graph_args, init_with=None):
	number_of_examples = X.shape[0]
	graphs = Graphs(number_of_examples, init_with=init_with, **graph_args)

	# Prepare nodes
	for graph_id in range(number_of_examples):
		graphs.set_number_of_graph_nodes(graph_id, 3)

	# Add nodes
	graphs.prepare_node_configuration()
	for graph_id in range(number_of_examples):
		graphs.add_graph_node(graph_id, "A", 1)
		graphs.add_graph_node(graph_id, "B", 2)
		graphs.add_graph_node(graph_id, "C", 1)

	# Add edges
	graphs.prepare_edge_configuration()
	for graph_id in range(number_of_examples):
		# Add edges
		graphs.add_graph_node_edge(graph_id, "A", "B", "Right")
		# graphs.add_graph_node_edge(graph_id, 'A', 'C', '2xRight')
		graphs.add_graph_node_edge(graph_id, "B", "A", "Left")
		graphs.add_graph_node_edge(graph_id, "B", "C", "Right")
		graphs.add_graph_node_edge(graph_id, "C", "B", "Left")
		# graphs.add_graph_node_edge(graph_id, 'C', 'A', '2xLeft')

		# Add symbol
		for node_id in range(graphs.number_of_graph_nodes[graph_id]):
			if X[graph_id, node_id] == 1:
				graphs.add_graph_node_property(graph_id, node_names[node_id], "X")

	# Finalize graph
	graphs.encode()
	return graphs


graphs_train = create_graph(
	X_train,
	graph_args,
)
graphs_patterns = create_graph(np.array(patterns), graph_args, init_with=graphs_train)


def run(epochs, tm_args):
	tm = MultiClassGraphTsetlinMachine(**tm_args)
	cs = np.zeros([len(patterns), epochs])

	for i in range(epochs):
		start_training = time()
		tm.fit(graphs_train, y_train, epochs=1, incremental=True)
		stop_training = time()
		# add clause sums
		y_pred = tm.predict(graphs_patterns)
		clause_sums = tm.score(graphs_patterns)
		clause_sums[:, 0] = -clause_sums[:, 0]
		cs[:, i] = clause_sums[np.arange(clause_sums.shape[0]), y_pred]

		result_train = 100 * (tm.predict(graphs_train) == y_train).mean()
		print("%d %.2f %.2f" % (i, result_train, stop_training - start_training), end="\r")

	return cs, tm


cs, tm = run(
	20,
	{
		"number_of_clauses": 6,
		"T": 20,
		"s": 2,
		"depth": 2,
		"message_size": 8,
		"message_bits": 2,
	},
)

weights = tm.get_state()[1].reshape(2, -1)

print(f"{graphs_train.hypervectors=}")
print(f"{graphs_train.X.shape=}")
print("Feature literals")
for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[int(tm.ta_action(depth=0, clause=clause, ta=i)) for i in range(graphs_train.hypervector_size * 2)])
print()

print("Message literals")
for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[int(tm.ta_action(depth=1, clause=clause, ta=i)) for i in range(tm.message_size * 2)])


clause_literals = tm.get_clause_literals(graphs_train.hypervectors)
message_clauses = tm.get_messages(1, len(graphs_train.edge_type_id))

num_symbols = len(graphs_train.symbol_id)
print("Actual clauses:")

for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	for literal in range(num_symbols):
		if clause_literals[clause, literal] == 1:
			print(f"{literal}", end=" ")

		if clause_literals[clause, literal + num_symbols] == 1:
			print(f"~{literal}", end=" ")

	print("")

for edge_type in range(len(graphs_train.edge_type_id)):
	print(f"Actual Messages for {edge_type=}:")

	for msg in range(tm.number_of_clauses):
		print(f"Message {msg} ", end=": ")

		for clause in range(tm.number_of_clauses):
			if message_clauses[edge_type, msg, clause] == 1:
				print(f"C:{clause}(", end=" ")

				for literal in range(num_symbols):
					if clause_literals[clause, literal] == 1:
						print(f"{literal}", end=" ")
					if clause_literals[clause, literal + num_symbols] == 1:
						print(f"~{literal}", end=" ")

				print(")", end=" ")

			if message_clauses[edge_type, msg, tm.number_of_clauses + clause] == 1:
				print(f"~C:{clause}(", end=" ")

				for literal in range(num_symbols):
					if clause_literals[clause, literal] == 1:
						print(f"{literal}", end=" ")
					if clause_literals[clause, literal + num_symbols] == 1:
						print(f"~{literal}", end=" ")

				print(")", end = " ")

		print("")


