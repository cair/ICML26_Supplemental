import numpy as np
from GraphTsetlinMachine.graphs import Graphs as original_Graphs


class Graphs(original_Graphs):
    """Make Graphs subsettable."""

    def __len__(self):
        return self.number_of_graphs

    def _get_indices(self, key) -> list[int]:
        if isinstance(key, int):
            if key < 0:
                key += self.number_of_graphs
            if key < 0 or key >= self.number_of_graphs:
                raise IndexError("Graph index out of range")
            indices = [key]
        elif isinstance(key, slice):
            indices = list(range(*key.indices(self.number_of_graphs)))
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            indices = []
            for k in key:
                if k < 0:
                    k += self.number_of_graphs
                if k < 0 or k >= self.number_of_graphs:
                    raise IndexError("Graph index out of range")
                indices.append(k)
        else:
            raise TypeError("Invalid graph index type")

        return indices

    def _create_subset(self, indices: list[int]):
        subset = Graphs(
            number_of_graphs=len(indices),
            double_hashing=self.double_hashing,
            one_hot_encoding=self.one_hot_encoding,
            init_with=self,
        )

        # Copy number_of_graph_nodes and graph_node_id
        for new_id, old_id in enumerate(indices):
            subset.number_of_graph_nodes[new_id] = self.number_of_graph_nodes[old_id]
            subset.graph_node_id[new_id] = self.graph_node_id[old_id]

        subset.prepare_node_configuration()

        # Copy node-level data
        for new_id, old_id in enumerate(indices):
            old_start = self.node_index[old_id]
            old_end = old_start + self.number_of_graph_nodes[old_id]
            new_start = subset.node_index[new_id]
            new_end = new_start + subset.number_of_graph_nodes[new_id]

            subset.node_type[new_start:new_end] = self.node_type[old_start:old_end]
            subset.number_of_graph_node_edges[new_start:new_end] = self.number_of_graph_node_edges[old_start:old_end]
            subset.graph_node_edge_counter[new_start:new_end] = self.graph_node_edge_counter[old_start:old_end]
            subset.X[new_start:new_end] = self.X[old_start:old_end]

        # Prepare edge configuration for subset
        subset.prepare_edge_configuration()

        # Copy edge data
        for new_id, old_id in enumerate(indices):
            for node_id in range(self.number_of_graph_nodes[old_id]):
                old_node_idx = self.node_index[old_id] + node_id
                new_node_idx = subset.node_index[new_id] + node_id

                old_edge_start = self.edge_index[old_node_idx]
                old_edge_count = self.graph_node_edge_counter[old_node_idx]
                new_edge_start = subset.edge_index[new_node_idx]

                subset.edge[new_edge_start : new_edge_start + old_edge_count] = self.edge[
                    old_edge_start : old_edge_start + old_edge_count
                ]

        subset.encode()

        return subset

    def __getitem__(self, index):
        indices = self._get_indices(index)
        if len(indices) == 0:
            raise ValueError("No graphs selected")
        return self._create_subset(indices)
