from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import numpy as np

from graphtm_exp.graph import Graphs
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm


class Benchmark:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        graphs: Graphs,
        save_dir: str,
        gtm_args: dict,
        X_test: np.ndarray | None = None,
        Y_test: np.ndarray | None = None,
        graphs_test: Graphs | None = None,
        epochs: int = 50,
    ):
        self.X = X
        self.Y = Y
        self.graphs = graphs
        self.save_dir = save_dir
        self.gtm_args = gtm_args
        self.epochs = epochs

        if X_test is not None:
            self.X_test = X_test
            self.Y_test = Y_test
            self.graphs_test = graphs_test
            self.X_train = X
            self.Y_train = Y
            self.graphs_train = graphs
        else:
            iota = np.arange(X.shape[0])
            train_idx, test_idx = train_test_split(iota, test_size=0.2, random_state=1)
            self.X_test = X[test_idx]
            self.Y_test = Y[test_idx]
            self.graphs_test = graphs[test_idx]
            self.X_train = X[train_idx]
            self.Y_train = Y[train_idx]
            self.graphs_train = graphs[train_idx]

    def metrics(self, ytrue, ypred):
        precision, recall, f1, _ = precision_recall_fscore_support(ytrue, ypred, average="weighted", zero_division=0)
        accuracy = accuracy_score(ytrue, ypred)
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    def fit_gtm(
        self,
        tm: MultiClassGraphTsetlinMachine,
        graphs_train: Graphs,
        y_train: np.ndarray,
        graphs_val: Graphs,
        y_val: np.ndarray,
    ):
        history = {}
        for epoch in (pbar := tqdm(range(self.epochs), desc="GTM Fit", leave=False, dynamic_ncols=True)):
            tm.fit(graphs_train, y_train, epochs=1, incremental=True)

            y_val_pred = tm.predict(graphs_val)
            metrics = self.metrics(y_val, y_val_pred)
            history[epoch] = metrics

            pbar.set_postfix_str(f"Acc: {metrics['accuracy']:.4f}")

        return history

    def save_results(self, history: dict[int, dict]):
        # TODO: Save results to a file in self.save_dir
        pass

    def create_splits(self):
        pass

    def run(self):
        # TODO: Run the benchmark
        # 1. Crete dataset splits
        # 2. For each split,
        #   Create a new model, and fit with the train/val split
        #   test dataset?
        #   Save the results
        pass
