# WARN: INCOMPLETE
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import pandas as pd

from graphtm_exp.graph import Graphs
from .timer import Timer
from datetime import datetime


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

        self.fname = f"{save_dir}/bm_res_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"  # TODO: Create empty file

        self.splits = self.create_splits()

    def create_splits(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Create Splits. Return a dict [split_name: (train_idx, val_idx)]
        """
        # Maybe this can be done in __init__?

        self.kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
        kf_splits = self.kf.split(np.zeros(self.Y_train.shape[0]), self.Y_train)

        splits = {}
        for fold, (train_idx, val_idx) in enumerate(kf_splits):
            splits[f"val_{fold}"] = (train_idx, val_idx)

        return splits

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
    ) -> dict[int, dict]:
        history: dict[int, dict] = {}
        for epoch in (pbar := tqdm(range(self.epochs), leave=False, dynamic_ncols=True)):
            with (fit_timer := Timer()):
                tm.fit(graphs_train, y_train, epochs=1, incremental=True)

            with (pred_timer := Timer()):
                y_val_pred = tm.predict(graphs_val)

            y_train_pred = tm.predict(graphs_train)

            val_metrics = self.metrics(y_val, y_val_pred)
            train_metrics = self.metrics(y_train, y_train_pred)
            metrics = {
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "fit_time": fit_timer.elapsed(),
                "pred_time": pred_timer.elapsed(),
            }
            history[epoch] = metrics
            pbar.set_postfix_str(f"Acc: Train={metrics['train_accuracy']:.4f}, Val={metrics['val_accuracy']:.4f}")

        return history

    def save_results(self, history: dict[int, dict]):
        # TODO: Most likely wrong
        df = pd.DataFrame.from_dict(history, orient="index")
        df.to_csv(self.fname, mode="a")

    def run(self):
        # Go through each split
        for split_name, (train_idx, val_idx) in self.splits.items():
            print(f"=============Running split: {split_name}=============")
            graphs_train_split = self.graphs_train[train_idx]
            y_train_split = self.Y_train[train_idx]
            graphs_val_split = self.graphs_train[val_idx]
            y_val_split = self.Y_train[val_idx]

            # GTM
            gtm = MultiClassGraphTsetlinMachine(**self.gtm_args)
            hist = self.fit_gtm(gtm, graphs_train_split, y_train_split, graphs_val_split, y_val_split)

            # Save Results
            self.save_results(hist)

        # Finally test set
        for rep in range(5):
            print(f"=============Final evaluation on test set ---- {rep}=============")
            gtm = MultiClassGraphTsetlinMachine(**self.gtm_args)
            hist = self.fit_gtm(gtm, self.graphs_train, self.Y_train, self.graphs_test, self.Y_test)
            self.save_results(hist)

        print(f"We are done! Results saved to {self.fname}.")
