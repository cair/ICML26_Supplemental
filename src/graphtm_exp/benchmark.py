# WARN: INCOMPLETE
import os
from datetime import datetime

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D as CoTM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from tmu.models.classification.vanilla_classifier import TMClassifier as VanillaTM
from tqdm import tqdm
from xgboost import XGBClassifier

from graphtm_exp.graph import Graphs

from .monitor import Monitor


class Benchmark:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        graphs: Graphs,
        save_dir: str,
        name: str,
        gtm_args: dict | None = None,
        xgb_args: dict | None = None,
        vanilla_tm_args: dict | None = None,
        cotm_args: dict | None = None,
        X_test: np.ndarray | None = None,
        Y_test: np.ndarray | None = None,
        graphs_test: Graphs | None = None,
        epochs: int = 50,
        gpu_polling_rate: float = 0.1,
        num_test_reps: int = 5,
    ):
        """
        Benchmark

        Parameters
        ----------
        X : np.ndarray
            Binarized Dataset. Should be of shape (n_samples, n_features)
        Y : np.ndarray
            Labels. Should be of shape (n_samples,)
        graphs : Graphs
            Dataset in graph format.
        save_dir : str
            Directory to save results.
        name : str
            Name for the experiment.
        gtm_args : dict | None
            Arguments for GTM model. If None, GTM is not run.
        xgb_args : dict | None
            Arguments for XGB model. If None, XGB is not run.
        vanilla_tm_args : dict | None
            Arguments for Vanilla TM model. If None, Vanilla TM is not run.
        cotm_args : dict | None
            Arguments for CoTM model. If None, CoTM is not run.
        X_test : np.ndarray | None
            Optional test set. If None, a split from X is used.
        Y_test : np.ndarray | None
            Optional test labels. If None, a split from Y is used.
        graphs_test : Graphs | None
            Optional test graphs. If None, a split from graphs is used.
        epochs : int
            Number of epochs to train each model.
        gpu_polling_rate : float
            Polling rate for GPU memory monitoring.
        num_test_reps : int
            Number of repetitions for final test evaluation.
        """

        gid = os.getenv("CUDA_DEVICE")
        self.gpu_id = int(gid) if gid is not None else 0
        self.gpu_polling_rate = gpu_polling_rate
        self.num_test_reps = num_test_reps

        self.X = X
        self.Y = Y
        self.graphs = graphs
        self.save_dir = save_dir
        self.gtm_args = gtm_args
        self.xgb_args = xgb_args
        self.vanilla_tm_args = vanilla_tm_args
        self.cotm_args = cotm_args
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

        self.fname = f"{save_dir}/{name}_{datetime.now().strftime('%a_%d_%b_%Y_%I_%M_%S_%p')}.csv"

        # Store metric order
        dummy_met = self.metrics(np.array([0, 1]), np.array([0, 1]))
        self.met_order = list(dummy_met.keys())

        # Create file and write header in same order
        with open(self.fname, "w") as f:
            f.write(
                f"Model,Split,Epoch,fit_time,pred_time,peak_gpu_mem_fit,peak_gpu_mem_infer,metric_type,{','.join(self.met_order)}\n"
            )

        # Create splits
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
        graphs_train: Graphs,
        y_train: np.ndarray,
        graphs_val: Graphs,
        y_val: np.ndarray,
        split_name: str,
    ) -> dict[int, dict]:
        if self.gtm_args is None:
            return {}
        tm = MultiClassGraphTsetlinMachine(**self.gtm_args)
        history: dict[int, dict] = {}
        for epoch in (pbar := tqdm(range(self.epochs), leave=False, dynamic_ncols=True)):
            with (fit_timer := Monitor(self.gpu_id)):
                tm.fit(graphs_train, y_train, epochs=1, incremental=True)

            with (pred_timer := Monitor(self.gpu_id)):
                y_val_pred = tm.predict(graphs_val)

            y_train_pred = tm.predict(graphs_train)

            val_metrics = self.metrics(y_val, y_val_pred)
            train_metrics = self.metrics(y_train, y_train_pred)
            metrics = {
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "fit_time": fit_timer.elapsed(),
                "pred_time": pred_timer.elapsed(),
                "peak_gpu_mem_fit": fit_timer.peak_memory(),
                "peak_gpu_mem_infer": pred_timer.peak_memory(),
            }
            history[epoch] = metrics
            pbar.set_postfix_str(f"Acc: Train={metrics['train_accuracy']:.4f}, Val={metrics['val_accuracy']:.4f}")

            # Save metrics to file
            row = f"GTM,{split_name},{epoch},{metrics['fit_time']},{metrics['pred_time']},{metrics['peak_gpu_mem_fit']},{metrics['peak_gpu_mem_infer']}"
            train_row = f"{row},train,{','.join(str(metrics[f'train_{k}']) for k in self.met_order)}"
            val_row = f"{row},val,{','.join(str(metrics[f'val_{k}']) for k in self.met_order)}"
            self.write_row(train_row)
            self.write_row(val_row)

        del tm
        print("GTM Done.")
        # Is history needed, when we write to file directly?
        return history

    def fit_xgb(self, X_train, y_train, X_val, y_val, split_name: str) -> dict[int, dict]:
        if self.xgb_args is None:
            return {}
        model = XGBClassifier(**self.xgb_args, device=f"cuda:{self.gpu_id}")
        history: dict[int, dict] = {}

        with (fit_timer := Monitor(self.gpu_id)):
            model.fit(X_train, y_train)

        with (pred_timer := Monitor(self.gpu_id)):
            y_val_pred = model.predict(X_val)

        y_train_pred = model.predict(X_train)

        val_metrics = self.metrics(y_val, y_val_pred)
        train_metrics = self.metrics(y_train, y_train_pred)
        metrics = {
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "fit_time": fit_timer.elapsed(),
            "pred_time": pred_timer.elapsed(),
            "peak_gpu_mem_fit": fit_timer.peak_memory(),
            "peak_gpu_mem_infer": pred_timer.peak_memory(),
        }
        history[0] = metrics

        # Save metrics to file
        row = f"XGB,{split_name},0,{metrics['fit_time']},{metrics['pred_time']},{metrics['peak_gpu_mem_fit']},{metrics['peak_gpu_mem_infer']}"
        train_row = f"{row},train,{','.join(str(metrics[f'train_{k}']) for k in self.met_order)}"
        val_row = f"{row},val,{','.join(str(metrics[f'val_{k}']) for k in self.met_order)}"
        self.write_row(train_row)
        self.write_row(val_row)

        del model
        print("XGB Done.")
        return history

    def fit_vanilla_tm(self, X_train, y_train, X_val, y_val, split_name: str) -> dict[int, dict]:
        if self.vanilla_tm_args is None:
            return {}

        model = VanillaTM(**self.vanilla_tm_args)
        history: dict[int, dict] = {}
        for epoch in (pbar := tqdm(range(self.epochs), leave=False, dynamic_ncols=True)):
            with (fit_timer := Monitor(self.gpu_id)):
                model.fit(X_train, y_train)

            with (pred_timer := Monitor(self.gpu_id)):
                y_val_pred = model.predict(X_val)

            y_train_pred = model.predict(X_train)

            val_metrics = self.metrics(y_val, y_val_pred)
            train_metrics = self.metrics(y_train, y_train_pred)
            metrics = {
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "fit_time": fit_timer.elapsed(),
                "pred_time": pred_timer.elapsed(),
                "peak_gpu_mem_fit": fit_timer.peak_memory(),
                "peak_gpu_mem_infer": pred_timer.peak_memory(),
            }
            history[epoch] = metrics
            pbar.set_postfix_str(f"Acc: Train={metrics['train_accuracy']:.4f}, Val={metrics['val_accuracy']:.4f}")

            # Save metrics to file
            row = f"VanillaTM,{split_name},{epoch},{metrics['fit_time']},{metrics['pred_time']},{metrics['peak_gpu_mem_fit']},{metrics['peak_gpu_mem_infer']}"
            train_row = f"{row},train,{','.join(str(metrics[f'train_{k}']) for k in self.met_order)}"
            val_row = f"{row},val,{','.join(str(metrics[f'val_{k}']) for k in self.met_order)}"
            self.write_row(train_row)
            self.write_row(val_row)

        del model
        print("Vanilla TM Done.")
        return history

    def fit_cotm(self, X_train, y_train, X_val, y_val, split_name: str) -> dict[int, dict]:
        if self.cotm_args is None:
            return {}

        model = CoTM(**self.cotm_args)
        history: dict[int, dict] = {}
        for epoch in (pbar := tqdm(range(self.epochs), leave=False, dynamic_ncols=True)):
            with (fit_timer := Monitor(self.gpu_id)):
                model.fit(X_train, y_train, epochs=1, incremental=True)

            with (pred_timer := Monitor(self.gpu_id)):
                y_val_pred = model.predict(X_val)

            y_train_pred = model.predict(X_train)

            val_metrics = self.metrics(y_val, y_val_pred)
            train_metrics = self.metrics(y_train, y_train_pred)
            metrics = {
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "fit_time": fit_timer.elapsed(),
                "pred_time": pred_timer.elapsed(),
                "peak_gpu_mem_fit": fit_timer.peak_memory(),
                "peak_gpu_mem_infer": pred_timer.peak_memory(),
            }
            history[epoch] = metrics
            pbar.set_postfix_str(f"Acc: Train={metrics['train_accuracy']:.4f}, Val={metrics['val_accuracy']:.4f}")

            # Save metrics to file
            row = f"CoTM,{split_name},{epoch},{metrics['fit_time']},{metrics['pred_time']},{metrics['peak_gpu_mem_fit']},{metrics['peak_gpu_mem_infer']}"
            train_row = f"{row},train,{','.join(str(metrics[f'train_{k}']) for k in self.met_order)}"
            val_row = f"{row},val,{','.join(str(metrics[f'val_{k}']) for k in self.met_order)}"
            self.write_row(train_row)
            self.write_row(val_row)

        del model
        print("CoTM Done.")
        return history

    def write_row(self, row: str):
        with open(self.fname, "a") as f:
            f.write(row + "\n")

    def run(self):
        # Go through each split
        for split_name, (train_idx, val_idx) in self.splits.items():
            print(f"=============Running split: {split_name}=============")
            graphs_train_split = self.graphs_train[train_idx]
            y_train_split = self.Y_train[train_idx]
            graphs_val_split = self.graphs_train[val_idx]
            y_val_split = self.Y_train[val_idx]
            x_train_split = self.X_train[train_idx]
            x_val_split = self.X_train[val_idx]

            # XGB
            xgb_hist = self.fit_xgb(x_train_split, y_train_split, x_val_split, y_val_split, split_name)

            # GTM
            gtm_hist = self.fit_gtm(graphs_train_split, y_train_split, graphs_val_split, y_val_split, split_name)

            # Vanilla TM
            van_tm_hist = self.fit_vanilla_tm(x_train_split, y_train_split, x_val_split, y_val_split, split_name)

            # CoTM
            cotm_hist = self.fit_cotm(x_train_split, y_train_split, x_val_split, y_val_split, split_name)

        # Finally test set
        for rep in range(self.num_test_reps):
            print(f"=============Final evaluation on test set ---- {rep}=============")

            # XGB
            hist = self.fit_xgb(self.X_train, self.Y_train, self.X_test, self.Y_test, f"test_{rep}")

            # GTM
            hist = self.fit_gtm(self.graphs_train, self.Y_train, self.graphs_test, self.Y_test, f"test_{rep}")

            # Vanilla TM
            hist = self.fit_vanilla_tm(self.X_train, self.Y_train, self.X_test, self.Y_test, f"test_{rep}")

            # CoTM
            hist = self.fit_cotm(self.X_train, self.Y_train, self.X_test, self.Y_test, f"test_{rep}")

        print(f"We are done! Results saved to {self.fname}.")
