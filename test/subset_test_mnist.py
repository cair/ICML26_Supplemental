import pickle
from lzma import LZMAFile

from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from tqdm import tqdm

from graphtm_exp.graph import Graphs
from graphtm_exp.timer import Timer


if __name__ == "__main__":
    with LZMAFile("./Mayur/MNIST/graphs_train.tar.xz", "rb") as f:
        dtrain = pickle.load(f)
    with LZMAFile("./Mayur/MNIST/graphs_test.tar.xz", "rb") as f:
        dtest = pickle.load(f)

    graphs_train: Graphs = dtrain["graph"]
    y_train = dtrain["y"]

    graphs_test: Graphs = dtest["graph"]
    y_test = dtest["y"]

    graphs_train.__class__ = Graphs
    graphs_test.__class__ = Graphs

    print("====================Training with graph splits====================")
    tm = MultiClassGraphTsetlinMachine(2500, 3125, 10, depth=1)
    for i in range(10):
        fit_time = 0.0
        for b in tqdm(range(0, y_train.shape[0], 10000), desc="Training batches", dynamic_ncols=True, leave=False):
            gsub = graphs_train[b : b + 10000]
            ysub = y_train[b : b + 10000]
            with (fit_timer := Timer()):
                tm.fit(gsub, ysub, epochs=1, incremental=True)
            fit_time += fit_timer.elapsed()

        with (test_timer := Timer()):
            pred_test = tm.predict(graphs_test)

        result_test = 100 * (pred_test == y_test).mean()
        result_train = 100 * (tm.predict(graphs_train) == y_train).mean()

        print(
            f"Epoch {i} | Train Acc: {result_train:.4f}, Test Acc: {result_test:.4f} | Train Time: {fit_time:.2f}, Test Time: {test_timer.elapsed():.2f}"
        )

    print("====================Training with original graphs====================")
    tm2 = MultiClassGraphTsetlinMachine(2500, 3125, 10, depth=1)
    for i in range(10):
        with (fit_timer := Timer()):
            tm2.fit(graphs_train, y_train, epochs=1, incremental=True)

        with (test_timer := Timer()):
            pred_test = tm2.predict(graphs_test)

        result_test = 100 * (pred_test == y_test).mean()
        result_train = 100 * (tm2.predict(graphs_train) == y_train).mean()

        print(
            f"[Original Graphs] Epoch {i} | Train Acc: {result_train:.4f}, Test Acc: {result_test:.4f} | Train Time: {fit_timer.elapsed():.2f}, Test Time: {test_timer.elapsed():.2f}"
        )
