########### Imports ###########
from graphtm_exp.benchmark import Benchmark


if __name__ == "__main__":
    ################ Load Data ################
    X_train = ...
    Y_train = ...

    ################ Generate/Load Graphs ################
    graphs_train = ...

    ############### Optinally Load Test Data ################
    X_test = ...
    y_test = ...
    graphs_test = ...

    ################ Benchmark Parameters ################
    save_dir = ...  # Directory to save results
    name = ...  # Name of the experiment
    gtm_args = {...}  # Graph TM parameters


    # Create Benchmark
    bm = Benchmark(
        X_train,
        Y_train,
        graphs_train,
        save_dir,
        name=name,
        gtm_args=gtm_args,
        X_test=X_test,
        Y_test=y_test,
        graphs_test=graphs_test,
    )

    # Run Benchmark
    bm.run()
