########### Imports ###########
from graphtm_exp.benchmark import Benchmark


if __name__ == "__main__":
    ################ Load Data ################
    # Binary dataset, in numpy array of shape (num_examples, num_features), and dtype uint32
    X_train = ...
    # Corresponding labels, in numpy array of shape (num_examples), and dtype uint32
    Y_train = ...

    ################ Generate/Load Graphs ################
    # Graphs object
    graphs_train = ...

    ############### Optinally Load Test Data ################
    X_test = ...
    y_test = ...
    graphs_test = ...

    ################ Benchmark Parameters ################
    save_dir = ...  # Directory to save results
    name = ...  # Name of the experiment
    gtm_args = {...}  # Graph TM parameters as dictionary, or None to skip GTM
    xgb_args = {...}  # XGBoost parameters as dictionary, or None to skip XGBoost
    vanilla_tm_args = {...}  # Vanilla TM parameters as dictionary, or None to skip Vanilla TM
    cotm_args = {...}  # CoTM parameters as dictionary, or None to skip CoTM


    # Create Benchmark
    bm = Benchmark(
        X_train,
        Y_train,
        graphs_train,
        save_dir,
        name=name,
        gtm_args=gtm_args,
        xgb_args=xgb_args,
        vanilla_tm_args=vanilla_tm_args,
        cotm_args=cotm_args,
        X_test=X_test,
        Y_test=y_test,
        graphs_test=graphs_test,
    )

    # Run Benchmark
    bm.run()
