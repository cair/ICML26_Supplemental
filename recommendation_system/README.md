# Recommendation System
## Overview

This project evaluates and compares three different models for recommendation tasks:
- **Graph Neural Network (graph_nn.py)** 
- **Graph Tsetlin Machine (graph_tm.py)** 
- **Tsetlin Machine Classifier (tm_classifier.py)**

## Requirements

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Run All Experiments

To run all experiments with different noise ratios:
```bash
bash main.sh
```

The `main.sh` script will:
- Run all three models (Graph NN, Graph Tsetlin Machine, and Tsetlin Machine Classifier)
- Test across multiple noise levels: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
- Execute experiments for 10 iterations 
- Save results to `experiment_results.csv`

### Run Individual Models

You can also run individual models with specific noise ratios:

```bash
python3 graph_nn.py --dataset_noise_ratio 0.01 --exp_id my_experiment

python3 graph_tm.py --dataset_noise_ratio 0.01 --exp_id my_experiment

python3 tm_classifier.py --dataset_noise_ratio 0.01 --exp_id my_experiment
```

## Results

Experiment results are saved to `experiment_results.csv` with metrics for each model and noise ratio combination.