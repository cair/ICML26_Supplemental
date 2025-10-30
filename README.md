# Experiments and Benchmarks for the GraphTsetlinMachine paper

## Quick Start

### Devcontainer
Devcontainer configuration is provided for VSCode.

- Remote SSH into cair-gpu17
- Clone THIS repo somewhere.
- Open repo as a folder (project) in VSCode.
- Run the `.devcontainer/build` script to create `devcontainer.json` file for the devcontainer, with proper user and stuff.
```bash
./.devcontainer/build
```

- When prompted to open in devcontainer, select "Reopen in Container".
    - If not prompted, open the command palette (Ctrl+Shift+P), and select "Dev Containers: Reopen in Container".
- You should be dropped in `/workspace` folder inside the container, with all the files and environment installed.
- Any changes to files in the devcontainer will be reflected on the host machine, and vice versa.

### Running experiment
- Copy the `template.py` file into your own folder.
- Fill in the parameters and datasets in `template.py` and rename it to `<your_file>`.
- The Benchmark needs your `Binarized Dataset`, `Graph Dataset`, and parameters for the different models.
    - An example with MultiValueXOR is in `test/test_bm_xor.py`.
    - Do NOT change `epochs`, `num_test_reps`
    - If your experiments are large (time-wise), can change `gpu_polling_rate` to 0.5
    - If your experiments are small (time-wise. e.g. training epoch takes 5 secs), change `gpu_polling_rate` to 0.01
- Epochs = 50, num_test_reps = 5 (these have already been set as default, so no need to pass these parameters in <your_file> for benchmarking)
- To test the script - activate the environment using `pixi shell` and run `python <your_file>`.
- To run the benchmark use `pixi run bm <your_folder>/<your_file> <gpuid>`. 

    - to get <gpuid>, run `nvtop` or `nvidia-smi`, check which gpu is free, choose that.

    - This will run the benchmark in a new tmux session, so that the experiment does not stop if the devcontainer is disconnected. 

    - To view, run `tmux attach`.

- Output is a csv file (results) and a pickle file(splits)

    - Results are on different validation splits, using different models, and reported as 'all classes' and 'per class'


## Details

### Environment Setup
Uses [Pixi](https://pixi.sh) to create and manage the environment. The environement is defined in the `pixi.yaml` file. 
To create the environment, run:

```bash
pixi install
```

### Installing additaional packages
If some package is missing from the environment, you can add it using `pixi add`.
If the package is available with conda:
``` bash
pixi add <package-name> 
```
If the package is only available with pip:
``` bash
pixi add --pypi <package-name>
```

### Activating the environment.
If you are using VsCode, the environement should get activated automatically.
To activate it manually:
``` bash
pixi shell
```

To verify that you are using the correct environement, you can run `which python` and check if the path points to the something like `...folder/.pixi/envs/.../python`



##

# TODO:
- [x] Add pixi environement
- [x] Add devcontainer
- [ ] Benchmarks
    - [x] GTM
    - [x] TM
    - [x] XGBoost
    - [x] Memory measurements
    - [ ] Energy measurements
- [x] Test environment
- [x] Test devcontainer
- [x] Add detailed instructions
