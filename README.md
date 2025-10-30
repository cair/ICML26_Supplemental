# Experiments and Benchmarks for the GraphTsetlinMachine paper

## Environment Setup
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

## Devcontainer
Devcontainer configuration is provided for VSCode.

- Remote SSH into cair-gpu17
- Clone this repo somewhere.
- Run the `.devcontainer/build` script to create `devcontainer.json` file for the devcontainer, with proper user and stuff.
```bash
./.devcontainer/build
```
- Open repository as a folder (project) in VSCode.
- When prompted to open in devcontainer, select "Reopen in Container".
- If not prompted, open the command palette (Ctrl+Shift+P), and select "Dev Containers: Reopen in Container".
- You should be dropped in `/workspace` folder inside the container, with all the files and environment installed.
- Any changes to files in the devcontainer will be reflected on the host machine, and vice versa.

## Running experiment
- Copy the `template.py` file into your own folder.
- Fill in the parameter and datasets.
- The Benchmark needs your `Binarized Dataset`, `Graph Dataset`, and parameters for the different models.
- An example for MultiValueXOR is in `test/test_bm_xor.py`.
- First test the script. Activate the environment using `pixi shell` and run `python <file>`.
- To run the benchmark use `pixi run bm <file> <gpuid>`. 

- - to get <gpuid>, run `nvtop` or `nvidia-smi`, check which gpu is free, choose that.

- - This will run the benchmark in a new tmux session, so that the experiment does not stop if the devcontainer is disconnected. 

- - To view, run `tmux attach`.

- - Output is a csv file (results) and a pickle file(splits)

- - Results are on different validation splits, using different models, and reported as 'all classes' and 'per class'

##

# TODO:
- [x] Add pixi environement
- [x] Add devcontainer
- [ ] Benchmarks
    - [x] GTM
    - [x] TM
    - [x] XGBoost
    - [x] Memory and energy measurements
    - [ ] Energy measurements
- [x] Test environment
- [x] Test devcontainer
- [ ] Add detailed instructions
