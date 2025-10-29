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
- Run the `make_env.sh` script to create `.env` file for the devcontainer.
```bash
sh .devcontainer/make_env.sh
```
- Open repository as a folder (project) in VSCode.
- When prompted to open in devcontainer, select "Reopen in Container".
- If not prompted, open the command palette (Ctrl+Shift+P), and select "Dev Containers: Reopen in Container".
- You should be dropped in `/workspace` folder inside the container, and it should have all the files.
- Any changes to files in the devcontainer will be reflected on the host machine, and vice versa.

## Running experiment
- Copy the `template.py` file into your own folder.
- Fill in the parameter and datasets.
- The Benchmark needs your `Binarized Dataset`, `Graph Dataset`, and parameters for the different models.
- An example for MultiValueXOR is in `test/test_bm_xor.py`.
- First test the script. Activate the environment using `pixi shell` and run `python <file>`.
- To run the benchmark use `pixi run bm <file> <gpuid>`. This will run the benchmark in a new tmux session, so that the exepriment does not stop if the devcontainer is disconnected.

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
