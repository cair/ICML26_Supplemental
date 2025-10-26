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

# TODO:
- [x] Add pixi environement
- [x] Add devcontainer
- [ ] Benchmarks
- [ ] Test environment
- [ ] Test devcontainer
- [ ] Add detailed instructions
