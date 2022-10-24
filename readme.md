# Pathfinding Neural Cellular Automata

Train Neural Cellular Automata (NCAs) to find optimal paths in grid-based mazes.

This project explores the algorithmic alignment of NCAs with path-finding problems in grid-based mazes, comparing these models to Graph Convolutional Networks (GCNs) and Multilayer Perceptrons (MLPs).

We also hand-code Breadth- and Depth-First Search implementations using pathfinding NCAs, to constructively illustrate this algorithmic alignment.

## Installation

This code was developed with python 3.9. First, modify `setup.sh` so that the command for torch installation matches your system, i.e., whether you have a GPU, 
and which version of CUDA it requires. You can also find the latest installation instructions for pytorch [here](https://pytorch.org/get-started/locally/). Then, run:
```
bash setup.sh
```

## Use

To run experiments involving multiple (repeated) runs over various hyper-parameters, we use the `BatchConfig` classes defined in `configs/sweeps`. These will define, for various hyper-parameters, lists of values for these hyperparemeters, with which experiments will be run in sequence (or in parallel, by submitting a series of independent jobs, using SLURM). For example, running
```
python run_batch.py +sweep=kernel slurm=True
```
 will submit a series of jobs on a SLURM cluster, involving experiments with NCAs with differently-sized convolutional kernels, by loading the `KernelSweep` config defined in `configs/sweeps/kernel.py`.
 
 When running for the first time, you will need to run this file with `gen_new_data=True` so that a new dataset of mazes and their solutions are generated for training.
 
 Note that you can also use `run_batch.py` to run a single experiment, by specifying a single value for each hyperparemeter in your sweep config. For example running `python run_batch.py +sweep=all` will run one experiment with the default settings detailed in the `Config` class in `configs/config.py`. By default jobs will be launched locally.

## Directions for future work / TODOs
- Making use of / reference to our hand-coded BFS and DFS NCAs, implement an NCA for computing the diameter of a maze, using the algorithm described in the paper "[Optimal distributed all pairs shortest paths and applications](https://dl.acm.org/doi/abs/10.1145/2332432.2332504)"
- Try generic skip connections (e.g., adding output after first layer to all following layers)
