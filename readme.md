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

Train a model with the default settings using 
```
python main.py
```
The default settings for an experiment are defined and described in the `Config` class in `configs/config.py` (note that we use [hydra](https://github.com/facebookresearch/hydra) for config files).

Config settings can be overwritten in the command line, e.g.
```
python main.py model=GCN
```

To run experiments involving multiple (repeated) runs over various hyper-parameters, we use the `BatchConfig` classes defined in `configs/sweeps`. These will define, for various hyper-parameters, lists of values for these hyperparemeters, with which experiments will be run in sequence (or in parallel, by submitting a series of independent jobs, using SLURM). For example, running
```
python run_batch.py +sweep=kernel slurm=True
```
 will submit a series of jobs on a SLURM cluster, involving experiments with NCAs with differently-sized convolutional kernels, by loading the `KernelSweep` config defined in `configs/sweeps/kernel.py`.

## Directions for future work / TODOs
- Making use of / reference to our hand-coded BFS and DFS NCAs, implement an NCA for computing the diameter of a maze, using the algorithm described in the paper "[Optimal distributed all pairs shortest paths and applications](https://dl.acm.org/doi/abs/10.1145/2332432.2332504)"
- Try generic skip connections (e.g., adding output after first layer to all following layers)