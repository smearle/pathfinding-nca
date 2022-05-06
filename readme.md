# Pathfinding Neural Cellular Automata

Train Neural Cellular Automata (NCAs) to find optimal paths in grid-based mazes.

## Installation

First, modify `setup.sh` so that the command for torch installation matches your system, i.e., whether you have a GPU, 
and which version of CUDA it requires. You can also find the updated 
```
bash setup.sh
```

## Directions for future work
- Try generic skip connections (e.g., adding output after first layer to all following layers)
- ~~Remove "pool" logic. For now, only apply loss at the end of an `n_layers` length episode of pathfinding.~~ Later, we can appy loss at regular intervals during this episode.
- Have an option to use skip connection or not. i.e., the initial maze is concatenated with the model's hidden activations after each layer.
- ~~Add MLP~~
- Option to treat model as CA (with shared weights) or vanilla (distinct weights at each layer)
    - CA: model is 1 or 2 layers, training loop repeatedly passes input (and maybe computes loss part-way through). (This is what we're doing now).
    - vanilla: model is many distinct layers, training loop passes input only once.
- Apply loss at different intervals.
    - How will this mesh with model variants above?