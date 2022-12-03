from dataclasses import dataclass, field
from typing import Any, List

from configs.env_gen import EnvGeneration2
from configs.sweeps.all import HyperSweepConfig


@dataclass
class GNNGridRepSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'gnn_grid_rep'
    exp_name: List[Any] = field(default_factory=lambda: [
        "0",
        "1",
        "2",
        "3",
        "4",
        # "5",
        # "6",
        # "7",
        # "8",
        # "9",
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        # EnvGeneration(),
        None,
    ])
    task: List[Any] = field(default_factory=lambda: [
        "pathfinding",
        # "diameter",
    ])
    model: List[Any] = field(default_factory=lambda: [
        # "NCA",
        # "GAT",
        "GCN",
        # "MLP",
        # "FixedBfsNCA",
        # "FixedDfsNCA",
    ])
    traversable_edges_only: List[Any] = field(default_factory=lambda: [
        False,
        True,
    ])
    positional_edge_features: List[Any] = field(default_factory=lambda: [
        False,
        # True,
    ])
    shared_weights: List[Any] = field(default_factory=lambda: [
        True,
        False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 16,
        # 24,
        # 32,
        # 48,
        96,
        128,
        256,
        # 512,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        16,
        32,
        # 48,
        64,
        # 96,
        # 128,
    ])
    # loss_interval: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
    # ])
    learning_rate: List[Any] = field(default_factory=lambda: [
        # 0.0001,
        0.00005,
    ])
    