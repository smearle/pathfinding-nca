from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.env_gen import EnvGeneration, EnvGeneration2
from configs.sweeps.all import HyperSweepConfig


@dataclass
class ScratchSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    exp_name: str = 'scratch'
    exp_name: List[Any] = field(default_factory=lambda: [
        # "0",
        # "1",
        # "2",
        # "3",
        # "4",
        # "10",
        # "11",
        # "12",
        "scratch",
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        # None,
        EnvGeneration2(),
    ])
    task: List[Any] = field(default_factory=lambda: [
        # "pathfinding",
        # "pathfinding_solnfree",
        "diameter",
        # "traveling",
        # "maze_gen",
    ])
    max_pooling: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    model: List[Any] = field(default_factory=lambda: [
        "NCA",
        # "GAT",
        # "GCN",
        # "MLP",
        # "BfsNCA",
        # "FixedBfsNCA",
        # "FixedDfsNCA",
    ])
    traversable_edges_only: List[Any] = field(default_factory=lambda: [
        False,
        # True,
    ])
    positional_edge_features: List[Any] = field(default_factory=lambda: [
        False,
        # True,
    ])
    shared_weights: List[Any] = field(default_factory=lambda: [
        True,
        # False,
    ])
    cut_conv_corners: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        # 14,
        # 8,
        # 16,
        # 24,
        # 32,
        # 48,
        # 96,
        128,
        # 256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        # 16,
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
    # # symmetric_conv: List[Any] = field(default_factory=lambda: [
    #     # True,
    #     # False,
    # # ])
    learning_rate: List[Any] = field(default_factory=lambda: [
        # 0.0001,
        0.00005,
        # 0.00001,
    ])
    