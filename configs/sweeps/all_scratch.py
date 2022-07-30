from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.env_gen import EnvGeneration
from configs.sweeps.all import HyperSweepConfig


@dataclass
class ScratchSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'scratch'
    save_interval: List[Any] = field(default_factory=lambda: [
        10000,
    ])
    log_interval: List[Any] = field(default_factory=lambda: [
        10000,
    ])
    n_updates: List[Any] = field(default_factory=lambda: [
        50000,
    ])

    exp_name: List[Any] = field(default_factory=lambda: [
        # "2",
        # "3",
        # "4",
        "scratch",
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        # EnvGeneration(),
        None,
    ])
    task: List[Any] = field(default_factory=lambda: [
        "pathfinding",
        # "diameter",
    ])
    # max_pooling: List[Any] = field(default_factory=lambda: [
        # True,
        # False,
    # ])
    model: List[Any] = field(default_factory=lambda: [
        # "NCA",
        # "GAT",
        "GCN",
        # "MLP",
        # "FixedBfsNCA",
        # "FixedDfsNCA",
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
    # #     # 4,
    # #     # 8,
        # 16,
        # 24,
    # #     # 32,
        # 48,
        # 96,
        128,
        # 256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        # 16,
    # #     # 48,
        64,
    #     # 96,
    # #     # 128,
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
    ])
    