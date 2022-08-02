from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.env_gen import EnvGeneration2
from configs.sweeps.all import HyperSweepConfig


@dataclass
class EvoDataScratchSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'evo_data'
    n_updates: int = 100000

    exp_name: List[Any] = field(default_factory=lambda: [
        # "2",
        # "3",
        "scratch",
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        EnvGeneration2(),
        # None,
    ])
    task: List[Any] = field(default_factory=lambda: [
        "diameter",
        # "pathfinding",
    ])
    model: List[Any] = field(default_factory=lambda: [
        "NCA",
        # "GCN",
    ])
    shared_weights: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
    #     # 4,
    #     # 8,
        # 32,
        # 48,
        96,
        # 128,
    #     # 256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
    #     # 48,
        64,
        # 96,
    #     # 128,
    ])
    loss_interval: List[Any] = field(default_factory=lambda: [
    #     # 4,
    #     # 8,
    #     # 16,
        # 32,
        64,
    ])
    symmetric_conv: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    