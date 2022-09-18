from dataclasses import dataclass, field
from typing import Any, List
import hydra
from configs.env_gen import EnvGeneration2

from configs.sweeps.all import HyperSweepConfig


@dataclass
class WeightSharingSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'shared_weights'

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
        # "10",
        # "11",
        # "12",
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        None,
        # EnvGeneration2(),
    ])
    task: List[Any] = field(default_factory=lambda: [
        "pathfinding",
        # "diameter",
    ])
    model: List[Any] = field(default_factory=lambda: [
        "NCA",
        # "GCN",
    ])
    shared_weights: List[Any] = field(default_factory=lambda: [
        True,
        # False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
    #     # 4,
    #     # 8,
        # 32,

        48,
        96,
        128,

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
        32,
        # 64,
    ])
    symmetric_conv: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    
