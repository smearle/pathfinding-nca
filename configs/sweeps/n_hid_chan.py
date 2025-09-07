from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.sweeps.all import HyperSweepConfig


@dataclass
class HidChanSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'n_hid_chan'
    model: List[Any] = field(default_factory=lambda: [
        # "GCN",
        "NCA",
    ])
    task: List[Any] = field(default_factory=lambda: [
        "pathfinding",
        # "diameter",
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 16,
        # 32,
        # 48,
        # 96,
        128,
        # 256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        # 64,
        # 96,
        128,
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        None,
        # EnvGeneration(),
    ])
    loss_fn: List[Any] = field(default_factory=lambda: [
        # "mse",
        "ce",
    ])

