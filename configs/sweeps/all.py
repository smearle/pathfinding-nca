from dataclasses import dataclass, field
from typing import Any, List
import hydra


@dataclass
class HyperSweepConfig():
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'batch'
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
        "2",
    ])

    model: List[Any] = field(default_factory=lambda: [
        # "GCN",
        "NCA",
    ])
    task: List[Any] = field(default_factory=lambda: [
        # "pathfinding",
        "diameter",
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        96,
        128,
        256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        # 48,
        64,
        96,
        128,
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        None,
        # EnvGeneration(),
    ])
    
