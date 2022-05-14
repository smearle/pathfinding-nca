from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.sweeps.all import HyperSweepConfig


@dataclass
class LossIntervalSweep(HyperSweepConfig):
    name: str = 'loss_interval'
    model: List[Any] = field(default_factory=lambda: [
        "NCA"
    ])
    task: List[Any] = field(default_factory=lambda: [
        "pathfinding",
        # "diameter",
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        None,
        # EnvGeneration(),
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
#         4,
#         8,
#         16,
#         24,
#         32,
        64,
        # 96,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
#         4,
#         8,
        # 16,
        # 32,
        96,
        # 128,
        # 256,
    ])
    loss_interval: List[Any] = field(default_factory=lambda: [
        4,
        8,
        16,
        32,
        64,
    ])
    shared_weights: List[bool] = field(default_factory=lambda: [
        False,
        True,
    ])

