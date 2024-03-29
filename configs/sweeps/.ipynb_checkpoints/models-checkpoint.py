from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.sweeps.all import HyperSweepConfig


@dataclass
class ModelSweep(HyperSweepConfig):
    name: str = 'models'
    model: List[Any] = field(default_factory=lambda: [
        "MLP",
#         "GCN"
        # "NCA",
        # "FixedBfsNCA",
    ])
    task: List[Any] = field(default_factory=lambda: [
        "diameter",
        "pathfinding",
    ])
    shared_weights: List[bool] = field(default_factory=lambda: [
        True,
        # False
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        4,
        8,
        16,
        # 24,
        # 32,
        # 64,
        # 96,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
#         4,
        # 8,
        # 16,
        # 32,
        # 96,
        # 128,
        256,
        # 512,
    ])
#     loss_interval: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 16,
#         32,
        # 64,
#     ])
    learning_rate: List[Any] = field(default_factory=lambda: [
        0.00005,
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        None,
        # EnvGeneration(),
    ])

