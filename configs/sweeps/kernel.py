
from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.env_gen import EnvGeneration
from configs.sweeps.all import HyperSweepConfig


@dataclass
class KernelSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""

    name: str = 'kernel'

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
        # "5",
        # "debug",

        "8",
        "9",
        "10",
        "11",
        "12",
    ])
    cut_corners: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    env_generation: List[Any] = field(default_factory=lambda: [
        # EnvGeneration(),
        None,
    ])
    task: List[Any] = field(default_factory=lambda: [
        "pathfinding",
        "diameter",
    ])
    model: List[Any] = field(default_factory=lambda: [
        "NCA",
        # "GCN",
    ])
    kernel_size: List[Any] = field(default_factory=lambda: [
        5,
        # 3,
    ])
    max_pool: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    shared_weights: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 32,

        48,
        96,
        128,

        # 256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        # 48,
        64,
        # 96,
        # 128,
    ])
    loss_interval: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 16,
        32,
        # 64,
    ])
    symmetric_conv: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    