
from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.sweeps.all import HyperSweepConfig


@dataclass
class HandcodedSweep(HyperSweepConfig):
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    name: str = 'cut_corners'

    exp_name: List[Any] = field(default_factory=lambda: [
        "debug",
        # "0",
    ])
    cut_conv_corners: List[Any] = field(default_factory=lambda: [
        # True,
        False,
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
        "BfsNCA",
        # "NCA",
        # "GCN",
    ])
    shared_weights: List[Any] = field(default_factory=lambda: [
        True,
        # False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        # 4,
        # 8,
        # 32,
        48,
        # 96,
        # 128,
        # 256,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
    # #     # 48,
        64,
    #     # 96,
    # #     # 128,
    ])
    loss_interval: List[Any] = field(default_factory=lambda: [
    # #     # 4,
    # #     # 8,
    # #     # 16,
        32,
    #     64,
    ])
    symmetric_conv: List[Any] = field(default_factory=lambda: [
        # True,
        False,
    ])
    