from dataclasses import dataclass, field
from typing import Any, List
import hydra
from configs.env_gen import EnvGeneration50

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
        "5",
        "6",
        "7",
        "8",
        "9",
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
        False,
    ])
    cut_corners: List[Any] = field(default_factory=lambda: [
        True,
        False,
    ])
    n_hid_chan: List[Any] = field(default_factory=lambda: [
        ### No shared weights only
        4,
        6,

        8,

        ### No shared weights only
        10,
        12,
        14,

        16,

        ### No shared weights only
        20,

        24,

        ### No shared weights only
        28,

        32,

        ### Shared weights only
        48,
        96,
        128,
        160,
        192,
        224,
        256,
        # 512,
        # 1024,
    ])
    n_layers: List[Any] = field(default_factory=lambda: [
        # 48,
        64,
        # 96,
        # 128,
    ])
    learning_rate: List[Any] = field(default_factory=lambda: [
        0.0001,
        # 0.00005,
    ])
