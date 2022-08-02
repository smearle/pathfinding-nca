from dataclasses import dataclass, field
from typing import Any, List
import hydra

from configs.env_gen import EnvGeneration


@dataclass
class HyperSweepConfig():
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    # FIXME: Why can't these guys just be ints, since we won't sweep over their value?
    name: str = 'batch'
    # save_interval: List[Any] = field(default_factory=lambda: [
        # 10000,
    # ])
    save_interval: int = 1000
    # log_interval: List[Any] = field(default_factory=lambda: [
        # 10000,
    # ])
    log_interval: int = 1000
    # n_updates: List[Any] = field(default_factory=lambda: [
        # 50000,
    # ])
    n_updates: int = 50000

    # exp_name: List[Any] = field(default_factory=lambda: [
        # "2",
        # "3",
        # "debug",
    # ])
    # env_generation: List[Any] = field(default_factory=lambda: [
        # EnvGeneration(),
        # None,
    # ])
    # task: List[Any] = field(default_factory=lambda: [
        # "pathfinding",
        # "diameter",
    # ])
    # model: List[Any] = field(default_factory=lambda: [
    #     # "NCA",
    #     "GCN",
    # ])
    # shared_weights: List[Any] = field(default_factory=lambda: [
    #     True,
    #     False,
    # ])
    # n_hid_chan: List[Any] = field(default_factory=lambda: [
    # #     # 4,
    # #     # 8,
    # #     # 32,
    #     48,
    #     96,
    #     128,
    # #     # 256,
    # ])
    # n_layers: List[Any] = field(default_factory=lambda: [
    # #     # 48,
    #     64,
    #     # 96,
    # #     # 128,
    # ])
    # loss_interval: List[Any] = field(default_factory=lambda: [
    # #     # 4,
    # #     # 8,
    # #     # 16,
    #     32,
    #     64,
    # ])
    # # symmetric_conv: List[Any] = field(default_factory=lambda: [
    #     # True,
    #     # False,
    # # ])
    