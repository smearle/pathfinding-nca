import os
from pathlib import Path

import torch as th

from configs.config import Config
import models


def set_exp_name(cfg: Config):
    # BACKWARD COMPATABILITY HACK. FIXME: Remove this when all experiments from before `full_exp_name` are obsolete.
    if '-' in cfg.exp_name:
        cfg.exp_name = cfg.exp_name.split('_')[-1]

    # assert '-' not in self.exp_name, "Cannot have hyphens in `exp_name` (to allow for a backward compatibility hack)"
    validate(cfg)
    # TODO: create distinct `full_exp_name` attribute where we'll write this thing, so we don't overwrite the user-
    # supplied `exp_name`.
    # In the meantime, using the presence of a hyphen to mean we have set the full `exp_name`:
    cfg.full_exp_name = ''.join([
        f"{cfg.model}",
        ("_diameter" if cfg.task == "diameter" else ""),
        ("_evoData" if cfg.env_generation is not None else ""),
        ("_noShared" if not cfg.shared_weights else ""),
        ("_noSkip" if not cfg.skip_connections else ""),
        ("_maxPool" if cfg.max_pool else ""),
        (f"_{cfg.kernel_size}-kern" if cfg.kernel_size != 3 else ""),
        f"_{cfg.n_hid_chan}-hid",
        f"_{cfg.n_layers}-layer",
        f"_lr-{'{:.0e}'.format(cfg.learning_rate)}",
        f"_{cfg.n_data}-data",
        (f"_{cfg.loss_interval}-loss" if cfg.loss_interval != cfg.n_layers else ''),
        ("_cutCorners" if cfg.cut_conv_corners and cfg.model == "NCA" else ""),
        ("_symmConv" if cfg.symmetric_conv and cfg.model == "NCA" else ""),
        ('_sparseUpdate' if cfg.sparse_update else ''),
        f"_{cfg.exp_name}",
    ])
    cfg.log_dir = os.path.join(Path(__file__).parent.parent, "runs", cfg.full_exp_name)

def validate(cfg: Config):
    cfg.device = "cuda" if th.cuda.is_available() else "cpu"
    model_cls = getattr(models, cfg.model)
    if not issubclass(model_cls, models.GNN):
        assert (cfg.traversable_edges_only is False and cfg.positional_edge_features is False), "Hyperparameters " \
            "relating to representation of (sub-)grids as graphs are applicable only to GNNs."
    if not cfg.model == "NCA":
        cfg.symmetric_conv = False
        cfg.max_pool = False
    if cfg.symmetric_conv:
        cfg.cut_conv_corners = True
    if cfg.task == "diameter":
        cfg.n_in_chan = 2
    if cfg.model == "FixedBfsNCA":
        assert cfg.task != "diameter", "No hand-coded model implemented for diameter task."
        # self.path_chan = self.n_in_chan + 1  # wait why is this??
        cfg.n_hid_chan = 7
        cfg.skip_connections = True
    if cfg.model == "FixedDfsNCA":
        cfg.n_hid_chan = 12
    # else:
    cfg.path_chan = cfg.n_in_chan
    cfg.load = True if cfg.render else cfg.load
    # self.minibatch_size = 1 if self.model == "GCN" else self.minibatch_size
    # self.val_batch_size = 1 if self.model == "GCN" else self.val_batch_size
    assert cfg.n_val_data % cfg.val_batch_size == 0, "Validation dataset size must be a multiple of val_batch_size."
    if cfg.sparse_update:
        assert cfg.shared_weights, "Sparse update only works with shared weights. (Otherwise early layers may not "\
            "be updated.)"
    if cfg.model == "GCN":
        cfg.cut_conv_corners = True
    elif cfg.cut_conv_corners:
        assert cfg.model == "NCA", "Cutting corners only works with NCA (optional) or GCN (forced)."
    if cfg.loss_interval is None:
        cfg.loss_interval = cfg.n_layers
    
    # For backward compatibility, we assume 50k updates, where each update is following a 32-batch of episodes. So
    # we ensure that we have approximately the same number of episodes given different batch sizes here.
    if cfg.minibatch_size != 32:
        cfg.n_updates = int(cfg.n_updates * 32 / cfg.minibatch_size) 

    assert cfg.n_layers % cfg.loss_interval == 0, "loss_interval should divide n_layers."
    if cfg.minibatch_size < 32:
        assert 32 % cfg.minibatch_size == 0, "minibatch_size should divide 32."
        cfg.n_updates = cfg.n_updates * 32 // cfg.minibatch_size

    if cfg.render:
        cfg.wandb = False
        # self.render_minibatch_size = 1