import os
from pathlib import Path

import torch as th

from configs.config import Config
from mazes import Tilesets
import models
from utils import get_ce_loss, get_mse_loss, maze_gen_loss, solnfree_pathfinding_loss


def set_exp_name(cfg: Config):
    # BACKWARD COMPATABILITY HACK. FIXME: Remove this when all experiments from before `full_exp_name` are obsolete.
    # if '-' in cfg.exp_name:
        # cfg.exp_name = cfg.exp_name.split('_')[-1]

    # Initialize the Tileset object and attach it to the global config.
    cfg.tileset=getattr(
            Tilesets, cfg.task.upper(),
        )()

    validate(cfg)

    sep = os.path.sep

    cfg.full_exp_name = ''.join([
        f"{cfg.task}", sep,
        f"{cfg.model}", sep,
        (f"genEnv" if cfg.env_generation is not None else ""), sep,
        ("noShared" if not cfg.shared_weights else ""),
        ("_noSkip" if not cfg.skip_connections else ""),
        ("_maxPool" if cfg.max_pool else ""),
        (f"_{cfg.kernel_size}-kern" if cfg.kernel_size != 3 else ""),
        f"_{cfg.n_hid_chan}-hid",
        f"_{cfg.n_layers}-layer",
        f"_lr-{'{:.0e}'.format(cfg.learning_rate)}",
        (f"_{cfg.loss_fn}-loss"),
        f"_{cfg.n_data}-data",
        (f"_{cfg.loss_interval}-loss" if cfg.loss_interval != cfg.n_layers else ''),
        ("_cutCorners" if cfg.cut_conv_corners and cfg.model == "NCA" else ""),
        ("_symmConv" if cfg.symmetric_conv and cfg.model == "NCA" else ""),
        ('_sparseUpdate' if cfg.sparse_update else ''),
        ('_edgeFeats' if cfg.positional_edge_features else ''),
        ('_travEdges' if cfg.traversable_edges_only else ''),
        f"_{cfg.exp_name}",
    ])
    cfg.log_dir = os.path.join(Path(__file__).parent.parent, "runs", cfg.full_exp_name)

    # Has to happen after setting the log_dir (should store name as different attribute).
    if cfg.task == "pathfinding_solnfree":
        cfg.loss_fn = solnfree_pathfinding_loss
    elif cfg.task == "maze_gen":
        cfg.loss_fn = maze_gen_loss
    elif cfg.loss_fn == "mse":
        cfg.loss_fn = get_mse_loss
    elif cfg.loss_fn == "ce":
        cfg.loss_fn = get_ce_loss
    else:
        raise ValueError(f"Unknown loss function {cfg.loss_fn}.")

def validate(cfg: Config):
    cfg.device = "cuda" if th.cuda.is_available() else "cpu"
    model_cls = getattr(models, cfg.model)
    if not issubclass(model_cls, models.GNN):
        assert (cfg.traversable_edges_only is False and cfg.positional_edge_features is False), "Hyperparameters " \
            "relating to representation of (sub-)grids as graphs are applicable only to GNNs."
    if not model_cls == models.GAT:
        assert cfg.positional_edge_features is False, f"Model class {model_cls} does not accept edge features."
    if not cfg.model == "NCA":
        cfg.symmetric_conv = False
        cfg.max_pool = False
    if cfg.symmetric_conv:
        cfg.cut_conv_corners = True
    cfg.n_in_chan = len(cfg.tileset.tiles)
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