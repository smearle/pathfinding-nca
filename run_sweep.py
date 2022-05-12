"""Use wandb to perform hyperparameter sweeps. 

Off-loading the work we were previously doing manually in `run_batch.py` for training off to wandb."""
import os
from pathlib import Path
from pdb import set_trace as TT
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from configs.config import BatchConfig, Config

from main import main_experiment

# TODO: The problem here is that we need to call `.set_exp_name()` (and `validate()`) on our configs before sending them
#   to wandb. Probably some way to do this...?

@hydra.main(config_path=None, config_name="batch_config")
def main_sweep(sweep_cfg: BatchConfig):
    dummy_exp_cfg = Config()
    hyperparam_dict = {k: {'values': v} for k, v in dict(sweep_cfg.sweep).items()}
    for k, v in sweep_cfg.items():
        if not hasattr(dummy_exp_cfg, k) and k not in hyperparam_dict:
            continue
        # setattr(hyperparam_dict, k, v)
        hyperparam_dict[k] = {'values': [v]}
        
    sweep_config = {
        'parameters': hyperparam_dict,
        'method': 'random',
        'metric': {
            'name': 'loss',
            'goal': 'minimize',
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='pathfinding-nca')
    print("Running wandb sweep:\n", sweep_id)
    wandb.agent(sweep_id, main_experiment)


if __name__ == "__main__":
    main_sweep()