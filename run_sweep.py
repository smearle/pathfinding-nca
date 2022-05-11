"""Use wandb to perform hyperparameter sweeps. 

Off-loading the work we were previously doing manually in `run_batch.py` for training off to wandb."""
import os
from pathlib import Path
from pdb import set_trace as TT
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from main import main_experiment


@hydra.main(config_path="configs", config_name="batch")
def main_sweep(sweep_cfg: DictConfig):
    hyperparam_dict = yaml.load(open(os.path.join(Path(__file__).parent, 'configs/batch_hyperparams/all.yaml'), 'r'), 
                                Loader=yaml.FullLoader)
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