import json
import numpy as np
from pdb import set_trace as TT
import torch as th
from configs.config import Config
from mazes import Mazes, render_discrete
from models.gnn import GCN
from models.nn import PathfindingNN

from utils import Logger, get_discrete_loss, get_mse_loss, to_path
            

def evaluate_train(model, cfg):
    """Evaluate the trained model on the training set."""
    # TODO: 
    pass


def evaluate(model: PathfindingNN, maze_data: Mazes, batch_size: int, name: str, cfg: Config, 
        is_eval: bool = False):
    """Evaluate the trained model on a test set.

    Args:
        model (PathfindingNN): The model to evaluate. 
        is_eval (bool): Whether we are evaluating after training, in which case we collect more stats. (Otherwise, we 
            are validating during training, and compute less expensive metrics.)
    """
    mazes_onehot, mazes_discrete, target_paths = maze_data.mazes_onehot, maze_data.mazes_discrete, \
        maze_data.target_paths

    batch_idxs = np.arange(0, mazes_onehot.shape[0])
    np.random.shuffle(batch_idxs)
    model.eval()
    # assert name in ['train', 'validate', 'test'], "Name of evaluation must be 'train', 'test' or 'validate'."

    if is_eval:
        n_eval_minibatches = mazes_onehot.shape[0] // batch_size
    else:
        n_eval_minibatches = 1

    with th.no_grad():
        losses = []
        discrete_losses = []
        eval_pcts_complete = []
        if is_eval:
            eval_completion_times = []
            # What would the loss be if the model output all zeros (sometimes this happens!). Treat this as 0 accuracy, so we 
            # can better analyzer performance of good models.
            baseline_losses = []
        i = 0

        for i in range(n_eval_minibatches):
            batch_idx = batch_idxs[np.arange(i*batch_size, (i+1)*batch_size, dtype=int)]
            x0 = mazes_onehot[batch_idx]
            # x0_discrete = test_mazes_discrete[batch_idx]
            x = model.seed(batch_size=batch_size, width=cfg.width, height=cfg.height)
            target_paths_minibatch = target_paths[batch_idx]
            path_lengths = target_paths_minibatch.float().sum((1, 2)).cpu().numpy()
            model.reset(x0)

            if is_eval:
                completion_times = np.empty(batch_size)
                completion_times.fill(np.nan)

            for j in range(cfg.n_layers):
                x = model(x)

                if j == cfg.n_layers - 1 or is_eval:
                    x_clipped = th.clip(x, 0, 1)

                if j == cfg.n_layers - 1:
                    if is_eval:
                        baseline_loss = get_mse_loss(th.zeros_like(x), target_paths_minibatch).item()
                        baseline_losses.append(baseline_loss)
                    eval_loss = get_mse_loss(x_clipped, target_paths_minibatch).item()
                    # print(baseline_loss, eval_loss)
                    eval_discrete_loss = get_discrete_loss(x_clipped, target_paths_minibatch).cpu().numpy()
                    eval_pct_complete = np.sum(eval_discrete_loss.reshape(batch_size, -1).sum(1) == 0) / batch_size
                    losses.append(eval_loss)
                    discrete_losses.append(eval_discrete_loss.mean().item())
                    eval_pcts_complete.append(eval_pct_complete)

                if is_eval:
                    eval_discrete_loss = get_discrete_loss(x_clipped, target_paths_minibatch).cpu()
                    completion_times = np.where(eval_discrete_loss.reshape(batch_size, -1).sum(dim=1) == 0, j, completion_times)
                    completion_times /= path_lengths
                    eval_completion_times.append(np.nanmean(completion_times))

                    # fig, ax = plt.subplots(figsize=(10, 10))
                    # solved_maze_ims = np.hstack(render_discrete(x0_discrete[:cfg.render_minibatch_size], cfg))
                    # target_path_ims = np.tile(np.hstack(
                        # target_paths_mini_batch[:cfg.render_minibatch_size].cpu())[...,None], (1, 1, 3)
                        # )
                    # predicted_path_ims = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
                    # img = (img - img.min()) / (img.max() - img.min())
                    # predicted_path_ims = np.hstack(predicted_path_ims)
                    # predicted_path_ims = np.tile(predicted_path_ims, (1, 1, 3))
                    # imgs = np.vstack([solved_maze_ims, target_path_ims, predicted_path_ims])
                    # plt.imshow(imgs)
                    # plt.show()

    losses = np.array(losses)
    discrete_losses = np.array(discrete_losses)
    accs = 1 - losses
    discrete_accs = 1 - discrete_losses
    if is_eval:
        baseline_accs = 1 - np.array(baseline_losses)
        accs = (accs - baseline_accs) / (1 - baseline_accs)
        discrete_accs = (discrete_accs - baseline_accs) / (1 - baseline_accs)
    mean_accs = np.mean(accs)
    std_accs = np.std(accs)
    mean_discrete_accs = np.mean(discrete_accs)
    std_discrete_accs = np.std(discrete_accs)

    stats = {
        'accs': (mean_accs, std_accs),
        'discrete_accs': (mean_discrete_accs, std_discrete_accs),
        'losses': (np.mean(losses), np.std(losses)),
        'disc_losses': (np.mean(discrete_losses), np.std(discrete_losses)),
        'pct_complete': (np.mean(eval_pcts_complete), np.std(eval_pcts_complete)),
    }
    if is_eval:
        stats.update({
            'completion_time': (np.nanmean(eval_completion_times), np.nanstd(eval_completion_times)),
        })
        # Dump stats to a json file
        with open(f'{cfg.log_dir}/{name}_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        print(f'{name} stats:')
        print(json.dumps(stats, indent=4))
    model.train()

    return stats


# def evaluate(model, data, batch_size, cfg, render=False):
#     # TODO: re-use this function when evaluating on training/test set, after training.
#     """Evaluate the trained model on a dataset without collecting gradients."""
#     mazes_onehot, mazes_discrete, target_paths = data.mazes_onehot, data.mazes_discrete, \
#         data.target_paths

#     with th.no_grad():
#         test_losses = []
#         i = 0

#         for i in range(mazes_onehot.shape[0] // batch_size):
#             batch_idx = np.arange(i*batch_size, (i+1)*batch_size, dtype=int)
#             x0 = mazes_onehot[batch_idx]
#             x0_discrete = mazes_discrete[batch_idx]

#             # if cfg.model == "MLP":
#                 # x = x0

#             # else:
#             x = model.seed(batch_size=batch_size)

#             target_paths_mini_batch = target_paths[batch_idx]
#             model.reset(x0)

#             for j in range(cfg.n_layers):
#                 x = model(x)

#                 if j == cfg.n_layers - 1:
#                     test_loss = get_mse_loss(x, target_paths_mini_batch).item()
#                     test_losses.append(test_loss)
#                     # if render:
#                     #     fig, ax = plt.subplots(figsize=(10, 10))
#                     #     solved_maze_ims = np.hstack(render_discrete(x0_discrete[:cfg.render_minibatch_size]))
#                     #     target_path_ims = np.tile(np.hstack(
#                     #         target_paths_mini_batch[:cfg.render_minibatch_size].cpu())[...,None], (1, 1, 3)
#                     #         )
#                     #     predicted_path_ims = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
#                     #     # img = (img - img.min()) / (img.max() - img.min())
#                     #     predicted_path_ims = np.hstack(predicted_path_ims)
#                     #     predicted_path_ims = np.tile(predicted_path_ims, (1, 1, 3))
#                     #     imgs = np.vstack([solved_maze_ims, target_path_ims, predicted_path_ims])
#                     #     plt.imshow(imgs)
#                     #     plt.show()

#     mean_eval_loss = np.mean(test_losses)
#     # print(f'Mean evaluation loss: {mean_eval_loss}') 

#     return mean_eval_loss

