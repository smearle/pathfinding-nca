import json
import numpy as np
from pdb import set_trace as TT
import torch as th
from mazes import render_discrete

from utils import get_discrete_loss, get_mse_loss, to_path
            

def evaluate_train(model, cfg):
    """Evaluate the trained model on the training set."""
    # TODO: 
    pass


def evaluate(model, maze_data, batch_size, name, cfg):
    """Evaluate the trained model on a test set."""
    model.eval()
    is_test = name == 'test'
    assert is_test or name in ['validate', 'train'], "Name of evaluation must be 'test' or 'validate'."
    # n_test_mazes = n_test_minibatches * cfg.minibatch_size
    test_mazes_onehot, test_mazes_discrete, target_paths = maze_data.mazes_onehot, maze_data.mazes_discrete, \
        maze_data.target_paths
    batch_idxs = np.arange(0, test_mazes_onehot.shape[0])
    np.random.shuffle(batch_idxs)

    if is_test:
        n_eval_minibatches = test_mazes_onehot.shape[0] // batch_size
    else:
        n_eval_minibatches = 1

    with th.no_grad():
        eval_losses = []
        eval_discrete_losses = []
        eval_pcts_complete = []
        if is_test:
            eval_completion_times = []
        i = 0

        for i in range(n_eval_minibatches):
            batch_idx = batch_idxs[np.arange(i*batch_size, (i+1)*batch_size, dtype=int)]
            x0 = test_mazes_onehot[batch_idx]
            # x0_discrete = test_mazes_discrete[batch_idx]
            x = model.seed(batch_size=batch_size)
            target_paths_minibatch = target_paths[batch_idx]
            path_lengths = target_paths_minibatch.float().sum((1, 2)).cpu().numpy()
            model.reset(x0)

            if is_test:
                completion_times = np.empty(batch_size)
                completion_times.fill(np.nan)

            for j in range(cfg.n_layers):
                x = model(x)

                if j == cfg.n_layers - 1:
                    eval_loss = get_mse_loss(x, target_paths_minibatch).item()
                    eval_discrete_loss = get_discrete_loss(x, target_paths_minibatch).cpu().numpy()
                    eval_pct_complete = np.sum(eval_discrete_loss.reshape(batch_size, -1).sum(1) == 0) / batch_size
                    eval_losses.append(eval_loss)
                    eval_discrete_losses.append(eval_discrete_loss.mean().item())
                    eval_pcts_complete.append(eval_pct_complete)

                if is_test:
                    eval_discrete_loss = get_discrete_loss(x, target_paths_minibatch).cpu()
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

    eval_accs = 1 - np.array(eval_losses)
    eval_discrete_accs = 1 - np.array(eval_discrete_losses)
    mean_eval_accs = np.mean(eval_accs)
    std_eval_accs = np.std(eval_accs)
    mean_eval_discrete_accs = np.mean(eval_discrete_accs)
    std_eval_discrete_accs = np.std(eval_discrete_accs)
    # print(f'Mean {name} loss: {mean_eval_loss}\nMean {name} discrete loss: {mean_discrete_eval_loss}') 

    stats = {
        'accs': (mean_eval_accs, std_eval_accs),
        'discrete_accs': (mean_eval_discrete_accs, std_eval_discrete_accs),
        'pct_complete': (np.mean(eval_pcts_complete), np.std(eval_pcts_complete)),
    }
    if is_test:
        stats.update({
            'completion_time': (np.nanmean(eval_completion_times), np.nanstd(eval_completion_times)),
        })
        # Dump stats to a json file
        with open(f'{cfg.log_dir}/{name}_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
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

