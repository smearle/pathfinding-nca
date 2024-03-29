import os
from pdb import set_trace as TT
import pickle
import PIL

from matplotlib import image, pyplot as plt
import numpy as np
# from ray.util.multiprocessing import Pool
from timeit import default_timer as timer
import torch as th
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from tqdm import tqdm
import wandb

from configs.config import Config
from configs.env_gen import EnvGeneration
from evaluate import evaluate
from mazes import Mazes, Tiles, bfs_grid, diameter, get_shortest_path, get_target_diam, get_target_path, render_multihot
from models.gnn import GCN, GNN
from models.nn import PathfindingNN
from utils import (backup_file, corner_idxs_3x3, corner_idxs_5x5, count_parameters, delete_backup, get_discrete_loss, 
    get_mse_loss, Logger, to_path, save)


def train(model: PathfindingNN, opt: th.optim.Optimizer, maze_data: Mazes, maze_data_val: Mazes, 
        maze_data_test_32: Mazes, target_paths: th.Tensor, logger: Logger, cfg: Config, cfg_32):
    tb_writer = None
    loss_fn = cfg.loss_fn
    mazes_onehot, edges, edge_feats, maze_ims = maze_data.mazes_onehot, maze_data.edges, maze_data.edge_feats, \
        maze_data.maze_ims
    print("Done unpacking mazes.")
    minibatch_size = min(cfg.minibatch_size, cfg.n_data)
    lr_sched = th.optim.lr_scheduler.MultiStepLR(opt, [10000], 0.1)
    logger.last_time = timer()
    hid_states = model.seed(batch_size=mazes_onehot.shape[0])

    n_params = count_parameters(model, cfg)
    print(f'Number of learnable model parameters: {n_params}')
    if cfg.wandb:
        if logger.n_step == 0:
            wandb.log({'n_params': n_params}, step=0)

    env_gen_cfg: EnvGeneration = cfg.env_generation
    if env_gen_cfg is not None:
        last_evo = 0
        loaded_env_losses = False
        if cfg.load:
            try:
                env_losses = pickle.load(open(f'{cfg.log_dir}/env_losses.pkl', 'rb'))
                loaded_env_losses = True
            except FileNotFoundError:
                print("Env losses array not found. Initializing from scratch.")
        if not loaded_env_losses:
            env_losses = th.empty((cfg.n_data))
            env_losses.fill_(-np.inf)

    for i in tqdm(range(logger.n_step, cfg.n_updates)):
        if tb_writer is None:
            tb_writer = SummaryWriter(log_dir=cfg.log_dir)
        if i == 0:
            tb_writer.add_scalar('model/n_params', n_params, 0)
        with th.no_grad():
            # Randomly select indices of data-points on which to train during this update step (i.e., minibatch)
            batch_idx = np.random.choice(hid_states.shape[0], minibatch_size, replace=False)
            render_batch_idx = batch_idx[:cfg.render_minibatch_size]

            # x = th.zeros(x_maze.shape[0], n_chan, x_maze.shape[2], x_maze.shape[3])
            # x[:, :4, :, :] = x_maze
            x0 = mazes_onehot[batch_idx].clone()
            target_paths_mini_batch = target_paths[batch_idx]

            e0 = None
            if cfg.traversable_edges_only:
                e0 = [edges[i].to(cfg.device) for i in batch_idx]
            ef = None
            if cfg.positional_edge_features:
                ef = [edge_feats[i].to(cfg.device) for i in batch_idx]
            model.reset(x0, e0=e0, edge_feats=ef)

        # TODO: move initial auxiliary state to model? Probably a better way...
        x = hid_states[batch_idx]

        # else:
        # FYI: this is from the differentiating NCA textures notebook. Weird checkpointing behavior indeed! See the
        #   comments about the `sparse_updates` arg. -SE
        # The following line is equivalent to this code:
        # for k in range(cfg.n_layers):
            # x = model(x)
        # It uses gradient checkpointing to save memory, which enables larger
        # batches and longer CA step sequences. Surprisingly, this version
        # is also ~2x faster than a simple loop, even though it performs
        # the forward pass twice!
        # x = th.utils.checkpoint.checkpoint_sequential([model]*cfg.n_layers, 32, x)

        n_subepisodes = cfg.n_layers // cfg.loss_interval
        loss = 0
        if cfg.env_generation:
            env_losses[batch_idx] = 0

        for _ in range(n_subepisodes):

            # Hackish way of storing gradient only at last "chunk" (with checkpointing), or not.
            if cfg.sparse_update:
                x = th.utils.checkpoint.checkpoint_sequential([model]*cfg.loss_interval, min(16, cfg.loss_interval), x)
            else:
                for _ in range(cfg.loss_interval):
                    x = model(x)

            # loss += get_mse_loss(x, target_paths_mini_batch)
            out_paths = to_path(x)
            batch_errs = loss_fn(out_paths, target_paths_mini_batch)
            # err = (out_paths - target_paths_mini_batch).square()
            loss = loss + batch_errs.mean()

            if cfg.env_generation:
                env_losses[batch_idx] += batch_errs.detach()
        
        loss = loss / n_subepisodes
        if cfg.env_generation:
            env_losses[batch_idx] = env_losses[batch_idx] / n_subepisodes
        discrete_loss = get_discrete_loss(x, target_paths_mini_batch).float().mean()

        with th.no_grad():
            if "Fixed" not in cfg.model:

                # FIXME: why are we resetting here??
                model.reset(x0, e0=e0, edge_feats=ef)

                loss.backward()

                # for p in model.parameters():
                for name, p in model.named_parameters():

                    if p.grad is None:
                        # assert cfg.model == "BfsNCA", f"Gradient of parameter {name} should not be None."
                        print(f"Gradient of parameter {name} is None.")
                        continue
                    # print(f"Gradient of parameter {name} has shape {p.grad.shape}.")

                    if "weight" in name:
                        # If this is a graph neural net, we are already ignoring the corners (they are not neighbors as
                        # defined by our grid representation of the maze).
                        if not isinstance(model, GNN) and cfg.cut_conv_corners:
                            # Zero out all the corners
                            if p.shape[-2:] == (3, 3):
                                p.grad[:, :, corner_idxs_3x3[:, 0], corner_idxs_3x3[:, 1]] = 0
                            elif p.shape[-2:] == (5, 5):
                                p.grad[:, :, corner_idxs_5x5[:, 0], corner_idxs_5x5[:, 1]] = 0

                        if cfg.symmetric_conv:
                            # Force the NCA to behave equivalently to a GCN (this is a sanity check).
                            symm_grad = p.grad[:, :, 1, 0] + p.grad[:, :, 0, 1] + p.grad[:, :, 1, 2] + p.grad[:, :, 2, 1]
                            symm_grad /= 4
                            p.grad[:, :, 1, 0] = p.grad[:, :, 0, 1] = p.grad[:, :, 1, 2] = p.grad[:, :, 2, 1] = symm_grad

                        elif issubclass(type(model), GNN):
                            # Match the symmetric conv above.
                            p.grad /= 4

                    p.grad /= (p.grad.norm()+1e-8)     # normalize gradients 

                opt.step()
                opt.zero_grad()

            # lr_sched.step()
            tb_writer.add_scalar("training/loss", loss.item(), i)
            if cfg.wandb:
                wandb.log({"training/loss": loss.item()}, step=i)
            logger.log(loss=loss.item(), discrete_loss=discrete_loss.item())
                    
            key_ckp_step = ((i > 0) and (i % 50000 == 0))
            last_step = (i == cfg.n_updates - 1)
            if i % cfg.save_interval == 0 or last_step or key_ckp_step:
                log_dir = cfg.log_dir
                if key_ckp_step:
                    log_dir = os.path.join(cfg.log_dir, f"iter_{i}")
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                save(model, opt, logger, log_dir)
                # print(f'Saved CA and optimizer state dicts and maze archive to {log_dir}')
                if env_gen_cfg is not None:
                    maze_data = Mazes(cfg, evo_mazes={
                        "mazes_onehot": mazes_onehot,
                        "mazes_discrete": mazes_onehot.argmax(dim=1),
                        "target_paths": target_paths,
                        "maze_ims": maze_ims,
                        "edges": edges,
                        "edge_feats": edge_feats,
                    })
                    evolved_data_path = os.path.join(log_dir, "evo_mazes.pkl")
                    evolved_data_losses_path = os.path.join(log_dir, "env_losses.pkl")
                    backup_file(evolved_data_path)
                    backup_file(evolved_data_losses_path)
                    with open(evolved_data_path, "wb") as f:
                        pickle.dump(maze_data, f)
                    with open(evolved_data_losses_path, "wb") as f:
                        pickle.dump(env_losses, f)
                    delete_backup(evolved_data_path)
                    delete_backup(evolved_data_losses_path)

            if i % cfg.eval_interval == 0:
                val_stats = evaluate(model, maze_data_val, cfg.val_batch_size, "validate", cfg)
                logger.log_val(val_stats)
                for k, v in val_stats.items():
                    tb_writer.add_scalar(f"validation/mean_{k}", v[0], i)
                    if cfg.wandb:
                        wandb.log({f"validation/mean_{k}": v[0]}, step=i)
                if cfg.model != "MLP":
                    val_stats_32 = evaluate(model, maze_data_test_32, cfg.val_batch_size, "validate", cfg_32)
                    # logger.log_val(val_stats_32)
                    for k, v in val_stats_32.items():
                        tb_writer.add_scalar(f"validation/32x32/mean_{k}", v[0], i)
                        if cfg.wandb:
                            wandb.log({f"validation/32x32/mean_{k}": v[0]}, step=i)

            if i % cfg.log_interval == 0 or last_step:
                log(logger, lr_sched, cfg)
            
            if i % cfg.log_interval == 0 or last_step:
                render_paths = np.hstack(to_path(x[:cfg.render_minibatch_size]).cpu())
                render_maze_ims = np.hstack(maze_ims[render_batch_idx])
                target_path_ims = np.hstack(target_paths[render_batch_idx].cpu())
                if cfg.manual_log or last_step or key_ckp_step:
                    vis_train(logger, render_maze_ims, render_paths, target_path_ims, render_batch_idx, cfg.log_dir)
                images = np.vstack((
                    render_maze_ims*255, 
                    np.tile(render_paths[...,None], (1, 1, 3))*255, 
                    np.tile(target_path_ims[...,None], (1, 1, 3))*255,
                    )) 
                tb_images = images.astype(np.uint8).transpose(2,0,1)
                tb_writer.add_image("examples", np.array(tb_images), i)
                if cfg.wandb:
                    images = wandb.Image(images, caption="Top: Input, Middle: Output, Bottom: Target")
                    wandb.log({"examples": images}, step=i)

            evo_loss_thresh = 1e-2
            # evo_loss_thresh = 1
            if env_gen_cfg is not None and loss < evo_loss_thresh and last_evo >= env_gen_cfg.gen_interval:
            # if True:
                last_evo = 0
                # and i % env_gen_cfg.gen_interval == 0 \

                # Select top k for mutattion. TODO: don't need to sort everything here.
                # ret = sorted(
                #     zip(env_losses, mazes_onehot, maze_ims, target_paths), key=lambda x: x[0], reverse=True)
                # ret = list(zip(*ret))
                # env_losses, mazes_onehot, maze_ims, target_paths = th.stack(ret[0]), th.stack(ret[1]), \
                #     np.stack(ret[2]), th.stack(ret[3])
                # offspring_maze_walls = mazes_onehot[:env_gen_cfg.evo_batch_size, :2].argmax(1)

                # Select random k for mutation.
                evo_batch_size = env_gen_cfg.evo_batch_size
                mut_idxs = th.randint(min(cfg.n_data, mazes_onehot.shape[0]), (evo_batch_size,))
                offspring_mazes_onehot = mazes_onehot[mut_idxs]
                offspring_maze_walls = mazes_onehot[mut_idxs, Tiles.WALL]

                # Mutate the maze walls (except at the border).
                disc_noise = th.randint(1, 2, offspring_maze_walls.shape)
                wall_mut_mask = th.rand(offspring_maze_walls.shape) < .1
                disc_noise *= wall_mut_mask
                disc_noise[:, 0, :] = disc_noise[:, -1] = 0
                disc_noise[:, :, 0] = disc_noise[:, :, -1] = 0
                offspring_maze_walls = (offspring_maze_walls + disc_noise) % 2
                offspring_mazes = offspring_maze_walls

                if cfg.task == "pathfinding":
                    # Mutate the source and target positions
                    n_src = th.sum(offspring_mazes_onehot[:, Tiles.SRC])
                    assert (n_src == evo_batch_size) and (n_src == th.sum(offspring_mazes_onehot[:, Tiles.TRG]))
                    src_idxs = th.argwhere(offspring_mazes_onehot[:, Tiles.SRC] == 1)
                    trg_idxs = th.argwhere(offspring_mazes_onehot[:, Tiles.TRG] == 1)
                    assert not th.any(th.sum(src_idxs == trg_idxs, 1) == 3)
                    # Evolve 5% of sources, 5% of targets.
                    src_mut_mask = th.rand(evo_batch_size) < 0.05
                    trg_mut_mask = th.rand(evo_batch_size) < 0.05
                    # Uniform randomly select new source/target locations.
                    # We'll use this heatmap, prefering max values to be our new location.
                    src_trg_loc_heat = th.rand((2, *offspring_mazes.shape))
                    src_trg_loc_heat[:, :, :, 0] = src_trg_loc_heat[:, :, :, -1] = -1
                    src_trg_loc_heat[:, :, 0, :] = src_trg_loc_heat[:, :, -1, :] = -1
                    src_loc_heat, trg_loc_heat = src_trg_loc_heat
                    # Don't let mutating sources overwrite old targets, in case we don't end up mutating the 
                    # corresponding target.
                    src_loc_heat[tuple(trg_idxs[:, i] for i in range(trg_idxs.shape[1]))] = -1
                    # Get new source locations.
                    flat_src_idxs_o = th.argmax(src_loc_heat.view(src_loc_heat.shape[0], -1), 1)
                    # NOTE: this is specific to 2D, could copy np.unravel_index type method to make it more general.
                    src_idxs_o = th.stack((th.arange(evo_batch_size), 
                        th.div(flat_src_idxs_o, src_loc_heat.shape[-2], rounding_mode='trunc'),
                        flat_src_idxs_o % src_loc_heat.shape[-2]), dim=1)
                    # src_idxs_o = (src_loc_heat == 
                    #     th.max(src_loc_heat, dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]).nonzero()
                    src_idxs[src_mut_mask] = src_idxs_o[src_mut_mask]
                    # Going from `argwhere` type indices to `where` type ones (tuples)
                    src_idxs = tuple(src_idxs[:, i] for i in range(src_idxs.shape[1]))
                    # We mask out the new sources to ensure new targets don't overlap.
                    trg_loc_heat[src_idxs] = -1
                    # trg_idxs_o = (trg_loc_heat == 
                    #     th.max(trg_loc_heat, dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]).nonzero()
                    flat_trg_idxs_o = th.argmax(trg_loc_heat.view(trg_loc_heat.shape[0], -1), 1)
                    trg_idxs_o = th.stack((th.arange(evo_batch_size), 
                        th.div(flat_trg_idxs_o, trg_loc_heat.shape[-2], rounding_mode='trunc'),
                        flat_trg_idxs_o % trg_loc_heat.shape[-2]), dim=1)
                    trg_idxs[trg_mut_mask] = trg_idxs_o[trg_mut_mask]
                    trg_idxs = tuple(trg_idxs[:, i] for i in range(trg_idxs.shape[1]))

                    # src_mut = th.randint(0, max(cfg.width, cfg.height), (2, evo_batch_size)) * src_mut_mask
                    # src_idxs_o = (src_idxs[0], (src_idxs[1] + src_mut[0]) % cfg.width, (src_idxs[2] + src_mut[1]) % cfg.height)
                    # trg_mut = th.randint(0, max(cfg.width, cfg.height), (2, evo_batch_size)) * trg_mut_mask
                    # trg_idxs_o = (trg_idxs[0], (trg_idxs[1] + trg_mut[0]) % cfg.width, (trg_idxs[2] + trg_mut[1]) % cfg.height)

                    # Place source and target inside borders.
                    offspring_mazes[src_idxs] = Tiles.SRC
                    offspring_mazes[trg_idxs] = Tiles.TRG
                    n_src = th.sum(offspring_mazes == Tiles.SRC)
                    assert not (n_src != th.sum(offspring_mazes == Tiles.TRG) or n_src != evo_batch_size)
                    # assert th.sum(offspring_mazes == Tiles.SRC) == th.sum(offspring_mazes == Tiles.TRG) == evo_batch_size

                offspring_mazes = offspring_mazes.type(th.int64)
                offspring_mazes_onehot = th.zeros(env_gen_cfg.evo_batch_size, cfg.n_in_chan, *offspring_mazes.shape[-2:])
                offspring_mazes_onehot.scatter_(1, offspring_mazes[:,None,...], 1)

                # Get their solutions.
                offspring_target_paths = th.zeros_like(offspring_maze_walls[:])
                if cfg.task == 'pathfinding':
                    # sols = pool.map(get_target_path, [maze_discrete for maze_discrete in offspring_mazes.cpu()])
                    sols_edges_feats = [get_target_path(maze_onehot, cfg) for maze_onehot in offspring_mazes_onehot.cpu()]
                    sols, edges_o, edges_feats_o = zip(*sols_edges_feats)
                elif cfg.task == 'diameter':
                    # sols = pool.map(get_target_diam, [maze_discrete for maze_discrete in offspring_mazes.cpu()])
                    diam_tsp_edges_feats = [get_target_diam(maze_onehot, cfg) for maze_onehot in offspring_mazes_onehot.cpu()]
                    sols, _, edges_o, edges_feats_o = zip(*diam_tsp_edges_feats)
                elif cfg.task == 'traveling':
                    diam_tsp_edges_feats = [get_target_diam(maze_onehot, cfg) for maze_onehot in offspring_mazes_onehot.cpu()]
                    _, sols, edges_o, edges_feats_o = zip(*diam_tsp_edges_feats)
                else:
                    raise NotImplementedError

                # Encode the solutions as 2D binary path arrays in either case.
                for si, sol in enumerate(sols):
                    if len(sol) == 0:
                        continue
                    sol = th.Tensor(sol).long()
                    offspring_target_paths[si, sol[:, 0], sol[:, 1]] = 1

                # for mi, maze_discrete in enumerate(offspring_mazes):
                    # get_target_path(maze_discrete)

                offspring_maze_ims = render_multihot(offspring_mazes_onehot, cfg)
                offspring_env_losses = th.zeros(env_gen_cfg.evo_batch_size)

                # TODO: Get edges of evolved mazes.
                model.reset(offspring_mazes_onehot, e0=e0)
                x = hid_states[:env_gen_cfg.evo_batch_size]
                for _ in range(n_subepisodes):

                    for _ in range(cfg.loss_interval):
                        x = model(x)

                    out_paths = to_path(x)
                    batch_errs = loss_fn(out_paths, offspring_target_paths)
                    # err = (out_paths - offspring_target_paths).square()
                    offspring_env_losses += batch_errs

                offspring_env_losses /= n_subepisodes

                # Add the offspring to the population, ranked by their fitness (the loss they induce from the model).
                # old_env_losses = env_losses

                env_losses, mazes_onehot, maze_ims, target_paths = th.cat((env_losses, offspring_env_losses), dim=0), \
                    th.cat((mazes_onehot, offspring_mazes_onehot), dim=0), \
                    np.concatenate((maze_ims, offspring_maze_ims), axis=0), \
                    th.cat((target_paths, offspring_target_paths), dim=0)

                edges += edges_o
                edge_feats += edges_feats_o

                env_losses = env_losses.cpu()

                kick_idxs = th.topk(env_losses, env_gen_cfg.evo_batch_size, largest=False, sorted=False)[1]
                n_kicked = th.sum(kick_idxs < cfg.n_data)
                kick_idxs = th.sort(kick_idxs)[0]

                remain_idxs = th.arange(0, kick_idxs[0])
                for ki, k in enumerate(kick_idxs[0:-1]):
                    remain_idxs = th.cat((remain_idxs, th.arange(k+1, kick_idxs[ki+1])))
                remain_idxs = th.cat((remain_idxs, th.arange(kick_idxs[-1]+1, cfg.n_data + env_gen_cfg.evo_batch_size)))

                remain_idxs = remain_idxs.cpu()
                env_losses = env_losses[remain_idxs].to(cfg.device)
                mazes_onehot = mazes_onehot[remain_idxs]
                maze_ims = maze_ims[remain_idxs]
                target_paths = target_paths[remain_idxs]
                edges = [edges[i] for i in remain_idxs]
                edge_feats = [edge_feats[i] for i in remain_idxs]
                # env_losses, mazes_onehot, maze_ims, target_paths = env_losses[remain_idxs], mazes_onehot[remain_idxs], \
                    # maze_ims[remain_idxs], target_paths[remain_idxs]

                # ret = list(zip(env_losses, mazes_onehot, maze_ims, target_paths))
                # ret = sorted(ret, key=lambda x: x[0], reverse=True)
                # ret = ret[:cfg.n_data]
                # ret = list(zip(*ret))
                # env_losses, mazes_onehot, maze_ims, target_paths = th.stack(ret[0]), th.stack(ret[1]), \
                    # np.stack(ret[2]), th.stack(ret[3])

                path_lengths = target_paths.sum((1, 2)).float()

                tb_writer.add_scalar(f"data/n_added", n_kicked, i)
                if cfg.wandb:
                    wandb.log({f"data/n_added": n_kicked}, step=i)

                for stat in ['mean', 'min', 'max', 'std']:
                    tb_writer.add_scalar(f"data/{stat}_path-length", getattr(path_lengths, stat)(), i)
                    if cfg.wandb:
                        wandb.log({f"data/{stat}_path-length": getattr(path_lengths, stat)()}, step=i)

            elif env_gen_cfg is not None:
                last_evo += 1
                

def log(logger, lr_sched, cfg):
    fps = cfg.n_layers * cfg.minibatch_size * cfg.log_interval / (timer() - logger.last_time)
    print('step_n:', len(logger.loss_log),
        ' loss: {:.6e}'.format(logger.loss_log[-1]),
        ' fps: {:.2f}'.format(fps), 
        ' lr:', lr_sched.get_last_lr(), # end=''
        )

def vis_train(logger, render_maze_ims, render_path_ims, target_path_ims, render_batch_idx, log_dir):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplot(411)
    # smooth_loss_log = smooth(logger.loss_log, 10)
    loss_log = np.array(logger.loss_log)
    discrete_loss_log = logger.discrete_loss_log
    discrete_loss_log = np.where(np.array(discrete_loss_log) == 0, 1e-8, discrete_loss_log)
    plt.plot(loss_log, '.', alpha=0.1, label='loss')
    plt.plot(discrete_loss_log, '.', alpha=0.1, label='discrete loss')
    val_loss = logger.get_val_stat('losses')
    val_discrete_loss = logger.get_val_stat('disc_losses')
    plt.plot(list(val_loss.keys()), [v[0] for v in list(val_loss.values())], '.', alpha=1.0, label='val loss')
    plt.plot(list(val_discrete_loss.keys()), [v[0] for v in list(val_discrete_loss.values())], '.', alpha=1.0, label='val discrete loss')
    plt.legend()
    plt.yscale('log')
    # plt.ylim(np.min(np.hstack((loss_log, discrete_loss_log))), logger.loss_log[0])
    plt.ylim(np.min(np.hstack((loss_log, discrete_loss_log))), 
             np.max(np.hstack((loss_log, discrete_loss_log, [v[0] for v in list(val_loss.values())]))))
        # imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        # imshow(np.hstack(imgs))
    plt.subplot(412)
    # Remove ticks and labels.
    plt.xticks([])
    plt.yticks([])
    plt.imshow(render_maze_ims)
    plt.subplot(413)
    plt.imshow(render_path_ims)    #, vmin=-1.0, vmax=1.0)
    plt.subplot(414)
    plt.imshow(target_path_ims)
        # plt.imshow(np.hstack(x[...,-2:-1,:,:].permute([0,2,3,1]).cpu()))
        # plt.imshow(np.hstack(ca.x0[...,0:1,:,:].permute([0,2,3,1]).cpu()))
        # print(f'path activation min: {render_paths.min()}, max: {render_paths.max()}')
    plt.savefig(f'{log_dir}/training_progress.png')
    plt.close()
    logger.last_time = timer()


def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
