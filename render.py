"""
NCA renderer — fixed & simplified

Key fixes
---------
1) Removed broken VideoWriter that referenced undefined FFMPEG_VideoWriter/mvp.
   Replaced with imageio for both GIF and MP4.
2) Deterministic dtype/scale handling: all frames are uint8 [0..255].
3) Robust grid tiling with optional maze preview tile and 1‑px gutters.
4) Safer normalization (handles constant arrays), consistent 3‑channel RGB.
5) Headless-friendly: OpenCV windows only if cfg.headless is False.
6) Matplotlib animation path simplified and corrected (scoped state, no re‑imshow).
7) Removed unused/buggy code paths (e.g., stray path_chan edits on mazes_discrete).
8) Feature‑flags are function arguments; avoids globals maze.
9) "oracle_out" support is capability-based (hasattr) instead of string compare.

How to use
----------
from nca_renderer_fixed import render_trained
render_trained(model, maze_data, cfg, n_render_chans=None, save_gif=True, save_mp4=False,
               render_type="cv2", render_name="", render_border=False)

This will write PNG frames under cfg.log_dir/renderer{render_name}/ and, if enabled,
create cfg.log_dir/behavior{render_name}.gif (or .mp4).
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import torch as th
from PIL import Image

from mazes import render_multihot

# --- helpers -----------------------------------------------------------------

def _to_numpy(x: np.ndarray | th.Tensor) -> np.ndarray:
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    return x


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """Ensure HxW or HxWxC -> HxWx3 uint8 in RGB order, scaled properly."""
    img = _to_numpy(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    # If float in [0,1] (or outside a bit), clamp & scale; if already uint8, keep
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0 + 0.5).astype(np.uint8)
    return img


def _stack_batch_horiz(batch_hw_or_hwc: np.ndarray) -> np.ndarray:
    """Given an array shaped [B,H,W] or [B,H,W,3], hstack along W across batch."""
    batch = _to_numpy(batch_hw_or_hwc)
    assert batch.ndim in (3, 4), f"Expected [B,H,W] or [B,H,W,3], got {batch.shape}"
    tiles = [batch[i] for i in range(batch.shape[0])]
    return np.hstack(tiles)


def _make_grid(tiles: Sequence[np.ndarray], n_cols: int, gutter_px: int = 1) -> np.ndarray:
    """Tiles are HxW or HxWx3 uint8. Returns a single HxWx3 uint8 image."""
    tiles_rgb = [_to_rgb_u8(t) for t in tiles]
    H, W = tiles_rgb[0].shape[:2]
    for t in tiles_rgb:
        assert t.shape[:2] == (H, W), "All tiles must have same HxW. Pad/resize upstream if needed."
    n = len(tiles_rgb)
    n_cols = max(1, n_cols)
    n_rows = (n + n_cols - 1) // n_cols

    gutter_v = np.zeros((H, gutter_px, 3), dtype=np.uint8)
    gutter_h = np.zeros((gutter_px, (W + gutter_px) * n_cols - gutter_px, 3), dtype=np.uint8)

    rows = []
    for r in range(n_rows):
        row_tiles = []
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx < n:
                row_tiles.append(tiles_rgb[idx])
            else:
                row_tiles.append(np.zeros_like(tiles_rgb[0]))
            if c < n_cols - 1:
                row_tiles.append(gutter_v)
        row = np.concatenate(row_tiles, axis=1)
        rows.append(row)
        if r < n_rows - 1:
            rows.append(gutter_h)
    return np.concatenate(rows, axis=0)


# --- core rendering -----------------------------------------------------------

def build_frame(x: th.Tensor,
                n_render_chans: int,
                cfg,
                oracle_out: Optional[th.Tensor] = None,
                maze_imgs: Optional[np.ndarray] = None,
                render_border: bool = False) -> np.ndarray:
    """Build a grid image showing last `n_render_chans` channels (batched hstack),
    with optional maze preview as the first tile and optional oracle age channel.

    x: [B,C,H,W] torch tensor
    maze_imgs: HxW or HxWx3 image representing the maze (already batched+hstacked upstream)
    Returns: H' x W' x 3 uint8 RGB image
    """
    B, C, H, W = x.shape
    n = min(n_render_chans, C)

    # Build per-channel tiles: hstack across batch for each channel
    chan_tiles: List[np.ndarray] = []
    for i in range(n):
        ch = x[:cfg.render_minibatch_size, C - 1 - i]  # last channels
        ch = _normalize01(_to_numpy(ch))
        ch = _stack_batch_horiz(ch)  # [H, B*W]
        chan_tiles.append(ch)

    # Optional oracle channel (age/etc.) — expect shape [B, H, W]
    if oracle_out is not None:
        if isinstance(oracle_out, th.Tensor):
            oo = oracle_out[:cfg.render_minibatch_size]
            if oo.ndim == 4:
                # If [B, C, H, W], take the last channel by convention
                oo = oo[:, -1]
        else:
            oo = oracle_out
        oo = _normalize01(_to_numpy(oo))
        oo = _stack_batch_horiz(oo)
        chan_tiles.append(oo)

    tiles_rgb: List[np.ndarray] = []

    # Prepend maze image if provided
    if maze_imgs is not None:
        maze_rgb = _to_rgb_u8(maze_imgs)
        tiles_rgb.append(maze_rgb[0])

    # Convert all channel tiles to RGB
    tiles_rgb.extend([_to_rgb_u8(t) for t in chan_tiles])

    # Choose columns: square-ish grid looks good; if few channels, put all in a row
    total = len(tiles_rgb)
    if n_render_chans > 3:
        n_cols = int(math.ceil(math.sqrt(total)))
    else:
        n_cols = total

    grid = _make_grid(tiles_rgb, n_cols=n_cols, gutter_px=1)

    # Optional red border around the first channel after the maze (i.e., tile index 1)
    if render_border and maze_imgs is not None and total >= 2:
        # Compute tile size in the composed grid
        # With n_cols tiles per row and 1px gutters, tile width = W, height = H
        tile_h, tile_w = tiles_rgb[0].shape[:2]
        # Maze is tile 0; target is tile 1 in the same row if exists
        # Row/col of tile1 (index 1)
        idx = 1
        r = idx // n_cols
        c = idx % n_cols
        # top-left corner inside grid image, accounting for gutters
        y0 = r * (tile_h + 1)
        x0 = c * (tile_w + 1)
        # draw rectangle in-place (red)
        cv2.rectangle(grid, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (255, 0, 0), thickness=1)

    return grid


# --- episode rendering loops --------------------------------------------------

def _reset_episode(bi: int,
                   model,
                   mazes_onehot: th.Tensor,
                   target_paths: th.Tensor,
                   batch_idxs: th.Tensor,
                   render_minibatch_size: int,
                   cfg,
                   edges=None,
                   edge_feats=None) -> Tuple[th.Tensor, np.ndarray, int]:
    # pick a slice of the pool
    batch_idxs = batch_idxs[th.arange(render_minibatch_size) + bi].cpu()
    pool = model.seed(batch_size=render_minibatch_size, width=cfg.width, height=cfg.height)
    x = pool
    x0 = mazes_onehot[batch_idxs]

    e0 = None
    if getattr(cfg, "traversable_edges_only", False):
        e0 = [edges[i] for i in batch_idxs]
    ef = None
    if getattr(cfg, "positional_edge_features", False):
        ef = [edge_feats[i] for i in batch_idxs]

    model.reset(x0, e0=e0, edge_feats=ef, new_batch_size=True)

    # Maze preview image: user-provided helper returns a list/concat already batched
    # We expect render_multihot(arr=..., target_path=..., cfg=cfg) -> list of HxW or an array
    maze_imgs = render_multihot(arr=mazes_onehot[batch_idxs], target_path=target_paths[batch_idxs], cfg=cfg)
    if isinstance(maze_imgs, list):
        maze_imgs = np.hstack(maze_imgs)
    maze_imgs = _to_rgb_u8(maze_imgs)

    bi = (bi + render_minibatch_size) % mazes_onehot.shape[0]
    return x, maze_imgs, bi


def _render_episode_cv2(bi: int,
                        model,
                        mazes_onehot: th.Tensor,
                        target_paths: th.Tensor,
                        batch_idxs: th.Tensor,
                        n_render_chans: int,
                        render_dir: str,
                        cfg,
                        frame_i: int,
                        edges=None,
                        edge_feats=None,
                        window_name: str = "pathfinding",
                        writer: Optional[imageio.core.format.Writer] = None,
                        render_border: bool = False) -> Tuple[int, int, th.Tensor]:
    x, maze_imgs, bi = _reset_episode(bi, model, mazes_onehot, target_paths, batch_idxs,
                                      cfg.render_minibatch_size, cfg, edges, edge_feats)
    has_oracle = hasattr(model, "oracle_out")
    oracle_out = model.oracle_out if has_oracle else None

    frame = build_frame(x, n_render_chans, cfg, oracle_out=oracle_out, maze_imgs=maze_imgs,
                        render_border=render_border)

    if not cfg.headless:
        imS = cv2.resize(frame[..., ::-1], tuple(map(int, getattr(cfg, "cv2_im_size", (1500, 1500)))))
        cv2.imshow(window_name, imS)
        cv2.waitKey(int(getattr(cfg, "cv2_wait_key_time", 1)))

    # save png
    png_path = os.path.join(render_dir, f"render_{frame_i}.png")
    Image.fromarray(frame).save(png_path)
    if writer is not None:
        writer.append_data(frame)

    # rollout
    for _ in range(cfg.n_layers):
        frame_i += 1
        x = model.forward(x)
        oracle_out = model.oracle_out if has_oracle else None
        frame = build_frame(x, n_render_chans, cfg, oracle_out=oracle_out, maze_imgs=maze_imgs,
                            render_border=render_border)

        if not cfg.headless:
            imS = cv2.resize(frame[..., ::-1], tuple(map(int, getattr(cfg, "cv2_im_size", (1500, 1500)))))
            cv2.imshow(window_name, imS)
            cv2.waitKey(int(getattr(cfg, "cv2_wait_key_time", 1)))

        Image.fromarray(frame).save(os.path.join(render_dir, f"render_{frame_i}.png"))
        if writer is not None:
            writer.append_data(frame)

    return bi, frame_i, x


# --- public API ---------------------------------------------------------------

def render_trained(model,
                   maze_data,
                   cfg,
                   n_render_chans: Optional[int] = None,
                   save_gif: bool = True,
                   save_mp4: bool = False,
                   render_type: str = "cv2",  # or "matplotlib"
                   render_name: str = "",
                   n_render_episodes: Optional[int] = 10,
                   render_border: bool = False) -> None:
    """Generate frames and optional GIF/MP4 of the model behavior.

    - Uses imageio for writing (no undefined FFMPEG_VideoWriter/mvp).
    - cv2 window shown only if cfg.headless == False.
    """
    model.eval()

    mazes_onehot = maze_data.mazes_onehot[None]
    target_paths = maze_data.target_paths if getattr(cfg, "task", "") != "traveling" else maze_data.target_travelings

    if n_render_chans is None:
        n_render_chans = int(getattr(model, "n_out_chan", getattr(model, "n_channels", 1)))

    render_minibatch_size = min(int(getattr(cfg, "render_minibatch_size", 1)), int(mazes_onehot.shape[0]))

    # Choose order: longest target path first
    path_lengths = target_paths.sum((1, 2))
    batch_idxs = path_lengths.sort(descending=True)[1]

    bi = 0
    if n_render_episodes is None:
        n_render_episodes = int(batch_idxs.numel())

    # Paths
    render_dir = os.path.join(cfg.log_dir, f"renderer{render_name}")
    if os.path.exists(render_dir):
        shutil.rmtree(render_dir)
    os.makedirs(render_dir, exist_ok=True)

    gif_writer = None
    mp4_writer = None
    try:
        if save_gif:
            gif_path = os.path.join(cfg.log_dir, f"behavior{render_name}.gif")
            # duration per frame (s). Use cfg if available else 0.1
            duration = float(getattr(cfg, "gif_frame_duration", 0.1))
            gif_writer = imageio.get_writer(gif_path, mode="I", duration=duration)
        if save_mp4:
            mp4_path = os.path.join(cfg.log_dir, f"behavior{render_name}.mp4")
            fps = float(getattr(cfg, "mp4_fps", 10.0))
            mp4_writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")

        if render_type == "cv2":
            writer = mp4_writer or gif_writer  # append frames to one writer if present
            for _ in range(n_render_episodes):
                bi, frame_i, _ = _render_episode_cv2(
                    bi, model, maze_data.mazes_onehot, target_paths, batch_idxs,
                    n_render_chans, render_dir, cfg, frame_i=0,
                    edges=getattr(maze_data, "edges", None),
                    edge_feats=getattr(maze_data, "edge_feats", None),
                    window_name="pathfinding",
                    writer=writer,
                    render_border=render_border,
                )
        elif render_type == "matplotlib":
            _render_with_matplotlib(model, maze_data, cfg, n_render_chans, render_dir, render_name)
        else:
            raise ValueError(f"Unknown render_type: {render_type}")
    finally:
        if gif_writer is not None:
            gif_writer.close()
        if mp4_writer is not None:
            mp4_writer.close()


# --- optional: matplotlib animation (single episode) --------------------------

def _render_with_matplotlib(model,
                            maze_data,
                            cfg,
                            n_render_chans: int,
                            render_dir: str,
                            render_name: str) -> None:
    """Single-episode Matplotlib animation that avoids re-imshow calls.
    Writes MP4 to cfg.log_dir/path_nca_anim{render_name}.mp4
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    model.eval()

    # Setup one episode worth of frames
    mazes_onehot = maze_data.mazes_onehot
    target_paths = maze_data.target_paths if getattr(cfg, "task", "") != "traveling" else maze_data.target_travelings
    path_lengths = target_paths.sum((1, 2))
    batch_idxs = path_lengths.sort(descending=True)[1]

    bi = 0
    x, maze_imgs, bi = _reset_episode(bi, model, mazes_onehot, target_paths, batch_idxs, cfg.render_minibatch_size, cfg,
                                      getattr(maze_data, "edges", None), getattr(maze_data, "edge_feats", None))
    has_oracle = hasattr(model, "oracle_out")

    frames: List[np.ndarray] = []
    oracle_outs: List[Optional[th.Tensor]] = []
    frames.append(build_frame(x, n_render_chans, cfg, oracle_out=model.oracle_out if has_oracle else None,
                              maze_imgs=maze_imgs))
    oracle_outs.append(model.oracle_out if has_oracle else None)

    for _ in range(cfg.n_layers):
        x = model.forward(x)
        frames.append(build_frame(x, n_render_chans, cfg, oracle_out=model.oracle_out if has_oracle else None,
                                  maze_imgs=maze_imgs))
        oracle_outs.append(model.oracle_out if has_oracle else None)

    # Matplotlib animation
    plt.axis('off')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    mat = ax.matshow(frames[0])
    ax.axis('off')

    def animate(i):
        mat.set_array(frames[i])
        return (mat,)

    anim = animation.FuncAnimation(fig, animate, interval=50, frames=len(frames), blit=True, save_count=len(frames))

    out_path = os.path.join(cfg.log_dir, f'path_nca_anim{render_name}.mp4')
    anim.save(out_path, fps=float(getattr(cfg, "mp4_fps", 10.0)), extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
