"""Gibson (F3Loc split) adapter: 1 cm floor plans, GT poses, per-frame depth columns.

Depth sources: ``f3loc:<fname>`` (F3Loc's per-frame 40-column depth text files) and
``unloc`` (UnLoc DepthAnything-V2 per-column mu / b-hat dumped per chunk as NPZ).
Motion on Gibson is the ground-truth increment (the benchmark has no odometry).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from ..scan import columns_to_points, pinhole_column_bearings
from ..windows import Frame
from .common import Scene, gt_increments, integrate, read_depth_txt, read_poses

NATIVE_CELL_M = 0.01
F_W = 3.0 / 8.0


def load_split(dataset_dir: Path) -> Dict[str, List[str]]:
    return yaml.safe_load((Path(dataset_dir) / "split.yaml").open())


def load_scene(dataset_dir: Path, name: str, *, cell_m: float) -> Scene:
    """Read map.png (1 cm) and pool it to ``cell_m`` (any non-free pixel in a cell -> wall)."""
    root = Path(dataset_dir) / name
    native = cv2.imread(str(root / "map.png"), cv2.IMREAD_GRAYSCALE)
    if native is None:
        raise FileNotFoundError(root / "map.png")
    occ = native
    if not np.isclose(cell_m, NATIVE_CELL_M):
        scale = NATIVE_CELL_M / cell_m
        h, w = native.shape
        avg = cv2.resize(native, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        occ = np.where(avg < 255, np.uint8(0), np.uint8(255))
    return Scene(name=name, root=root, occupancy=occ, native_occupancy=native, cell_m=float(cell_m), poses=read_poses(root / "poses.txt"))


@dataclass
class GibsonDepth:
    """Per-frame depth columns + optional per-column b-hat for a whole scene."""
    depths: List[Optional[np.ndarray]]
    bhat: List[Optional[np.ndarray]]
    bearings: np.ndarray
    radial: bool = False


def load_f3loc_depth(scene: Scene, fname: str) -> GibsonDepth:
    d = read_depth_txt(scene.root / fname)
    return GibsonDepth(depths=d, bhat=[None] * len(d), bearings=pinhole_column_bearings(len(d[0]), F_W))


def load_unloc_depth(scene: Scene, dump_dir: Path, n_cols: int = 40) -> GibsonDepth:
    n = len(scene.poses)
    depths: List[Optional[np.ndarray]] = [None] * n
    bhat: List[Optional[np.ndarray]] = [None] * n
    base, k = 0, 0
    while True:
        p = Path(dump_dir) / f"{scene.name}_chunk{k:02d}.npz"
        if not p.exists():
            break
        z = np.load(p, allow_pickle=False)
        mu = np.asarray(np.asarray(z["mu_columns"], dtype=np.float32), dtype=np.float64)
        b = np.asarray(np.asarray(z["b_columns"], dtype=np.float32), dtype=np.float64)
        for i in range(mu.shape[0]):
            if base + i < n:
                depths[base + i] = mu[i].copy()
                bhat[base + i] = b[i].copy()
        base += mu.shape[0]
        k += 1
    if not any(d is not None for d in depths):
        raise FileNotFoundError(f"no UnLoc dump chunks for {scene.name} in {dump_dir}")
    return GibsonDepth(depths=depths, bhat=bhat, bearings=pinhole_column_bearings(n_cols, F_W))


@dataclass
class Chunk:
    idx: int
    frames: List[Frame]              # matcher input (b-hat columns dropped)
    prepass_frames: List[Frame]      # pre-pass input (all columns)
    gauge_heading: Optional[float]   # heading of the odometry frame at frame 0; None = motion already in the map frame
    prepass_gauge_heading: float
    gt_xy: List[Tuple[float, float]]


def iter_chunks(scene: Scene, depth: GibsonDepth, *, traj_len: int, bhat_drop_m: Optional[float],
                max_chunks: Optional[int] = None) -> Iterator[Chunk]:
    n_chunks = len(scene.poses) // traj_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    for ci in range(n_chunks):
        base = ci * traj_len
        poses = scene.poses[base: base + traj_len]
        cum = integrate(None, gt_increments(poses), scene.cell_m)
        frames, pre = [], []
        for t in range(traj_len):
            d = depth.depths[base + t]
            if d is None:
                d = np.zeros(len(depth.bearings))
            b = depth.bhat[base + t]
            pts = columns_to_points(d, depth.bearings, radial=depth.radial, bhat=b, bhat_drop_m=bhat_drop_m)
            pts_all = columns_to_points(d, depth.bearings, radial=depth.radial) if (b is not None and bhat_drop_m is not None) else pts
            cx, cy, ct = cum[t]
            # dtypes of the paper's runs (load-bearing): the matcher accumulates float32 points, the pre-pass float64
            frames.append(Frame(points=pts.astype(np.float32, copy=False), cum_dx_m=cx, cum_dy_m=cy, cum_dtheta_rad=ct))
            pre.append(Frame(points=np.asarray(pts_all, dtype=np.float64), cum_dx_m=cx, cum_dy_m=cy, cum_dtheta_rad=ct))
        yield Chunk(idx=ci, frames=frames, prepass_frames=pre, gauge_heading=float(poses[0].theta),
                    prepass_gauge_heading=float(poses[0].theta), gt_xy=[(p.x, p.y) for p in poses])
