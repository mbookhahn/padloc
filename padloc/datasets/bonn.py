"""Bonn LiLocBench adapter (prepared sessions): 2D LiDAR / D455 depth columns + external odometry.

A prepared dataset dir holds ``metadata.json`` (m_per_px), ``split.yaml``, and one
directory per session with ``map.png`` (or a shared one), ``poses.txt`` and
``depth<N>.txt``. Depth sources: ``laser_full`` (811 cols, 270 deg ranges),
``laser_360`` (1080 cols), ``depth90`` (90 pinhole columns, 106 deg), ``camera_depth``
(D455 columns with fx/cx intrinsics). Odometry: a TUM trajectory aligned to frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from ..odometry import align_to_frames, increments, load_tum
from ..scan import columns_to_points, fov_column_bearings, intrinsics_column_bearings, pinhole_column_bearings
from ..windows import Frame
from .common import Scene, gt_increments, integrate, read_depth_txt, read_poses
from .gibson import Chunk, GibsonDepth, F_W


def load_split(dataset_dir: Path):
    return yaml.safe_load((Path(dataset_dir) / "split.yaml").open())


def load_scene(dataset_dir: Path, name: str) -> Scene:
    dataset_dir = Path(dataset_dir)
    root = dataset_dir / name
    map_path = root / "map.png" if (root / "map.png").is_file() else dataset_dir / "map.png"
    occ = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if occ is None:
        raise FileNotFoundError(map_path)
    cell = float(json.load((dataset_dir / "metadata.json").open())["m_per_px"])
    return Scene(name=name, root=root, occupancy=occ, native_occupancy=occ, cell_m=cell, poses=read_poses(root / "poses.txt"))


def load_depth(scene: Scene, source: str, *, camera_fx: Optional[float] = None, camera_cx: Optional[float] = None,
               camera_cols: Optional[int] = None) -> GibsonDepth:
    if source == "laser_full":
        n, bearings, radial = 811, fov_column_bearings(811, 270.0), True
    elif source == "laser_360":
        n, bearings, radial = 1080, fov_column_bearings(1080, 360.0), True
    elif source == "depth90":
        n, bearings, radial = 90, pinhole_column_bearings(90, F_W), False
    elif source == "camera_depth":
        n, bearings, radial = int(camera_cols), intrinsics_column_bearings(int(camera_cols), float(camera_fx), float(camera_cx)), False
    else:
        raise ValueError(f"unknown Bonn depth source {source!r}")
    d = read_depth_txt(scene.root / f"depth{n}.txt")
    return GibsonDepth(depths=d, bhat=[None] * len(d), bearings=bearings, radial=radial)


def iter_chunks(scene: Scene, depth: GibsonDepth, *, traj_len: int, odometry_tum: Optional[Path],
                frame_period_s: float = 1.0, tolerance_s: float = 1.0, max_chunks: Optional[int] = 200) -> Iterator[Chunk]:
    n = len(scene.poses)
    aligned = align_to_frames(load_tum(odometry_tum), n, frame_period_s=frame_period_s, tolerance_s=tolerance_s) if odometry_tum else None
    incs_vio_all = increments(aligned) if aligned is not None else None
    n_chunks = n // traj_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    for ci in range(n_chunks):
        base = ci * traj_len
        poses = scene.poses[base: base + traj_len]
        vio = incs_vio_all[base: base + traj_len] if incs_vio_all is not None else None
        cum = integrate(vio, gt_increments(poses), scene.cell_m)
        frames, pre = [], []
        for t in range(traj_len):
            d = depth.depths[base + t]
            pts = columns_to_points(d, depth.bearings, radial=depth.radial)
            cx, cy, ct = cum[t]
            frames.append(Frame(points=pts.astype(np.float32, copy=False), cum_dx_m=cx, cum_dy_m=cy, cum_dtheta_rad=ct))
            pre.append(Frame(points=np.asarray(pts, dtype=np.float64), cum_dx_m=cx, cum_dy_m=cy, cum_dtheta_rad=ct))
        # external odometry -> gauge-invariant mode (heading of the odometry frame at frame 0);
        # no odometry file -> ground-truth increments, already in the map frame (gauge None)
        if aligned is None:
            gauge = None
        else:
            gauge = float(aligned[base].theta) if aligned[base] is not None else float(poses[0].theta)
        yield Chunk(idx=ci, frames=frames, prepass_frames=pre, gauge_heading=gauge,
                    prepass_gauge_heading=float(poses[0].theta), gt_xy=[(p.x, p.y) for p in poses])
