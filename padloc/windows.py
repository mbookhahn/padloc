"""Sliding-window target accumulation.

A chunk's frames (local wall points + cumulative odometry) are grouped into W-frame
windows every S frames. Each window is rasterized ONCE in the gauge-rotated
odometry frame, centred on its odometry centroid; heading candidates are applied
afterwards by rotating that raster (``raster.rotate_grid``), because under the
gauge-invariant motion the window raster itself does not depend on the heading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import raster as _raster


@dataclass(frozen=True)
class Frame:
    """One frame: local wall points (x right, y forward; metres) + cumulative odometry."""
    points: np.ndarray          # (N, 2) float32, bearing order
    cum_dx_m: float
    cum_dy_m: float
    cum_dtheta_rad: float


@dataclass(frozen=True)
class WindowRaster:
    grid: np.ndarray            # T (side x side) float32, wall hits = 1
    mask: np.ndarray            # M (side x side) float32, wall = wall_weight, free = 1, unobserved = 0
    centroid_px: Tuple[int, int]   # odometry centroid of the window in the gauge frame (x, y) cells
    frames: Tuple[int, int]     # [start, end) frame indices in the chunk


def window_schedule(n_frames: int, window: int, stride: int) -> List[Tuple[int, int]]:
    """[(start, end)] of W-frame windows every S frames; the last window ends at n."""
    if window <= 0 or stride <= 0 or n_frames <= window:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - window + 1, stride))
    if starts and starts[-1] != n_frames - window:
        starts.append(n_frames - window)
    return [(s, s + window) for s in starts]


def build_window_raster(frames: Sequence[Frame], gauge_xy: Sequence[Tuple[float, float]], *, side: int,
                        cell_m: float, cfg, frame_range: Tuple[int, int], centred: bool = True) -> WindowRaster:
    """Max-accumulate the frames' rasters placed by their gauge-rotated odometry. ``centred``
    puts the integer centroid of the placements at the raster centre (centroid-indexed
    NCC field); otherwise the chunk-start origin sits at the centre."""
    inv = 1.0 / cell_m
    placements = [(int(round(x * inv)), int(round(y * inv))) for x, y in gauge_xy]
    if placements:
        cx = int(round(sum(p[0] for p in placements) / len(placements)))
        cy = int(round(sum(p[1] for p in placements) / len(placements)))
    else:
        cx = cy = 0
    ox, oy = (cx, cy) if centred else (0, 0)
    acc = np.zeros((side, side), dtype=np.float32)
    acc_m = np.zeros_like(acc)
    for f, (px, py) in zip(frames, placements):
        fg, fm = _raster.rasterize_points(f.points, radius_m=cfg.grid_radius_m, cell_m=cell_m, wall_weight=cfg.wall_weight,
                                          no_hit_threshold_m=cfg.no_hit_threshold_m, max_segment_jump_m=cfg.max_segment_jump_m,
                                          max_depth_ratio=cfg.max_depth_ratio, max_chord_radial_cos=cfg.max_chord_radial_cos,
                                          rotation_rad=f.cum_dtheta_rad)
        _raster.paste_max(acc, fg, py - oy, px - ox)
        _raster.paste_max(acc_m, fm, py - oy, px - ox)
    return WindowRaster(grid=acc, mask=acc_m, centroid_px=(cx, cy), frames=(int(frame_range[0]), int(frame_range[1])))
