"""Stage 2 — depth columns -> local wall scan (points in the sensor frame).

A frame's depth is a vector over columns with known bearings. Perpendicular depth
(pinhole z) is converted to slant range along the bearing; LiDAR ranges are used
as-is. Columns with an aleatoric scale b > ``bhat_drop_m`` are dropped before
rasterization (membership, not weighting — see the paper). Points are returned as
(x right, y forward) in metres, in bearing order (consecutive rays).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def pinhole_column_bearings(n_cols: int, F_W: float = 3.0 / 8.0) -> np.ndarray:
    """Bearings of W pinhole columns with focal/width ratio F_W (F3Loc convention)."""
    half = (n_cols - 1) / 2.0
    return np.arctan((np.arange(n_cols, dtype=np.float64) - half) / (n_cols * F_W))


def intrinsics_column_bearings(n_cols: int, fx: float, cx: float) -> np.ndarray:
    """Bearings of camera columns from fx/cx (positive = left, matching the prep)."""
    return np.arctan((cx - np.arange(int(n_cols), dtype=np.float64)) / fx)


def fov_column_bearings(n_cols: int, fov_deg: float) -> np.ndarray:
    half = np.radians(fov_deg) / 2.0
    return np.linspace(-half, half, int(n_cols))


def drop_columns(depth: np.ndarray, score: np.ndarray, threshold: float) -> np.ndarray:
    """Set columns whose score (e.g. b-hat) exceeds ``threshold`` to NaN (dropped)."""
    d = np.asarray(depth, dtype=np.float64).copy()
    s = np.asarray(score, dtype=np.float64)
    n = min(d.shape[0], s.shape[0])
    valid = np.zeros(d.shape[0], dtype=bool)
    valid[:n] = np.isfinite(d[:n]) & (d[:n] > 0) & np.isfinite(s[:n])
    if valid.any():
        d[valid & (s > float(threshold))] = np.nan
    return d


def columns_to_points(depth: np.ndarray, bearings: np.ndarray, *, radial: bool,
                      bhat: Optional[np.ndarray] = None, bhat_drop_m: Optional[float] = None) -> np.ndarray:
    """Depth columns (+ optional per-column b-hat) -> (N, 2) local points, bearing order."""
    d = np.asarray(depth, dtype=np.float64)
    if bhat is not None and bhat_drop_m is not None:
        d = drop_columns(d, bhat, bhat_drop_m)
    ang = np.asarray(bearings, dtype=np.float64)
    if ang.shape[0] != d.shape[0]:
        raise ValueError(f"bearings length {ang.shape[0]} != depth length {d.shape[0]}")
    rays = d if radial else d / np.cos(ang)
    rays = np.nan_to_num(np.where(np.isfinite(rays), rays, np.nan), nan=0.0)
    valid = np.isfinite(rays) & (rays > 0)
    return np.stack([rays[valid] * np.sin(ang[valid]), rays[valid] * np.cos(ang[valid])], axis=1)
