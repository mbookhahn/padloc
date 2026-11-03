"""Target rasterization — local wall points -> (T, M) raster pair.

T (the target) has 1 at wall-hit cells; M (the mask) has ``wall_weight`` at wall
hits, 1 on observed free space (the wedge between consecutive rays, or the ray
itself when the pair is not a plausible surface chord) and 0 where nothing was
observed. Unobserved zeros therefore drop out of the masked NCC while observed-free
zeros penalize reference walls.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def grid_side(radius_m: float, cell_m: float) -> int:
    """Odd side length of a square raster covering +-radius_m."""
    side = int(np.ceil(2 * radius_m / cell_m))
    return side + 1 if side % 2 == 0 else side


def _draw_ray(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, value: float = 1.0) -> None:
    """Bresenham line (endpoint excluded) into ``mask``."""
    h, w = mask.shape
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err, r, c = dc - dr, r0, c0
    while True:
        at_end = r == r1 and c == c1
        if not at_end and 0 <= r < h and 0 <= c < w:
            mask[r, c] = value
        if at_end:
            return
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def rasterize_points(points_xy: np.ndarray, *, radius_m: float, cell_m: float, wall_weight: float,
                     no_hit_threshold_m: Optional[float], max_segment_jump_m: float,
                     max_depth_ratio: float, max_chord_radial_cos: float,
                     rotation_rad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize one frame's local wall points (sensor at the centre, x right / y down).

    ``points_xy`` are in the sensor frame, ordered by bearing (consecutive rays).
    Rays whose slant >= ``no_hit_threshold_m`` escaped without a wall: they carve no
    free space and splat no wall. Returns (grid, mask) float32 of equal odd size.
    """
    side = grid_side(radius_m, cell_m)
    grid = np.zeros((side, side), dtype=np.float32)
    mask = np.zeros((side, side), dtype=np.float32)
    cx = side // 2
    # NOTE: arithmetic is done in the points' own dtype (float32 for accumulated frames,
    # float64 for the pre-pass) — cell rounding must match the published numerics. The
    # slants that gate no-hit rays and ray pairs come from the UNROTATED points: the
    # rotated float32 slant can cross the 10 m no-hit threshold or a ratio gate.
    pts = np.asarray(points_xy).reshape(-1, 2)
    n = len(pts)
    slants = np.hypot(pts[:, 0], pts[:, 1]).astype(np.float64) if n else np.empty((0,), dtype=np.float64)
    if rotation_rad != 0.0 and pts.size:
        c, s = float(np.cos(rotation_rad)), float(np.sin(rotation_rad))
        pts = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], axis=1)
    no_hit = (slants >= float(no_hit_threshold_m)) if (no_hit_threshold_m is not None and n) \
        else np.zeros(n, dtype=bool)
    rr = np.round(cx + pts[:, 1] / cell_m).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
    cc = np.round(cx + pts[:, 0] / cell_m).astype(np.int64) if n else np.zeros(0, dtype=np.int64)

    # wall hits
    splat = ~no_hit & (rr >= 0) & (rr < side) & (cc >= 0) & (cc < side)
    if splat.any():
        np.maximum.at(grid, (rr[splat], cc[splat]), 1.0)

    # observed free space: wedge between plausible ray pairs, single ray otherwise
    if n >= 2:
        r0, c0, r1, c1 = rr[:-1], cc[:-1], rr[1:], cc[1:]
        dpx = np.hypot((r1 - r0).astype(np.float64), (c1 - c0).astype(np.float64)) * cell_m
        s_i, s_j = slants[:-1], slants[1:]
        ratio = np.maximum(s_i, s_j) / np.maximum(1e-06, np.minimum(s_i, s_j))
        dr_ch, dc_ch = (r1 - r0).astype(np.float64), (c1 - c0).astype(np.float64)
        chord = np.hypot(dr_ch, dc_ch)
        mid_r, mid_c = 0.5 * (r0 + r1) - cx, 0.5 * (c0 + c1) - cx
        mid_len = np.hypot(mid_r, mid_c)
        cos_radial = np.where((chord < 1e-06) | (mid_len < 1e-06), 0.0,
                              np.abs(dr_ch * mid_r + dc_ch * mid_c) / np.maximum(chord * mid_len, 1e-12))
        bad = (dpx > max_segment_jump_m) | (ratio > max_depth_ratio) | \
              (cos_radial > max_chord_radial_cos) | no_hit[:-1] | no_hit[1:]
        for i in np.where(~bad)[0]:
            tri = np.array([[cx, cx], [int(cc[i]), int(rr[i])], [int(cc[i + 1]), int(rr[i + 1])]], dtype=np.int32)
            cv2.fillPoly(mask, [tri], 1.0)
        for i in np.where(bad)[0]:
            if not no_hit[i]:
                _draw_ray(mask, cx, cx, int(rr[i]), int(cc[i]))
            if not no_hit[i + 1]:
                _draw_ray(mask, cx, cx, int(rr[i + 1]), int(cc[i + 1]))
    else:
        for i in range(n):
            if not no_hit[i]:
                _draw_ray(mask, cx, cx, int(rr[i]), int(cc[i]))

    wall = grid > 0
    mask[wall] = float(wall_weight) * grid[wall]
    mask[cx, cx] = max(mask[cx, cx], 1.0)
    return grid, mask


def rotate_grid(grid: np.ndarray, theta_rad: float) -> np.ndarray:
    """Nearest-neighbour rotation of a square raster about its centre (zero fill)."""
    h, w = grid.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    yy, xx = np.indices((h, w), dtype=np.float32)
    yy, xx = yy - cy, xx - cx
    src_x = np.round(c * xx + s * yy + cx).astype(np.int32)
    src_y = np.round(-s * xx + c * yy + cy).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(grid)
    out[valid] = grid[src_y[valid], src_x[valid]]
    return out


def paste_max(acc: np.ndarray, tile: np.ndarray, row_off: int, col_off: int) -> None:
    """acc = max(acc, tile) with the tile's centre placed at (row_off, col_off) from acc's centre."""
    ha, wa = acc.shape
    ht, wt = tile.shape
    pr = ha // 2 + row_off - ht // 2
    pc = wa // 2 + col_off - wt // 2
    r0, r1 = max(0, pr), min(ha, pr + ht)
    c0, c1 = max(0, pc), min(wa, pc + wt)
    if r1 <= r0 or c1 <= c0:
        return
    sr0, sc0 = r0 - pr, c0 - pc
    np.maximum(acc[r0:r1, c0:c1], tile[sr0:sr0 + (r1 - r0), sc0:sc0 + (c1 - c0)], out=acc[r0:r1, c0:c1])
