"""Stage 3 — odometry integration.

Per-frame planar increments (dx, dy, dtheta) in the odometry frame are summed into
cumulative motion c_t relative to the chunk's first frame. The odometry frame is
known only up to a rigid transform to the map: PadLoc removes the *rotation* part
of that gauge by rotating c_t by (pi/2 - heading_0), the heading of the odometry
frame at the chunk start, and lets the heading candidates absorb the remainder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float


def load_tum(path: str | Path) -> List[Tuple[float, float, float, float]]:
    """TUM trajectory -> [(t, x, y, yaw)] (yaw from the quaternion)."""
    out = []
    with Path(path).open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#") or len(parts) < 8:
                continue
            t, x, y, _z, qx, qy, qz, qw = (float(v) for v in parts[:8])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            out.append((t, x, y, yaw))
    return out


def align_to_frames(traj: Sequence[Tuple[float, float, float, float]], n_frames: int, *,
                    frame_period_s: Optional[float] = None, tolerance_s: float = 0.1,
                    frame_timestamps: Optional[Sequence[float]] = None) -> List[Optional[Pose2D]]:
    """Nearest-in-time trajectory pose per frame (None when no pose within tolerance).

    Frame k is at t0 + k * frame_period_s (t0 = first trajectory stamp) unless explicit
    ``frame_timestamps`` are given.
    """
    if not traj:
        return [None] * n_frames
    if frame_timestamps is not None:
        targets = list(frame_timestamps)
    elif frame_period_s is not None:
        targets = [traj[0][0] + k * float(frame_period_s) for k in range(n_frames)]
    else:
        raise ValueError("specify frame_timestamps or frame_period_s")
    ts = np.asarray([p[0] for p in traj], dtype=np.float64)
    out: List[Optional[Pose2D]] = []
    for tgt in targets:
        idx = int(np.searchsorted(ts, tgt))
        best, best_dt = None, float("inf")
        for c in ([idx] if idx < len(traj) else []) + ([idx - 1] if idx > 0 else []):
            dt = abs(ts[c] - tgt)
            if dt < best_dt:
                best, best_dt = c, dt
        out.append(Pose2D(traj[best][1], traj[best][2], traj[best][3]) if best is not None and best_dt <= tolerance_s else None)
    return out


def increments(poses: Sequence[Optional[Pose2D]]) -> List[Optional[Tuple[float, float, float]]]:
    """Per-frame (dx, dy, dtheta) between consecutive poses (None where unavailable)."""
    out: List[Optional[Tuple[float, float, float]]] = [None]
    for a, b in zip(poses[:-1], poses[1:]):
        if a is None or b is None:
            out.append(None)
        else:
            dth = (b.theta - a.theta + math.pi) % (2 * math.pi) - math.pi
            out.append((b.x - a.x, b.y - a.y, dth))
    return out


def cumulative_motion(incs: Sequence[Optional[Tuple[float, float, float]]], *,
                      cell_m: Optional[float] = None) -> List[Tuple[float, float, float]]:
    """Cumulative (x, y, theta) from frame 0 of a chunk; missing increments are zero.

    ``cell_m`` reproduces the published numerics for external (TUM) odometry, which
    was integrated in grid units: each increment is converted to cells and back.
    """
    cx = cy = ct = 0.0
    out = [(0.0, 0.0, 0.0)]
    for d in incs[1:]:
        if d is not None:
            if cell_m is not None:
                cx += (d[0] / cell_m) * cell_m
                cy += (d[1] / cell_m) * cell_m
            else:
                cx += d[0]
                cy += d[1]
            ct += d[2]
        out.append((cx, cy, ct))
    return out


def gauge_rotate(xy: Sequence[Tuple[float, float]], anchor_heading_rad: float) -> List[Tuple[float, float]]:
    """Rotate cumulative motion by (pi/2 - heading_0): the gauge-invariant motion."""
    a = math.pi / 2.0 - float(anchor_heading_rad)
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y, s * x + c * y) for x, y in xy]
