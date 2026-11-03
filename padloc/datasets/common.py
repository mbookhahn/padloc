from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..odometry import Pose2D


@dataclass
class Scene:
    name: str
    root: Path
    occupancy: np.ndarray            # raster at the matching resolution (0 = wall)
    native_occupancy: np.ndarray     # raster at the plan's native resolution (for the Hough histogram)
    cell_m: float
    poses: List[Pose2D]              # ground truth, map metres (origin = raster centre)


def read_poses(path: Path) -> List[Pose2D]:
    out = []
    with Path(path).open() as f:
        for line in f:
            parts = line.split()
            if parts:
                out.append(Pose2D(float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def read_depth_txt(path: Path) -> List[np.ndarray]:
    out = []
    with Path(path).open() as f:
        for line in f:
            vals = [float(x) for x in line.split() if x]
            if vals:
                out.append(np.asarray(vals, dtype=np.float64))
    return out


def gt_increments(poses: Sequence[Pose2D]) -> List[Optional[Tuple[float, float, float]]]:
    out: List[Optional[Tuple[float, float, float]]] = [None]
    for a, b in zip(poses[:-1], poses[1:]):
        out.append((b.x - a.x, b.y - a.y, (b.theta - a.theta + math.pi) % (2 * math.pi) - math.pi))
    return out


def integrate(incs_vio: Optional[Sequence[Optional[Tuple[float, float, float]]]], incs_gt: Sequence[Optional[Tuple[float, float, float]]],
              cell_m: float) -> List[Tuple[float, float, float]]:
    """Cumulative motion from frame 0: external odometry where available (integrated in grid
    units, the published numerics), ground-truth increments otherwise."""
    cx = cy = ct = 0.0
    out = [(0.0, 0.0, 0.0)]
    for t in range(1, len(incs_gt)):
        v = incs_vio[t] if incs_vio is not None else None
        if v is not None:
            cx += (v[0] / cell_m) * cell_m
            cy += (v[1] / cell_m) * cell_m
            ct += v[2]
        elif incs_gt[t] is not None:
            cx += incs_gt[t][0]
            cy += incs_gt[t][1]
            ct += incs_gt[t][2]
        out.append((cx, cy, ct))
    return out
