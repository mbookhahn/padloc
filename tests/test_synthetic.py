"""Self-contained smoke test: a synthetic two-room plan, ray-cast depth along a
trajectory, noise-free odometry -> PadLoc must localize the chunk end within 1 m.
Runs in seconds on CPU; needs no fixtures."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padloc import Frame, PadLoc, PadLocConfig, Reference   # noqa: E402
from padloc.scan import columns_to_points, pinhole_column_bearings   # noqa: E402


def toy_plan(cell_m=0.10):
    """18 x 12 m plan: outer walls, a partition with a door, a closet. 0 = wall, 255 = free."""
    H, W = int(12 / cell_m), int(18 / cell_m)
    occ = np.full((H, W), 255, np.uint8)
    occ[0, :] = occ[-1, :] = 0
    occ[:, 0] = occ[:, -1] = 0
    occ[:, W // 2] = 0                      # partition
    occ[H // 2 - 6: H // 2 + 6, W // 2] = 255   # door
    occ[H // 4, W // 4: W // 4 + 30] = 0    # a short wall stub
    occ[H // 4: H // 4 + 25, W // 4 + 30] = 0
    return occ


def raycast(occ, cell_m, x, y, theta, bearings, max_r=10.0):
    """Perpendicular (pinhole-z) depth per bearing from world pose (x, y, theta).
    Positive bearing = right of the heading (the scan-frame convention: x right, y forward)."""
    H, W = occ.shape
    out = np.zeros(len(bearings))
    for i, b in enumerate(bearings):
        a = theta - b
        for r in np.arange(0.05, max_r, cell_m / 2):
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            c, rr = int(round(px / cell_m + W / 2)), int(round(py / cell_m + H / 2))
            if not (0 <= rr < H and 0 <= c < W) or occ[rr, c] == 0:
                out[i] = r * math.cos(b)   # perpendicular depth
                break
    return out


def test_localizes_synthetic_chunk():
    cell = 0.10
    occ = toy_plan(cell)
    cfg = PadLocConfig(cell_m=cell, accum_radius_m=12.0)
    ref = Reference.from_config(occ, cfg)
    loc = PadLoc(ref, cfg)
    bearings = pinhole_column_bearings(40)
    # trajectory: walk across the left room and through the door, heading ~ +x
    T = 35
    poses = [(-6.5 + 0.25 * t, -2.0 + 0.05 * t, 0.15 * math.sin(t / 6.0)) for t in range(T)]
    frames = []
    for t, (x, y, th) in enumerate(poses):
        d = raycast(occ, cell, x, y, th, bearings)
        pts = columns_to_points(d, bearings, radial=False)
        x0, y0, th0 = poses[0]
        # odometry = ground-truth increments expressed in the frame-0 odometry frame
        # (the sensor's own frame: the pipeline removes the gauge by the frame-0 heading)
        frames.append(Frame(points=pts.astype(np.float32), cum_dx_m=x - x0, cum_dy_m=y - y0, cum_dtheta_rad=th - th0))
    res = loc.run_chunk(frames, gauge_heading=poses[0][2])
    gx, gy, _ = poses[-1]
    px, py = res.poses_xy_m[-1]
    err = math.hypot(px - gx, py - gy)
    assert err < 1.0, f"chunk-end error {err:.2f} m (theta {math.degrees(res.theta):.1f} deg)"


if __name__ == "__main__":
    test_localizes_synthetic_chunk()
    print("synthetic localization OK")
