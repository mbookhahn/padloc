"""Stage 6/7 — heading selection and sparse Gaussian observations from NCC fields.

A raw masked-NCC field is indexed by the target's top-left shift. ``place_field``
re-indexes it to the map frame for a chosen reference point of the target
(centroid-indexed for the filter, anchor-indexed for cross-window summation).
Heading selection: each window field is centred by its valid-cell mean and
weighted by (N/|M|)^p; the heading whose window-sum peaks highest wins (Eq. 4).
Observations: top-K NMS peaks + finite-difference Hessian covariance (Eq. Z_w).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class WindowField:
    """One (heading, window) masked-NCC evaluation, kept in the raw (padded) frame."""
    ncc: np.ndarray           # padded correlation field
    overlap: np.ndarray       # overlap mass N(p) in the same frame
    target_shape: Tuple[int, int]
    n_target_mask: float      # |M_w| = sum of the (rotated) target mask


def place_field(field: np.ndarray, shift: Tuple[int, int], ref_shape: Tuple[int, int]) -> np.ndarray:
    """Cyclic shift then crop to the reference: map cell = target reference point."""
    return np.roll(field, shift=shift, axis=(0, 1))[: ref_shape[0], : ref_shape[1]]


def centroid_shift(target_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Shift that puts the target centre (= window odometry centroid) at the field index."""
    return (target_shape[0] // 2, target_shape[1] // 2)


def anchor_shift(target_shape: Tuple[int, int], centroid_px: Tuple[int, int], theta: float) -> Tuple[int, int]:
    """Shift that puts the chunk-start origin at the field index: the centroid offset
    (cx, cy) rotated by the heading is subtracted from the centre shift."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    cx, cy = centroid_px
    ox = c * cx - s * cy
    oy = s * cx + c * cy
    return (target_shape[0] // 2 - int(round(oy)), target_shape[1] // 2 - int(round(ox)))


def centred_weighted_field(wf: WindowField, shift: Tuple[int, int], ref_shape: Tuple[int, int], *,
                           overlap_power: float, centre: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Field for heading selection: (NCC - mean over valid cells, if ``centre``) * (N/|M|)^p."""
    im = place_field(wf.ncc, shift, ref_shape)
    valid = np.isfinite(im) & (im != 0.0)
    if not valid.any():
        return None
    if centre:
        c = np.zeros(ref_shape, dtype=np.float32)
        c[valid] = im[valid] - float(im[valid].mean())
    else:
        c = np.where(valid, im, 0.0).astype(np.float32)
    if wf.n_target_mask > 0:
        frac = np.clip(place_field(wf.overlap, shift, ref_shape) / wf.n_target_mask, 0.0, 1.0).astype(np.float32)
        if overlap_power == 0.5:
            w = np.sqrt(frac)
        elif overlap_power == 1.0:
            w = frac
        else:
            w = np.power(frac, overlap_power)
        c = c * w.astype(np.float32)
    return c, valid


@dataclass(frozen=True)
class HeadingChoice:
    theta: float
    anchor_xy_m: Tuple[float, float]   # argmax of the best heading's window sum (fallback anchor)
    per_theta_sum: Dict[float, np.ndarray]


def select_heading(fields: Dict[float, List[Optional[Tuple[np.ndarray, np.ndarray]]]], *, ref_shape: Tuple[int, int],
                   cell_to_world, fused: bool = True) -> Optional[HeadingChoice]:
    """``fields[theta]`` = per-window (field, valid) or None.

    fused: sum the windows per heading; the heading with the highest summed peak wins and
    the peak cell is the fallback anchor. Not fused (single window): the heading and cell
    of the single highest field value win.
    """
    if not fused:
        best, best_v = None, -np.inf
        for theta, per_window in fields.items():
            for fw in per_window:
                if fw is None:
                    continue
                f, v = fw
                idx = int(np.argmax(np.where(v, f, -np.inf)))
                pr, pc = divmod(idx, ref_shape[1])
                score = float(f[pr, pc])
                if score > best_v:
                    best_v, best = score, (float(theta), pr, pc)
        if best is None:
            return None
        return HeadingChoice(theta=best[0], anchor_xy_m=cell_to_world(best[1], best[2]), per_theta_sum={})
    per_theta_sum: Dict[float, np.ndarray] = {}
    per_theta_touched: Dict[float, np.ndarray] = {}
    combined: Optional[np.ndarray] = None
    for theta, per_window in fields.items():
        summed, touched = None, None
        for fw in per_window:
            if fw is None:
                continue
            f, v = fw
            summed = f.copy() if summed is None else summed + f
            touched = v.copy() if touched is None else touched | v
        if summed is None:
            continue
        per_theta_sum[float(theta)] = summed
        per_theta_touched[float(theta)] = touched
        masked = np.where(touched, summed, -np.inf)
        combined = masked.copy() if combined is None else np.maximum(combined, masked)
    if combined is None or not np.isfinite(combined).any():
        return None
    idx = int(np.argmax(np.where(np.isfinite(combined), combined, -np.inf)))
    pr, pc = divmod(idx, ref_shape[1])
    best_theta, best_v = None, -np.inf
    for theta, fld in per_theta_sum.items():
        t = per_theta_touched[theta]
        if not t[pr, pc]:
            continue
        v = float(fld[pr, pc])
        if v > best_v:
            best_v, best_theta = v, theta
    if best_theta is None:
        return None
    return HeadingChoice(theta=float(best_theta), anchor_xy_m=cell_to_world(pr, pc), per_theta_sum=per_theta_sum)


# ---------------------------------------------------------------- sparse observations
def nms_topk(surface: np.ndarray, k: int, min_sep_cells: int) -> np.ndarray:
    s = surface.copy()
    h, w = s.shape
    out = np.full((k, 3), np.nan, dtype=np.float64)
    for i in range(k):
        idx = int(np.argmax(s))
        r, c = divmod(idx, w)
        v = float(s[r, c])
        if not np.isfinite(v):
            break
        out[i] = (r, c, v)
        s[max(0, r - min_sep_cells): min(h, r + min_sep_cells + 1), max(0, c - min_sep_cells): min(w, c + min_sep_cells + 1)] = -np.inf
    return out


def hessian_covariance(surface: np.ndarray, row: int, col: int, fd: int = 1) -> Optional[np.ndarray]:
    """Negative inverse Hessian (finite differences) at a peak; None if ill-conditioned."""
    r, c = int(round(row)), int(round(col))
    h, w = surface.shape
    if r - fd < 0 or r + fd >= h or c - fd < 0 or c + fd >= w:
        return None
    f00 = surface[r, c]
    s2 = float(fd) ** 2
    Hrr = (surface[r + fd, c] - 2.0 * f00 + surface[r - fd, c]) / s2
    Hcc = (surface[r, c + fd] - 2.0 * f00 + surface[r, c - fd]) / s2
    Hrc = (surface[r + fd, c + fd] - surface[r + fd, c - fd] - surface[r - fd, c + fd] + surface[r - fd, c - fd]) / (4.0 * s2)
    H = np.array([[Hrr, Hrc], [Hrc, Hcc]], dtype=np.float64)
    if not np.all(np.isfinite(H)):
        return None
    try:
        cov = -np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    cov = 0.5 * (cov + cov.T)
    ev = np.linalg.eigvalsh(cov)
    if not np.all(np.isfinite(ev)) or float(ev.min()) <= 0.0:
        return None
    return cov


Observation = Tuple[float, np.ndarray, np.ndarray]   # (log weight, mean xy [m], cov xy [m^2])


def sparse_observations(surface: np.ndarray, *, k: int, nms_sep_cells: int, fallback_cov_cells: float, cell_m: float,
                        cell_to_world, score_scale: float = 1.0) -> List[Observation]:
    """Top-K NMS peaks of a centroid-indexed field -> softmax-weighted Gaussian observations.

    ``score_scale`` multiplies the surface before extraction (the published runs use
    stride/window, which sharpens the softmax and scales the curvature covariance).
    """
    surf = surface
    fallback_var = (fallback_cov_cells * cell_m) ** 2
    if score_scale != 1.0 and score_scale > 0.0:
        surf = (surface * float(score_scale)).astype(surface.dtype, copy=False)
        fallback_var = fallback_var / float(score_scale)
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    items: List[Observation] = []
    for row, col, score in nms_topk(surf, k, nms_sep_cells):
        if not np.isfinite(row):
            continue
        x, y = cell_to_world(row, col)
        cov_rc = hessian_covariance(surf, row, col)
        cov = (cell_m ** 2 * (P @ cov_rc @ P.T)) if cov_rc is not None else fallback_var * np.eye(2)
        items.append((float(score), np.array([x, y], dtype=np.float64), cov))
    if not items:
        return []
    log_ws = np.asarray([it[0] for it in items], dtype=np.float64)
    log_ws = log_ws - float(logsumexp(log_ws))
    return [(float(lw), m, c) for lw, (_, m, c) in zip(log_ws, items)]
