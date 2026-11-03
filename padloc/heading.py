"""Stage 4 — heading candidates from wall-angle histograms.

Walls in a floor plan take few orientations. A wall-angle histogram (mod pi, length
weighted Hough segments) of the accumulated scan, cross-correlated with the
reference histogram, yields the gauge rotations that align scan walls to plan walls.
A von-Mises EMA tracker smooths the per-window candidates over the chunk; mirror
expansion adds the 180-degree counterpart of each surviving candidate. If no window
yields a candidate, a chunk-end Hough over the whole chunk does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from . import raster as _raster


# ------------------------------------------------------------------ histograms
def angle_histogram_from_walls(walls_uint8: np.ndarray, *, n_bins: int = 180, hough_threshold: int = 50,
                               min_line_length_px: float = 20.0, max_line_gap_px: float = 5.0,
                               smoothing_sigma_deg: float = 2.0, density_gamma: float = 0.0,
                               density_radius_px: float = 15.0) -> np.ndarray:
    """Length-weighted wall-angle histogram (mod pi) of the Hough segments of a wall raster."""
    lines = cv2.HoughLinesP(walls_uint8, rho=1.0, theta=np.pi / 180.0, threshold=int(hough_threshold),
                            minLineLength=float(min_line_length_px), maxLineGap=float(max_line_gap_px))
    hist = np.zeros(n_bins, dtype=np.float64)
    if lines is None or len(lines) == 0:
        return hist
    coords = lines.reshape(-1, 4).astype(np.float64)
    dx, dy = coords[:, 2] - coords[:, 0], coords[:, 3] - coords[:, 1]
    lengths = np.hypot(dx, dy)
    bins = ((np.arctan2(dy, dx) % np.pi) / np.pi * n_bins).astype(np.int64) % n_bins
    votes = lengths
    if density_gamma > 0.0 and coords.shape[0] > 1:
        mids = np.stack([(coords[:, 0] + coords[:, 2]) * 0.5, (coords[:, 1] + coords[:, 3]) * 0.5], axis=1)
        cnt = np.array([len(nb) for nb in cKDTree(mids).query_ball_point(mids, r=float(density_radius_px))], dtype=np.float64)
        votes = votes * np.power(np.maximum(cnt, 1.0), -float(density_gamma))
    np.add.at(hist, bins, votes)
    if smoothing_sigma_deg > 0:
        hist = gaussian_filter1d(hist, sigma=smoothing_sigma_deg / (180.0 / n_bins), mode="wrap")
    return hist


def _xcorr_peak_kappa(corr: np.ndarray, idx: int) -> float:
    n = int(corr.shape[0])
    peak = float(corr[idx])
    if peak <= 0:
        return 0.0
    half = 0.5 * peak

    def hwhm(direction: int) -> float:
        prev = peak
        for j in range(1, n // 2 + 1):
            v = float(corr[(idx + direction * j) % n])
            if v <= half:
                return j - 1 + (prev - half) / max(prev - v, 1e-09)
            prev = v
        return float(n // 2)

    hwhm_bins = 0.5 * (hwhm(1) + hwhm(-1))
    if hwhm_bins <= 0:
        return 0.0
    return float((2.0 * np.sqrt(2.0 * np.log(2.0)) / max(2.0 * hwhm_bins * (np.pi / n), 1e-06)) ** 2)


def rotation_candidates(ref_hist: np.ndarray, tgt_hist: np.ndarray, *, peak_threshold: float, max_peaks: int,
                        min_peak_separation_deg: float = 15.0) -> List[Tuple[float, float, float]]:
    """Circular cross-correlation peaks -> [(rotation_rad in [0, pi), strength, kappa)]."""
    n = int(ref_hist.shape[0])
    if tgt_hist.shape[0] != n:
        raise ValueError("ref and tgt histograms must have the same length")
    if ref_hist.sum() <= 0 or tgt_hist.sum() <= 0:
        return []
    corr = np.fft.irfft(np.conj(np.fft.rfft(ref_hist)) * np.fft.rfft(tgt_hist), n=n)
    peak_mask = (corr >= np.roll(corr, 1)) & (corr > np.roll(corr, -1)) & (corr > 0)
    peak_idxs = np.where(peak_mask)[0]
    if peak_idxs.size == 0:
        return []
    max_c = float(corr.max())
    if max_c <= 0:
        return []
    thresh = peak_threshold * max_c
    min_sep = max(1, int(round(min_peak_separation_deg / (180.0 / n))))
    selected: List[int] = []
    for idx in peak_idxs[np.argsort(-corr[peak_idxs])]:
        if corr[idx] < thresh:
            continue
        if any(min(abs(int(idx) - s) % n, n - abs(int(idx) - s) % n) < min_sep for s in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= max_peaks:
            break
    return [((-float(k) / float(n) * np.pi) % np.pi, float(corr[k] / max_c), _xcorr_peak_kappa(corr, k)) for k in selected]


# ------------------------------------------------------------------ tracker
def _circ_dist_pi(a: float, b: float) -> float:
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


@dataclass
class _VMComponent:
    C: float
    S: float
    W: float
    kappa: float

    @property
    def mu(self) -> float:
        return 0.5 * math.atan2(self.S, self.C) % math.pi


class HeadingTracker:
    """Von-Mises mixture over the doubled wall angle, updated by EMA'd histograms.

    ``update(hist)`` folds a new window histogram into the EMA target histogram,
    re-derives cross-correlation candidates, and assigns them to components (gate
    8 degrees) with forgetting; ``candidates()`` returns (mu, weight, kappa) sorted
    by weight.
    """

    def __init__(self, ref_hist: np.ndarray, *, hist_ema: float, forget: float, peak_threshold: float,
                 max_peaks: int, gate_deg: float = 8.0, prune_weight_frac: float = 0.05) -> None:
        self.ref_hist = np.asarray(ref_hist, dtype=np.float64)
        self.hist_ema, self.forget = float(hist_ema), float(forget)
        self.gate = math.radians(gate_deg)
        self.peak_threshold, self.max_peaks = float(peak_threshold), int(max_peaks)
        self.prune_weight_frac = float(prune_weight_frac)
        self.H: Optional[np.ndarray] = None
        self.components: List[_VMComponent] = []

    def update(self, frame_hist: np.ndarray) -> None:
        fh = np.asarray(frame_hist, dtype=np.float64)
        self.H = fh.copy() if self.H is None else self.hist_ema * self.H + (1.0 - self.hist_ema) * fh
        if self.ref_hist.sum() <= 0 or self.H.sum() <= 0:
            return
        obs = rotation_candidates(self.ref_hist, self.H, peak_threshold=self.peak_threshold, max_peaks=self.max_peaks)
        for c in self.components:
            c.C *= self.forget
            c.S *= self.forget
            c.W *= self.forget
        for rot, strength, kappa in obs:
            rot = rot % math.pi
            w = (1.0 - self.forget) * float(strength)
            best, best_d = None, self.gate
            for c in self.components:
                d = _circ_dist_pi(rot, c.mu)
                if d < best_d:
                    best_d, best = d, c
            if best is not None:
                best.C += w * math.cos(2.0 * rot)
                best.S += w * math.sin(2.0 * rot)
                best.W += w
                best.kappa = 0.7 * best.kappa + 0.3 * float(kappa)
            else:
                self.components.append(_VMComponent(C=w * math.cos(2.0 * rot), S=w * math.sin(2.0 * rot), W=w, kappa=float(kappa)))
        self._merge_and_prune()

    def _merge_and_prune(self) -> None:
        merged: List[_VMComponent] = []
        for c in sorted(self.components, key=lambda x: -x.W):
            hit = next((m for m in merged if _circ_dist_pi(c.mu, m.mu) < self.gate), None)
            if hit is not None:
                hit.C += c.C
                hit.S += c.S
                hit.W += c.W
                hit.kappa = max(hit.kappa, c.kappa)
            else:
                merged.append(c)
        if merged:
            wmax = max(m.W for m in merged) or 1.0
            merged = [m for m in merged if m.W >= self.prune_weight_frac * wmax]
        self.components = merged

    def candidates(self) -> List[Tuple[float, float, float]]:
        if not self.components:
            return []
        wmax = max(c.W for c in self.components) or 1.0
        return sorted(((c.mu, c.W / wmax, c.kappa) for c in self.components), key=lambda t: -t[1])


# ------------------------------------------------------------------ chunk pre-pass
def window_wall_raster(frames: Sequence, gauge_xy: Sequence[Tuple[float, float]], *, side: int, cell_m: float,
                       grid_radius_m: float, wall_weight: float, no_hit_threshold_m: Optional[float],
                       max_segment_jump_m: float, max_depth_ratio: float, max_chord_radial_cos: float) -> np.ndarray:
    """Binary (uint8 0/255) wall raster of several frames placed by their gauge-rotated motion."""
    acc = np.zeros((side, side), dtype=np.float32)
    tx0, ty0 = gauge_xy[0]
    inv = 1.0 / cell_m
    for f, (tx, ty) in zip(frames, gauge_xy):
        fg, _ = _raster.rasterize_points(f.points, radius_m=grid_radius_m, cell_m=cell_m, wall_weight=wall_weight,
                                         no_hit_threshold_m=no_hit_threshold_m, max_segment_jump_m=max_segment_jump_m,
                                         max_depth_ratio=max_depth_ratio, max_chord_radial_cos=max_chord_radial_cos,
                                         rotation_rad=f.cum_dtheta_rad)
        _raster.paste_max(acc, fg, int(round((ty - ty0) * inv)), int(round((tx - tx0) * inv)))
    return (acc > 0).astype(np.uint8) * 255


def chunk_heading_candidates(frames: Sequence, gauge_xy: Sequence[Tuple[float, float]], ref_hist: np.ndarray, *,
                             cfg, side: int) -> List[Tuple[float, float, float]]:
    """Causal pre-pass over a chunk: every ``prepass_stride`` frames, accumulate a
    ``prepass_window``-frame wall raster, histogram it, update the tracker.
    Returns the tracker's candidates [(rotation_rad, weight, kappa)]."""
    n = len(frames)
    W, S = int(cfg.prepass_window), int(cfg.prepass_stride)
    if 0 < W < n:
        starts = list(range(0, n, S if S > 0 else W))
    else:
        starts, W = [0], n
    tracker = HeadingTracker(ref_hist, hist_ema=cfg.tracker_ema, forget=cfg.tracker_forget,
                             peak_threshold=cfg.xcorr_peak_threshold, max_peaks=cfg.xcorr_max_peaks)
    for s in starts:
        e = min(s + W, n)
        walls = window_wall_raster(frames[s:e], gauge_xy[s:e], side=side, cell_m=cfg.cell_m,
                                   grid_radius_m=cfg.grid_radius_m, wall_weight=cfg.wall_weight,
                                   no_hit_threshold_m=cfg.no_hit_threshold_m, max_segment_jump_m=cfg.max_segment_jump_m,
                                   max_depth_ratio=cfg.max_depth_ratio, max_chord_radial_cos=cfg.max_chord_radial_cos)
        hist = angle_histogram_from_walls(walls, n_bins=cfg.hist_bins, hough_threshold=cfg.hough_threshold,
                                          min_line_length_px=cfg.hough_min_line_px, max_line_gap_px=cfg.hough_max_gap_px,
                                          smoothing_sigma_deg=cfg.hist_smoothing_deg, density_gamma=cfg.hough_density_gamma,
                                          density_radius_px=cfg.hough_density_radius_px)
        tracker.update(hist)
    return tracker.candidates()


def chunk_end_candidates(frames: Sequence, ref_hist: np.ndarray, *, cfg, side: int) -> List[Tuple[float, float, float]]:
    """Fallback when the pre-pass yields no candidate (short windows can see too few walls):
    one wall raster of the whole chunk, frames placed by their raw odometry, histogrammed
    with the reference-map Hough settings (no density weighting), cross-correlated with the
    plan histogram. Returns [(rotation_rad, strength, kappa)]."""
    acc = np.zeros((side, side), dtype=np.float32)
    inv = 1.0 / cfg.cell_m
    for f in frames:
        fg, _ = _raster.rasterize_points(f.points, radius_m=cfg.grid_radius_m, cell_m=cfg.cell_m, wall_weight=cfg.wall_weight,
                                         no_hit_threshold_m=cfg.no_hit_threshold_m, max_segment_jump_m=cfg.max_segment_jump_m,
                                         max_depth_ratio=cfg.max_depth_ratio, max_chord_radial_cos=cfg.max_chord_radial_cos,
                                         rotation_rad=f.cum_dtheta_rad)
        _raster.paste_max(acc, fg, int(round(f.cum_dy_m * inv)), int(round(f.cum_dx_m * inv)))
    hist = angle_histogram_from_walls((acc > 0).astype(np.uint8) * 255, n_bins=cfg.hist_bins,
                                      hough_threshold=cfg.ref_hough_threshold, min_line_length_px=cfg.ref_hough_min_line_px,
                                      max_line_gap_px=cfg.ref_hough_max_gap_px, smoothing_sigma_deg=cfg.hist_smoothing_deg)
    return rotation_candidates(ref_hist, hist, peak_threshold=cfg.xcorr_peak_threshold, max_peaks=cfg.xcorr_max_peaks,
                               min_peak_separation_deg=cfg.xcorr_min_peak_sep_deg)


def expand_candidates(peaks: Sequence[Tuple[float, float, float]], *, mirror: bool = True) -> List[float]:
    """Heading set Theta: each candidate plus its 180-degree counterpart (mirror) ..."""
    k_count, k_step = (2, math.pi) if mirror else (4, math.pi / 2.0)
    return [(float(p[0]) + k * k_step) % (2.0 * math.pi) for p in peaks for k in range(k_count)]
