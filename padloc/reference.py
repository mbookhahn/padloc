"""Stage 1 — reference preparation (once per floor plan).

occupancy raster -> wall mask -> concave-hull interior mask M_R -> blurred wall
distance field R -> crop to the occupied bounding box -> reference wall-angle
histogram (for the heading pre-pass) -> per-target-size FFT caches (lazy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import ncc as _ncc
from .heading import angle_histogram_from_walls

try:  # the published interior mask uses a concave hull of the wall pixels
    from concave_hull import concave_hull as _concave_hull
except ImportError as _e:  # pragma: no cover
    _concave_hull = None


def detect_walls(occupancy: np.ndarray) -> np.ndarray:
    """Boolean wall mask from a raster whose convention may be 0=wall or >0=wall."""
    raw = occupancy.astype(np.float32)
    if raw.max() > 1.0:
        zero_count = int((raw == 0).sum())
        return raw == 0 if zero_count <= raw.size - zero_count else raw > 0
    return raw > 0


def distance_field(wall: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian-of-distance blur of a binary wall raster (0 outside 3 sigma)."""
    occupied = np.abs(wall) > 0
    if not occupied.any():
        return np.zeros_like(wall, dtype=np.float32)
    if sigma_cells <= 0:
        return occupied.astype(np.float32)
    d2 = distance_transform_edt(~occupied) ** 2
    out = np.where(d2 <= (3.0 * sigma_cells) ** 2, np.exp(-d2 / (2.0 * sigma_cells * sigma_cells)), 0.0)
    return out.astype(np.float32)


def interior_mask_concave_hull(walls: np.ndarray, *, concavity: float, length_threshold_px: float,
                               max_points: int) -> np.ndarray:
    if _concave_hull is None:
        raise ImportError("the `concave-hull` package is required for the interior mask")
    coords = np.column_stack(np.where(walls))
    if coords.size == 0:
        return np.zeros(walls.shape, dtype=np.float32)
    if len(coords) > max_points:
        coords = coords[np.random.default_rng(0).choice(len(coords), max_points, replace=False)]
    hull = np.asarray(_concave_hull(coords.astype(np.float64), concavity=float(concavity),
                                    length_threshold=float(length_threshold_px)), dtype=np.int32)
    if hull.shape[0] < 3:
        return np.zeros(walls.shape, dtype=np.float32)
    mask = np.zeros(walls.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.fliplr(hull)], 1)
    return mask.astype(np.float32)


def crop_to_mask(ref: np.ndarray, mask: np.ndarray, padding: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    interior = mask > 0
    if not interior.any():
        return ref, mask, (0, 0)
    r_idx = np.where(np.any(interior, axis=1))[0]
    c_idx = np.where(np.any(interior, axis=0))[0]
    r0, r1 = max(0, int(r_idx[0]) - padding), min(ref.shape[0], int(r_idx[-1]) + 1 + padding)
    c0, c1 = max(0, int(c_idx[0]) - padding), min(ref.shape[1], int(c_idx[-1]) + 1 + padding)
    return ref[r0:r1, c0:c1].copy(), mask[r0:r1, c0:c1].copy(), (r0, c0)


@dataclass
class Reference:
    """Prepared floor plan: blurred wall field + interior mask, cropped, with FFT caches."""
    field: np.ndarray            # R  (cropped, float32)
    mask: np.ndarray             # M_R (cropped, float32 in {0,1})
    crop_r0: int
    crop_c0: int
    orig_h: int
    orig_w: int
    cell_m: float
    angle_hist: np.ndarray       # reference wall-angle histogram (mod pi)
    walls_full: np.ndarray       # binary walls of the uncropped raster (for diagnostics/viz)
    _fft: Dict[Tuple[int, int], object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_occupancy(cls, occupancy: np.ndarray, *, cell_m: float, blur_sigma_m: float,
                       native_occupancy: Optional[np.ndarray] = None,
                       mask_concavity: float = 2.0, mask_length_threshold_px: float = 10.0,
                       mask_max_hull_points: int = 8000, crop_padding_cells: int = 8,
                       ref_hough_threshold: int = 50, ref_hough_min_line_px: float = 20.0,
                       ref_hough_max_gap_px: float = 5.0, hist_bins: int = 180,
                       hist_smoothing_deg: float = 2.0) -> "Reference":
        """``native_occupancy``: the plan at its native (finer) resolution, used only for
        the reference wall-angle histogram (Hough at native scale); defaults to ``occupancy``."""
        walls = detect_walls(occupancy)
        ref_full = distance_field(walls.astype(np.float32), blur_sigma_m / cell_m)
        mask_full = interior_mask_concave_hull(walls, concavity=mask_concavity,
                                               length_threshold_px=mask_length_threshold_px,
                                               max_points=mask_max_hull_points)
        ref, mask, (r0, c0) = crop_to_mask(ref_full, mask_full, crop_padding_cells)
        hough_walls = detect_walls(native_occupancy if native_occupancy is not None else occupancy)
        hist = angle_histogram_from_walls(hough_walls.astype(np.uint8) * 255, n_bins=hist_bins,
                                          hough_threshold=ref_hough_threshold,
                                          min_line_length_px=ref_hough_min_line_px,
                                          max_line_gap_px=ref_hough_max_gap_px,
                                          smoothing_sigma_deg=hist_smoothing_deg)
        return cls(field=ref, mask=mask, crop_r0=int(r0), crop_c0=int(c0),
                   orig_h=int(occupancy.shape[0]), orig_w=int(occupancy.shape[1]),
                   cell_m=float(cell_m), angle_hist=hist, walls_full=walls)

    @classmethod
    def from_config(cls, occupancy: np.ndarray, cfg, native_occupancy: Optional[np.ndarray] = None) -> "Reference":
        return cls.from_occupancy(
            occupancy, cell_m=cfg.cell_m, blur_sigma_m=cfg.blur_sigma_m, native_occupancy=native_occupancy,
            mask_concavity=cfg.mask_concavity, mask_length_threshold_px=cfg.mask_length_threshold_px,
            mask_max_hull_points=cfg.mask_max_hull_points, crop_padding_cells=cfg.crop_padding_cells,
            ref_hough_threshold=cfg.ref_hough_threshold, ref_hough_min_line_px=cfg.ref_hough_min_line_px,
            ref_hough_max_gap_px=cfg.ref_hough_max_gap_px, hist_bins=cfg.hist_bins,
            hist_smoothing_deg=cfg.hist_smoothing_deg)

    # -- geometry helpers ------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return self.field.shape

    def cell_to_world(self, row: float, col: float) -> Tuple[float, float]:
        """Cropped-reference cell -> world metres (map origin at the raster centre)."""
        return ((col + self.crop_c0 - self.orig_w / 2.0) * self.cell_m,
                (row + self.crop_r0 - self.orig_h / 2.0) * self.cell_m)

    def world_to_cell(self, x_m: float, y_m: float) -> Tuple[float, float]:
        return (y_m / self.cell_m + self.orig_h / 2.0 - self.crop_r0,
                x_m / self.cell_m + self.orig_w / 2.0 - self.crop_c0)

    # -- FFT cache per target size -------------------------------------------------
    def fft_cache(self, target_shape: Tuple[int, int]):
        key = (int(target_shape[0]), int(target_shape[1]))
        if key not in self._fft:
            self._fft[key] = _ncc.precompute_reference_cache(self.field, self.mask, key)
        return self._fft[key]
