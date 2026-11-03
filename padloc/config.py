"""PadLoc configuration — one frozen dataclass that *is* the hyperparameter table.

Values are the published defaults of the WACV 2027 paper (Gibson-selected,
then frozen). The front-end determined knobs are explicit: ``cell_m`` (map
resolution), ``accum_radius_m`` (Gibson 15, Bonn 26), ``sigma_motion_cells`` (0.5
noisy depth / 1.0 clean LiDAR), ``blur_sigma_m`` (0.24; 0.15 for the Bonn
modularity table) and the fusion window ``window``/``stride`` with the pre-pass
window ``prepass_window``/``prepass_stride`` (5/5 for per-frame depth; 30/30 when
the front-end reconstructs in 30-frame batches, as on Hilti).

Two back-end alternatives the supplement discusses stay selectable: ``ncc_d2_exact=False``
(Padfield's Algorithm 1 applied unchanged to the premasked target, D2 from M^2 T^2) and
``consensus_accum="resum"`` (every window re-summed at each read with rounded offsets).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class PadLocConfig:
    # ---- geometry (front-end / dataset determined) --------------------------------
    cell_m: float                      # map resolution (m / cell); the matching grid
    accum_radius_m: float = 15.0       # chunk accumulator half-size (m): 15 Gibson, 26 Bonn
    grid_radius_m: float = 8.0         # per-frame target raster half-size (m)
    window_radius_m: float = 11.0      # per-window (centroid-indexed) target half-size (m)

    # ---- depth -> scan ------------------------------------------------------------
    no_hit_threshold_m: Optional[float] = 10.0   # rays with slant >= this escaped (no wall)
    max_segment_jump_m: float = 1.5              # free-space wedge: max chord between rays
    max_depth_ratio: float = 3.0                 #   ... max depth ratio of a ray pair
    max_chord_radial_cos: float = 0.85           #   ... reject near-radial chords
    bhat_drop_m: Optional[float] = 0.30          # drop columns with aleatoric scale b > this

    # ---- reference --------------------------------------------------------------
    blur_sigma_m: float = 0.24         # distance-field blur of reference walls (m)
    wall_weight: float = 55.0          # target mask weight of a wall hit vs free space 1
    mask_concavity: float = 2.0        # concave-hull interior mask
    mask_length_threshold_px: float = 10.0
    mask_max_hull_points: int = 8000
    crop_padding_cells: int = 8

    # ---- heading pre-pass (Stage 4) -------------------------------------------------
    prepass_window: int = 5            # frames accumulated per pre-pass scan
    prepass_stride: int = 5            # pre-pass cadence (frames)
    hough_threshold: int = 25          # chunk-scan Hough (permissive: accumulated target)
    hough_min_line_px: float = 10.0
    hough_max_gap_px: float = 5.0
    hough_density_gamma: float = 0.5   # down-weight segments in dense clusters
    hough_density_radius_px: float = 15.0
    ref_hough_threshold: int = 50      # reference-map Hough (native resolution, no density)
    ref_hough_min_line_px: float = 20.0
    ref_hough_max_gap_px: float = 5.0
    hist_bins: int = 180
    hist_smoothing_deg: float = 2.0
    xcorr_peak_threshold: float = 0.05
    xcorr_max_peaks: int = 8
    xcorr_min_peak_sep_deg: float = 15.0
    tracker_ema: float = 0.6           # von-Mises EMA heading tracker
    tracker_forget: float = 0.6
    tracker_kappa_process: float = 50.0
    mirror_expansion: bool = True      # add the 180-degree counterpart of every candidate

    # ---- windows + matching (Stage 5/6) -------------------------------------------
    window: int = 5                    # fusion window W (frames)
    stride: int = 5                    # fusion stride S (frames)
    min_overlap: float = 0.7           # masked-NCC overlap rejection
    ncc_d2_exact: bool = True          # D2 from the weighted second moment M T^2 (False: M^2 T^2)
    overlap_power: float = 0.70        # heading selection weight (N/|M|)^p

    # ---- sparse observations + GM-BF (Stage 7/8) -----------------------------------
    top_k: int = 10                    # NMS peaks per window = mixture cap K_max
    nms_sep_cells: int = 1
    fallback_cov_cells: float = 5.0    # isotropic covariance when curvature is ill-conditioned
    sigma_motion_cells: float = 0.5    # per-frame motion-noise scale (cells); 1.0 clean LiDAR
    gate_d2: float = 16.0              # squared-Mahalanobis gate on the MAP component
    heal_m: float = 0.3                # consensus flip-heal radius (m)
    consensus_accum: str = "running"   # anchor-space running sum | "resum" (re-sum, rounded offsets)
    lag_L: int = -1                    # streaming lag in windows; -1 = offline (whole chunk)
    back_windows: int = -1             # bounded past horizon; -1 = unbounded
    emit_at_midframe: bool = True      # streamed pose reported at the window mid-frame
    seed: int = 0

    def with_(self, **kw) -> "PadLocConfig":
        return replace(self, **kw)


GIBSON = dict(accum_radius_m=15.0, sigma_motion_cells=0.5, blur_sigma_m=0.24)
BONN_LIDAR = dict(accum_radius_m=26.0, sigma_motion_cells=1.0, blur_sigma_m=0.15)
BONN_RGBD = dict(accum_radius_m=26.0, sigma_motion_cells=0.5, blur_sigma_m=0.15)
BONN_STRUCTURAL = dict(accum_radius_m=26.0, sigma_motion_cells=0.5, blur_sigma_m=0.24)
BATCH30 = dict(window=30, stride=30, prepass_window=30, prepass_stride=30)   # 30-frame batch front-ends (Hilti)
