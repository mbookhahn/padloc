# Paper → `PadLocConfig`

Where each parameter of the paper (main paper Sec. 3, supplement Table S1 "Principal
hyperparameters") lives in `PadLocConfig`, with the value every reproduced number uses.

## Fixed across all datasets (Table S1, top)

| paper | `PadLocConfig` field | value |
|---|---|---|
| wall-hit mask weight w_wall (Sec. 3.2) | `wall_weight` | 55 |
| b̂ column-drop threshold (Sec. 3.2) | `bhat_drop_m` | 0.30 m (front-ends that expose b̂; `None` disables) |
| overlap-rejection threshold (Sec. 3.3) | `min_overlap` | 0.7 |
| heading-overlap exponent (Eq. 4) | `overlap_power` | 0.70 |
| top-K peaks / mixture cap K_max (Sec. 3.3, 3.4) | `top_k` | 10 |
| Mahalanobis gate, squared (Eq. 6) | `gate_d2` | 16 |
| heal threshold τ (Eq. 7) | `heal_m` | 0.3 m |
| consensus eligibility (Eq. 9) | fixed in `consensus.py` | C(p) ≥ max(1, round(N/2)) |
| pre-pass scan / cadence (Sec. 3.2) | `prepass_window` / `prepass_stride` | 5 / 5 frames |
| heading-tracker EMA (Sec. 3.2) | `tracker_ema` (forgetting: `tracker_forget`) | 0.6 (0.6) |

## Front-end-determined (Table S1, bottom)

| paper | `PadLocConfig` field | value |
|---|---|---|
| fusion window W/S (Sec. 3.4) | `window` / `stride` | 5 / 5; 30 / 30 for 30-frame batch front-ends (`BATCH30`, also sets the pre-pass) |
| motion-noise scale σ (Eq. 5) | `sigma_motion_cells` | 0.5; 1.0 clean LiDAR |
| reference blur σ_blur (Sec. 3.2, supplement A) | `blur_sigma_m` | 0.24 m learned depth; 0.15 m range sensors (LiDAR, D455) |
| cell size δ | `cell_m` | 0.10 m Gibson; 0.05 m Bonn (native map) |

The dataset presets in `padloc.config` bundle these per front-end: `GIBSON`,
`BONN_LIDAR`, `BONN_RGBD`, `BONN_STRUCTURAL` (they also set the chunk accumulator size
`accum_radius_m`, 15 m Gibson, 26 m Bonn).

## Method switches (Sec. 3)

| paper | field | value |
|---|---|---|
| exact weighted second moment M T² in D₂ (Eq. 2) | `ncc_d2_exact` | True (`False`: M² T², Padfield's Algorithm 1 applied unchanged; supplement A) |
| mirror expansion of heading candidates (Sec. 3.2) | `mirror_expansion` | True |
| running consensus sums (Sec. 3.4) | `consensus_accum` | `"running"` (`"resum"`: re-sum with rounded offsets; supplement P) |
| lag L / past horizon (Sec. 3.4) | `lag_L` / `back_windows` | 0 = causal stream (frozen readout); −1 = offline; past unbounded (−1) |
| pose reported at the window's mid-frame (Sec. 3.4) | `emit_at_midframe` | True |
| isotropic covariance fallback (Sec. 3.3) | `fallback_cov_cells` | 5 cells |

## Implementation constants (not tabulated in the paper)

| field | value | role |
|---|---|---|
| `grid_radius_m` / `window_radius_m` | 8 / 11 m | per-frame and per-window target raster half-size |
| `no_hit_threshold_m` | 10 m | rays at or beyond this slant are treated as escaped (no wall) |
| `max_segment_jump_m` / `max_depth_ratio` / `max_chord_radial_cos` | 1.5 m / 3.0 / 0.85 | free-space wedge between neighbouring rays |
| `hough_threshold` / `hough_min_line_px` / `hough_max_gap_px` | 25 / 10 / 5 | target Hough (accumulated scans) |
| `hough_density_gamma` / `hough_density_radius_px` | 0.5 / 15 px | down-weights segments in dense clusters |
| `ref_hough_threshold` / `ref_hough_min_line_px` / `ref_hough_max_gap_px` | 50 / 20 / 5 | reference-map Hough |
| `hist_bins` / `hist_smoothing_deg` | 180 / 2° | wall-angle histogram |
| `xcorr_peak_threshold` / `xcorr_max_peaks` / `xcorr_min_peak_sep_deg` | 0.05 / 8 / 15° | histogram cross-correlation peaks |
| `tracker_kappa_process` | 50 | von-Mises process concentration |
| `nms_sep_cells` | 1 | NMS separation of the top-K peaks |
| `mask_concavity` / `mask_length_threshold_px` / `mask_max_hull_points` | 2.0 / 10 / 8000 | concave-hull interior mask M_R |
| `crop_padding_cells` | 8 | reference crop padding |

Fixed by construction (not knobs): accumulated window targets, the moving-pose GM-BF,
the consensus sum over centred fields, gauge-invariant odometry, uniform window weights,
composite tempering α = S/W.
