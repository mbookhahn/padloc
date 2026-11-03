"""PadLoc — the published pipeline as one object.

    ref = Reference.from_config(occupancy, cfg)
    loc = PadLoc(ref, cfg)
    result = loc.run_chunk(frames, prepass_frames, gauge_heading, prepass_gauge_heading)

``frames`` carry each frame's local wall points and cumulative odometry; the
result holds the heading, the chunk anchor, per-frame poses and (optionally) the
bounded-lag streaming poses and the frozen readout built from them. One chunk =
one cold start (honest per-chunk reset).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import consensus as _cons
from . import heading as _heading
from . import ncc as _ncc
from . import observe as _obs
from . import raster as _raster
from . import windows as _win
from .config import PadLocConfig
from .gmbf import WindowFilter
from .odometry import gauge_rotate
from .reference import Reference
from .windows import Frame


@dataclass
class ChunkResult:
    theta: float                                   # selected heading (rad, map frame)
    anchor_xy_m: Tuple[float, float]               # frame-0 position in map metres
    poses_xy_m: List[Tuple[float, float]]          # per-frame positions (anchor + map-frame odometry)
    cum_motion_map_m: List[Tuple[float, float]]    # map-frame cumulative odometry under theta
    windows: List[Tuple[int, int]] = field(default_factory=list)      # kept windows [start, end)
    window_xhat_m: List[Tuple[float, float]] = field(default_factory=list)   # GM-BF MAP per window
    stream_xy_m: List[Optional[Tuple[float, float]]] = field(default_factory=list)  # bounded-lag poses
    stream_conf: List[float] = field(default_factory=list)            # PSR of each streamed pick
    stream_frame: List[int] = field(default_factory=list)             # frame each streamed pose refers to
    stream_anchor_xy_m: List[Optional[Tuple[float, float]]] = field(default_factory=list)  # frame-0 anchor per window
    stream_lag_L: int = -1                                            # lag the stream was decoded with
    consensus_score: Optional[float] = None
    consensus_psr: Optional[float] = None

    def frozen_poses_xy_m(self) -> List[Tuple[float, float]]:
        """Frozen readout (needs ``run_chunk(..., stream=True)``): frame t reports the anchor of
        the latest window emitted by t, plus the map-frame odometry to t, and is never revised.
        With lag L a window's anchor is emitted when window w + L ends (L < 0: at the chunk end);
        frames before the first emission take the first emission. A window whose read failed
        the coverage gate reports NaN."""
        n = len(self.cum_motion_map_m)
        if not self.windows:
            return list(self.poses_xy_m)
        if len(self.stream_anchor_xy_m) != len(self.windows):
            raise ValueError("frozen readout needs the stream: run_chunk(..., stream=True)")
        last = len(self.windows) - 1
        emit = [n - 1 if self.stream_lag_L < 0 else self.windows[min(k + self.stream_lag_L, last)][1] - 1
                for k in range(len(self.windows))]
        out = []
        for t in range(n):
            avail = [k for k in range(len(self.windows)) if emit[k] <= t]
            a = self.stream_anchor_xy_m[max(avail) if avail else 0]
            dx, dy = self.cum_motion_map_m[t]
            out.append((float("nan"), float("nan")) if a is None else (a[0] + dx, a[1] + dy))
        return out


class PadLoc:
    def __init__(self, reference: Reference, cfg: PadLocConfig, *, batched_ncc: bool = False) -> None:
        """``batched_ncc``: evaluate all heading candidates of a window in one batched FFT
        call (GPU backend; last-ulp rounding may differ from the per-call path)."""
        self.ref = reference
        self.cfg = cfg
        self.batched_ncc = bool(batched_ncc)
        self.accum_side = _raster.grid_side(cfg.accum_radius_m, cfg.cell_m)
        self.win_side = _raster.grid_side(cfg.window_radius_m, cfg.cell_m)

    # ------------------------------------------------------------------ helpers
    def _cum_motion_map(self, frames: Sequence[Frame], theta: float, gauge_heading: Optional[float]) -> List[Tuple[float, float]]:
        """Map-frame cumulative motion under heading ``theta``. With a gauge heading the
        odometry is rotated by theta + pi/2 - heading_0; with ``None`` the odometry is
        already in the map frame (e.g. ground-truth increments) and used as is."""
        if gauge_heading is None:
            return [(float(f.cum_dx_m), float(f.cum_dy_m)) for f in frames]
        phi = float(theta) + math.pi / 2.0 - float(gauge_heading)
        c, s = math.cos(phi), math.sin(phi)
        return [(c * f.cum_dx_m - s * f.cum_dy_m, s * f.cum_dx_m + c * f.cum_dy_m) for f in frames]

    @staticmethod
    def _placement_xy(frames: Sequence[Frame], theta: float, gauge_heading: Optional[float]) -> List[Tuple[float, float]]:
        """Frame placements in the window raster frame. Gauge mode: theta-independent rotation
        by pi/2 - heading_0. Map-frame mode: the map-frame motion rotated by -theta (the raster
        is rotated by +theta afterwards)."""
        if gauge_heading is not None:
            return gauge_rotate([(f.cum_dx_m, f.cum_dy_m) for f in frames], gauge_heading)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        return [(c * f.cum_dx_m + s * f.cum_dy_m, -s * f.cum_dx_m + c * f.cum_dy_m) for f in frames]

    def _ncc(self, grid: np.ndarray, mask: np.ndarray, theta: float, side: int) -> Optional[_obs.WindowField]:
        rg = _raster.rotate_grid(grid, theta)
        rm = _raster.rotate_grid(mask, theta)
        if not np.any(rm):
            return None
        ncc, ov = _ncc.masked_ncc(self.ref.fft_cache((side, side)), rg, rm, min_overlap=self.cfg.min_overlap, return_overlap=True,
                                  d2_exact=self.cfg.ncc_d2_exact)
        return _obs.WindowField(ncc=ncc, overlap=ov, target_shape=rg.shape, n_target_mask=float(np.asarray(rm).sum()))

    def _ncc_all(self, rasters: Sequence[_win.WindowRaster], thetas: Sequence[float], side: int) -> Dict[float, List[Optional[_obs.WindowField]]]:
        """All (heading, window) fields; batched over headings per window when enabled."""
        if not self.batched_ncc:
            return {th: [self._ncc(w.grid, w.mask, th, side) for w in rasters] for th in thetas}
        out: Dict[float, List[Optional[_obs.WindowField]]] = {th: [] for th in thetas}
        cache = self.ref.fft_cache((side, side))
        for w in rasters:
            rot = [(_raster.rotate_grid(w.grid, th), _raster.rotate_grid(w.mask, th)) for th in thetas]
            live = [i for i, (_, rm) in enumerate(rot) if np.any(rm)]
            res = _ncc.masked_ncc_batched(cache, [rot[i][0] for i in live], [rot[i][1] for i in live],
                                          min_overlap=self.cfg.min_overlap, return_overlap=True,
                                          d2_exact=self.cfg.ncc_d2_exact) if live else []
            by_i = {i: r for i, r in zip(live, res)}
            for i, th in enumerate(thetas):
                if i in by_i:
                    ncc, ov = by_i[i]
                    out[th].append(_obs.WindowField(ncc=ncc, overlap=ov, target_shape=rot[i][0].shape, n_target_mask=float(np.asarray(rot[i][1]).sum())))
                else:
                    out[th].append(None)
        return out

    # ------------------------------------------------------------------ main entry
    def run_chunk(self, frames: Sequence[Frame], *, gauge_heading: Optional[float], prepass_frames: Optional[Sequence[Frame]] = None,
                  prepass_gauge_heading: Optional[float] = None, stream: bool = False) -> ChunkResult:
        """``gauge_heading``: heading of the odometry frame at frame 0 (gauge-invariant mode),
        or ``None`` when the frames' cumulative motion is already expressed in the map frame."""
        cfg, ref = self.cfg, self.ref
        n = len(frames)
        ref_shape = ref.shape
        # --- heading candidates (causal pre-pass over the chunk)
        pf = prepass_frames if prepass_frames is not None else frames
        pg = prepass_gauge_heading if prepass_gauge_heading is not None else (gauge_heading if gauge_heading is not None else 0.0)
        peaks = _heading.chunk_heading_candidates(pf, gauge_rotate([(f.cum_dx_m, f.cum_dy_m) for f in pf], pg), ref.angle_hist, cfg=cfg, side=self.accum_side)
        if not peaks:   # no pre-pass window saw enough walls: chunk-end Hough over all frames
            peaks = _heading.chunk_end_candidates(frames, ref.angle_hist, cfg=cfg, side=self.accum_side)
        thetas = _heading.expand_candidates(peaks, mirror=cfg.mirror_expansion) if peaks else list(np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False))
        fused = n > cfg.window and cfg.stride > 0 and cfg.window > 0
        sched = _win.window_schedule(n, cfg.window, cfg.stride) if fused else [(max(0, n - cfg.window) if cfg.window > 0 else 0, n)]
        side = self.win_side if fused else self.accum_side

        def _rasters_for(theta: float) -> List[_win.WindowRaster]:
            xy = self._placement_xy(frames, theta, gauge_heading)
            return [_win.build_window_raster(frames[s:e], xy[s:e], side=side, cell_m=cfg.cell_m, cfg=cfg,
                                             frame_range=(s, e) if fused else (0, n), centred=fused) for s, e in sched]

        # --- window rasters: built once in gauge mode (theta-independent), per heading in map-frame mode
        if gauge_heading is not None:
            shared = _rasters_for(0.0)
            rasters_by_theta = {th: shared for th in thetas}
            fields = self._ncc_all(shared, thetas, side)
        else:
            rasters_by_theta = {th: _rasters_for(th) for th in thetas}
            fields = {th: [self._ncc(w.grid, w.mask, th, side) for w in rasters_by_theta[th]] for th in thetas}
        # --- heading selection on anchor-indexed, centred, overlap-weighted fields
        def _sel_shift(wf, w, th):
            return _obs.anchor_shift(wf.target_shape, w.centroid_px, th) if fused else _obs.centroid_shift(wf.target_shape)
        sel_fields = {th: [None if wf is None else _obs.centred_weighted_field(wf, _sel_shift(wf, w, th), ref_shape,
                                                                             overlap_power=cfg.overlap_power, centre=fused)
                           for wf, w in zip(fl, rasters_by_theta[th])] for th, fl in fields.items()}
        choice = _obs.select_heading(sel_fields, ref_shape=ref_shape, cell_to_world=ref.cell_to_world, fused=fused)
        if choice is None:   # no valid field anywhere: report the chunk-start at the map centre
            cum = self._cum_motion_map(frames, 0.0, gauge_heading)
            return ChunkResult(theta=0.0, anchor_xy_m=(0.0, 0.0), poses_xy_m=[(x, y) for x, y in cum], cum_motion_map_m=cum)
        theta = choice.theta
        cum = self._cum_motion_map(frames, theta, gauge_heading)
        anchor = choice.anchor_xy_m
        if not fused:
            return ChunkResult(theta=theta, anchor_xy_m=anchor, poses_xy_m=[(anchor[0] + dx, anchor[1] + dy) for dx, dy in cum], cum_motion_map_m=cum)
        # --- moving-pose GM-BF over the centroid-indexed fields of the selected heading
        filt = WindowFilter(sigma_motion_cells=cfg.sigma_motion_cells, cell_m=cfg.cell_m, stride=cfg.stride, n_frames=n, gate_d2=cfg.gate_d2, k_max=cfg.top_k)
        score_scale = float(cfg.stride) / float(cfg.window)
        xhat: List[Tuple[float, float]] = []
        cens: List[Tuple[float, float]] = []
        cfields: List[_cons.Field] = []
        kept: List[Tuple[int, int]] = []
        for wf, w in zip(fields[theta], rasters_by_theta[theta]):
            if wf is None:
                continue
            imc = _obs.place_field(wf.ncc, _obs.centroid_shift(wf.target_shape), ref_shape)
            vc = np.isfinite(imc) & (imc != 0.0)
            if not vc.any():
                continue
            surf = np.where(vc, imc, float(imc[vc].min())).astype(np.float32)
            s, e = w.frames
            seg = cum[s:e]
            cv = (sum(p[0] for p in seg) / max(len(seg), 1), sum(p[1] for p in seg) / max(len(seg), 1))
            obs = _obs.sparse_observations(surf, k=cfg.top_k, nms_sep_cells=cfg.nms_sep_cells, fallback_cov_cells=cfg.fallback_cov_cells,
                                           cell_m=cfg.cell_m, cell_to_world=ref.cell_to_world, score_scale=score_scale)
            fr, fc = divmod(int(np.argmax(surf)), surf.shape[1])
            xhat.append(filt.step(cv, obs, fallback_xy=ref.cell_to_world(fr, fc)))
            cens.append((float(cv[0]), float(cv[1])))
            cfields.append((imc, vc))
            kept.append((s, e))
        if not xhat:
            return ChunkResult(theta=theta, anchor_xy_m=anchor, poses_xy_m=[(anchor[0] + dx, anchor[1] + dy) for dx, dy in cum], cum_motion_map_m=cum)
        # --- consensus: heal, shift-and-sum, eligibility; anchor = peak - last centroid
        xh_roll = _cons.heal(xhat, cens, cfg.heal_m)
        pick = _cons.consensus_peak(xh_roll, cfields, cell_m=cfg.cell_m, ref_shape=ref_shape, to_cell=ref.world_to_cell,
                                    accum=cfg.consensus_accum)
        score = psr = None
        if pick is not None:
            px, py = ref.cell_to_world(pick[0], pick[1])
            anchor = (px - cens[-1][0], py - cens[-1][1])
            score, psr = pick[2], pick[3]
        else:   # nothing eligible: fall back to the filter's last MAP
            anchor = (xhat[-1][0] - cens[-1][0], xhat[-1][1] - cens[-1][1])
        res = ChunkResult(theta=theta, anchor_xy_m=anchor, poses_xy_m=[(anchor[0] + dx, anchor[1] + dy) for dx, dy in cum],
                          cum_motion_map_m=cum, windows=kept, window_xhat_m=xhat, consensus_score=score, consensus_psr=psr)
        if stream:
            picks = _cons.streaming_peaks(xh_roll, cfields, lag_L=cfg.lag_L, back_windows=cfg.back_windows, cell_m=cfg.cell_m,
                                          ref_shape=ref_shape, to_cell=ref.world_to_cell, accum=cfg.consensus_accum)
            res.stream_lag_L = cfg.lag_L
            for (s, e), pk, cv in zip(kept, picks, cens):
                mid = (s + e) // 2
                tgt = cum[min(mid, n - 1)] if cfg.emit_at_midframe else cv
                if pk is None:
                    res.stream_xy_m.append(None)
                    res.stream_anchor_xy_m.append(None)
                    res.stream_conf.append(float("nan"))
                else:
                    px, py = ref.cell_to_world(pk[0], pk[1])
                    res.stream_xy_m.append((px - cv[0] + tgt[0], py - cv[1] + tgt[1]))
                    res.stream_anchor_xy_m.append((px - cv[0], py - cv[1]))
                    res.stream_conf.append(float(pk[3]))
                res.stream_frame.append(int(mid))
        return res
