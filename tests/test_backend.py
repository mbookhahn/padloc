"""Unit tests of the paper's back-end: exact weighted masked NCC, running-sum
consensus, 5/5 windows, the frozen readout, the chunk-end heading fallback and the
rasterizer's no-hit gate. Fixture-free; seconds on CPU."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padloc import ChunkResult, PadLoc, PadLocConfig, Reference   # noqa: E402
from padloc import consensus as cons   # noqa: E402
from padloc import ncc   # noqa: E402
from padloc.config import BATCH30   # noqa: E402
from padloc.windows import window_schedule   # noqa: E402


# ------------------------------------------------------------------ exact weighted masked NCC
def _ncc_problem(seed=0, wall_weight=55.0):
    rng = np.random.default_rng(seed)
    mr = np.zeros((40, 48), np.float32)
    mr[3:37, 4:44] = 1.0
    r = (rng.random((40, 48)) * mr).astype(np.float32)        # premasked reference
    t = (rng.random((12, 14)) < 0.25).astype(np.float32)      # binary wall hits
    m = (rng.random((12, 14)) < 0.8).astype(np.float32)       # observed free space
    m[t > 0] = wall_weight
    t *= m > 0
    return r, mr, t, m


def _direct(r, mr, t, m, i, j, d2_exact):
    """The weighted masked NCC at shift (i, j), target top-left on reference cell (i, j)."""
    h, w = t.shape
    R, MR = r[i:i + h, j:j + w].astype(np.float64), mr[i:i + h, j:j + w].astype(np.float64)
    T, M = t.astype(np.float64), m.astype(np.float64)
    Tt = M * T
    N = (MR * M).sum() + 0.001
    rm, mt = (R * M).sum(), (MR * Tt).sum()
    num = (R * Tt).sum() - rm * mt / N
    d1 = (R * R * M).sum() - rm ** 2 / N
    d2 = (MR * (M * T * T if d2_exact else Tt * Tt)).sum() - mt ** 2 / N
    return num / math.sqrt(d1 * d2)


@pytest.mark.parametrize("d2_exact", [True, False])
def test_ncc_matches_direct_formula(d2_exact):
    r, mr, t, m = _ncc_problem()
    cache = ncc._cache_numpy(r, mr, t.shape)
    field = ncc._ncc_numpy(cache, t, m, min_overlap=0.1, return_overlap=False, d2_exact=d2_exact)
    for i, j in [(3, 4), (10, 17), (20, 30), (25, 33)]:
        assert field[i, j] == pytest.approx(_direct(r, mr, t, m, i, j, d2_exact), rel=2e-3, abs=2e-4)


def test_ncc_exact_term_matters_only_for_weighted_masks():
    r, mr, t, m = _ncc_problem(seed=1)
    cache = ncc._cache_numpy(r, mr, t.shape)
    run = lambda mm, ex: ncc._ncc_numpy(cache, t, mm, min_overlap=0.1, return_overlap=False, d2_exact=ex)   # noqa: E731
    assert np.abs(run(m, True) - run(m, False)).max() > 1e-2         # wall weight 55: M T^2 != M^2 T^2
    mb = (m > 0).astype(np.float32)                                  # binary mask: M T^2 == M^2 T^2
    np.testing.assert_allclose(run(mb, True), run(mb, False), atol=1e-5)   # CPU irfft2 is not bit-repeatable


def test_ncc_torch_exact_matches_numpy():
    pytest.importorskip("torch")
    r, mr, t, m = _ncc_problem(seed=2)
    ref = ncc._ncc_numpy(ncc._cache_numpy(r, mr, t.shape), t, m, min_overlap=0.5, return_overlap=False)
    tc = ncc._cache_torch(r, mr, t.shape)
    one = ncc._ncc_torch_single(tc, t, m, min_overlap=0.5, return_overlap=False)
    many = ncc._ncc_torch_batched(tc, [t, t], [m, m], min_overlap=0.5, return_overlap=False)
    h, w = r.shape[0] - t.shape[0] + 1, r.shape[1] - t.shape[1] + 1    # shifts with the target inside the reference
    np.testing.assert_allclose(one[:h, :w], ref[:h, :w], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(many[1][:h, :w], one[:h, :w], rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ running-sum consensus
H_REF, W_REF, CELL = 37, 41, 0.1


def _to_cell(x, y):   # a cropped reference: world (x, y) m -> fractional (row, col)
    return (y / CELL + 50 / 2.0 - 6, x / CELL + 60 / 2.0 - 9)


def _windows(n, seed=0, on_grid=False):
    rng = np.random.default_rng(seed)
    fields, xh = [], []
    p = np.array([0.31, -0.47])
    for _ in range(n):
        f = rng.standard_normal((H_REF, W_REF)).astype(np.float32)
        v = np.zeros((H_REF, W_REF), bool)
        r0, c0 = rng.integers(0, 12, 2)
        v[r0:r0 + 25, c0:c0 + 28] = True
        fields.append((f, v))
        p = p + rng.normal(0.0, 0.35, 2)
        xy = np.round(p / CELL) * CELL if on_grid else p
        xh.append((float(xy[0]), float(xy[1])))
    return fields, xh


def _quantized_resum(xh, fields, target):
    """Independent re-sum with the snapped offsets g_j - g_t."""
    g = lambda xy: tuple(int(round(v)) for v in _to_cell(*xy))   # noqa: E731
    gt = g(xh[target])
    S = cnt = None
    for (f, v), xy in zip(fields, xh):
        gj = g(xy)
        o = (gj[0] - gt[0], gj[1] - gt[1])
        c = np.zeros_like(f)
        c[v] = f[v] - float(f[v].mean())
        cr = np.roll(c, (-o[0], -o[1]), axis=(0, 1))
        mr = np.roll(v, (-o[0], -o[1]), axis=(0, 1)).astype(np.int32)
        S = cr if S is None else S + cr
        cnt = mr if cnt is None else cnt + mr
    return cons._select(S, cnt, len(fields))


def _peaks(accum, xh, fields, **kw):
    return cons.streaming_peaks(xh, fields, cell_m=CELL, ref_shape=(H_REF, W_REF), to_cell=_to_cell, accum=accum, **kw)


def test_running_sum_equals_quantized_resum_bitwise():
    fields, xh = _windows(14)
    for t in (-1, 0, 6):
        got = cons.consensus_peak(xh, fields, cell_m=CELL, ref_shape=(H_REF, W_REF), to_cell=_to_cell, target_idx=t)
        assert got == _quantized_resum(xh, fields, t % len(xh))
    for L in (-1, 0, 2):
        want = [_quantized_resum(xh[:(len(xh) if L < 0 else min(len(xh), w + L + 1))], fields[:(len(xh) if L < 0 else min(len(xh), w + L + 1))], w)
                for w in range(len(xh))]
        assert _peaks("running", xh, fields, lag_L=L, back_windows=-1) == want


def test_running_sum_finite_back_subtracts_expired_windows():
    fields, xh = _windows(14, seed=3)
    back = 4
    got = _peaks("running", xh, fields, lag_L=0, back_windows=back)
    for w, pk in enumerate(got):
        lo = max(0, w - back)
        want = _quantized_resum(xh[lo:w + 1], fields[lo:w + 1], w - lo)
        assert pk[:2] == want[:2]
        assert pk[2] == pytest.approx(want[2], abs=1e-4) and pk[3] == pytest.approx(want[3], rel=1e-4)


def test_running_and_rounded_resum_coincide_on_grid_positions():
    fields, xh = _windows(10, seed=5, on_grid=True)
    for L in (-1, 0, 3):
        assert _peaks("running", xh, fields, lag_L=L, back_windows=-1) == _peaks("resum", xh, fields, lag_L=L, back_windows=-1)


def test_unknown_accum_mode_raises():
    fields, xh = _windows(3)
    with pytest.raises(ValueError):
        cons.consensus_peak(xh, fields, cell_m=CELL, ref_shape=(H_REF, W_REF), to_cell=_to_cell, accum="vote")


# ------------------------------------------------------------------ rasterizer gates
def test_no_hit_gate_uses_unrotated_slant():
    from padloc.raster import rasterize_points
    # raw float32 slant >= 10 m (escaped ray); the rotated float32 slant is < 10 m
    far = np.array([-7.485086917877197, -6.631249904632568], np.float32)
    rot = -0.11632244573811656
    p = far[None]
    assert np.hypot(p[:, 0], p[:, 1])[0] >= 10.0
    c, s = float(np.cos(rot)), float(np.sin(rot))
    q = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], axis=1)   # the rasterizer's float32 rotation
    assert np.hypot(q[:, 0], q[:, 1])[0] < 10.0
    near = (far * np.float32(0.95)).astype(np.float32)
    kw = dict(radius_m=11.0, cell_m=0.1, wall_weight=55.0, no_hit_threshold_m=10.0, max_segment_jump_m=1.5,
              max_depth_ratio=3.0, max_chord_radial_cos=0.85, rotation_rad=rot)
    grid, mask = rasterize_points(np.stack([near, far]), **kw)
    assert grid.sum() == 1.0                    # only the near ray splats a wall
    g1, m1 = rasterize_points(near[None], **kw)
    assert np.array_equal(mask, m1)             # the escaped ray carves no wedge


# ------------------------------------------------------------------ heading candidates
def _synthetic_chunk(n=20):
    from test_synthetic import raycast, toy_plan
    from padloc import Frame
    from padloc.scan import columns_to_points, pinhole_column_bearings
    cell = 0.10
    occ = toy_plan(cell)
    bearings = pinhole_column_bearings(40)
    poses = [(-6.5 + 0.25 * t, -2.0 + 0.05 * t, 0.15 * math.sin(t / 6.0)) for t in range(n)]
    x0, y0, th0 = poses[0]
    frames = [Frame(points=columns_to_points(raycast(occ, cell, x, y, th, bearings), bearings, radial=False).astype(np.float32),
                    cum_dx_m=x - x0, cum_dy_m=y - y0, cum_dtheta_rad=th - th0) for x, y, th in poses]
    return occ, cell, frames, poses


def test_chunk_end_hough_fallback_when_prepass_is_empty(monkeypatch):
    from padloc import heading, observe
    occ, cell, frames, poses = _synthetic_chunk(35)
    cfg = PadLocConfig(cell_m=cell, accum_radius_m=12.0)
    loc = PadLoc(Reference.from_config(occ, cfg), cfg)
    peaks = heading.chunk_end_candidates(frames, loc.ref.angle_hist, cfg=cfg, side=loc.accum_side)
    assert peaks and all(0.0 <= p[0] < math.pi for p in peaks)
    seen = {}
    real_select = observe.select_heading

    def spy(sel_fields, **kw):
        seen["thetas"] = sorted(sel_fields)
        return real_select(sel_fields, **kw)

    monkeypatch.setattr(heading, "chunk_heading_candidates", lambda *a, **k: [])
    monkeypatch.setattr(observe, "select_heading", spy)
    loc.run_chunk(frames, gauge_heading=poses[0][2])
    assert seen["thetas"] == sorted(heading.expand_candidates(peaks, mirror=cfg.mirror_expansion))


# ------------------------------------------------------------------ windows / published defaults
def test_published_window_defaults():
    cfg = PadLocConfig(cell_m=0.10)
    assert (cfg.window, cfg.stride, cfg.prepass_window, cfg.prepass_stride) == (5, 5, 5, 5)
    assert cfg.ncc_d2_exact and cfg.consensus_accum == "running"
    b = cfg.with_(**BATCH30)
    assert (b.window, b.stride, b.prepass_window, b.prepass_stride) == (30, 30, 30, 30)
    assert window_schedule(15, cfg.window, cfg.stride) == [(0, 5), (5, 10), (10, 15)]
    assert window_schedule(23, cfg.window, cfg.stride)[-1] == (18, 23)


# ------------------------------------------------------------------ frozen readout
def _result(lag):
    wins = [(0, 5), (5, 10), (10, 15)]
    cum = [(0.1 * t, -0.05 * t) for t in range(15)]
    anchors = [(1.0, 2.0), (1.5, 2.5), (3.0, 1.0)]
    last = [(anchors[-1][0] + dx, anchors[-1][1] + dy) for dx, dy in cum]
    return ChunkResult(theta=0.0, anchor_xy_m=anchors[-1], poses_xy_m=last, cum_motion_map_m=cum, windows=wins,
                       stream_anchor_xy_m=anchors, stream_lag_L=lag), cum, anchors


@pytest.mark.parametrize("lag,serving", [(0, [0] * 9 + [1] * 5 + [2]), (1, [0] * 14 + [2])])
def test_frozen_rule(lag, serving):
    res, cum, anchors = _result(lag)
    got = res.frozen_poses_xy_m()
    want = [(anchors[k][0] + cum[t][0], anchors[k][1] + cum[t][1]) for t, k in enumerate(serving)]
    assert got == want


def test_frozen_needs_stream():
    res, _, _ = _result(0)
    res.stream_anchor_xy_m = []
    with pytest.raises(ValueError):
        res.frozen_poses_xy_m()


def test_frozen_last_frame_equals_chunk_end_on_synthetic_chunk():
    occ, cell, frames, poses = _synthetic_chunk()
    cfg = PadLocConfig(cell_m=cell, accum_radius_m=12.0, lag_L=0)
    loc = PadLoc(Reference.from_config(occ, cfg), cfg)
    res = loc.run_chunk(frames, gauge_heading=poses[0][2], stream=True)
    assert res.windows == [(0, 5), (5, 10), (10, 15), (15, 20)]
    fz = res.frozen_poses_xy_m()
    assert fz[-1] == res.poses_xy_m[-1]          # L = 0, unbounded past: the last read is the chunk-end read
    assert fz[10] == (res.stream_anchor_xy_m[1][0] + res.cum_motion_map_m[10][0],
                      res.stream_anchor_xy_m[1][1] + res.cum_motion_map_m[10][1])
