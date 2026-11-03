#!/usr/bin/env python3
"""Reproduce the paper's PadLoc consensus numbers on the fixture package.

    python examples/reproduce_tables.py [--fixtures DIR] [--out DIR] [FILTER ...]

Cells are ``table/group/T`` (e.g. ``gibson/unloc/15``, ``bonn_modular/lidar_vo/300``,
``bonn_unloc/wheel:static_0/150``); positional filters select cells by substring.
Every chunk is scored under both readouts of the paper: the chunk-end readout
(every scored frame from the last window's consensus) and the frozen readout (each
frame keeps the causal L = 0 pose emitted by its time). Writes one CSV per cell
(per-chunk rows, both readouts) and prints, per cell and readout, the pooled last-10
SR@1 m, the mean (R) and median (sR) last-10 RMSE over successful chunks, the all-trial
median (mR) and SWA, followed by the verdict against ``expected.json`` (the published SR).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from padloc import PadLoc, PadLocConfig, Reference          # noqa: E402
from padloc.config import BONN_LIDAR, BONN_RGBD, BONN_STRUCTURAL, GIBSON   # noqa: E402
from padloc.datasets import bonn, gibson                     # noqa: E402
from padloc.metrics import chunk_metrics, pooled             # noqa: E402

GIB_SCENES = ["Spencerville", "Spotswood", "Springhill", "Stilwell", "Stokes", "Sumas", "Superior", "Swormville", "Woonsocket"]
GIB_T = [15, 20, 35, 50, 100]
USEQS = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
CAM = dict(camera_fx=384.891785, camera_cx=324.345856, camera_cols=640)
TOL = 0.0051
FIELDS = ("scene", "chunk_idx", "rmse_last10_m", "success_1_0m", "success_0_5m", "success_0_3m", "success_0_2m", "n_frames", "theta_deg",
          "frozen_rmse_last10_m", "frozen_success_1_0m")
READOUTS = {"chunk_end": ("success_1_0m", "rmse_last10_m"), "frozen": ("frozen_success_1_0m", "frozen_rmse_last10_m")}


def run_scene(scene, depth, cfg, *, traj_len, chunks, out_csv, log):
    ref = Reference.from_config(scene.occupancy, cfg, native_occupancy=scene.native_occupancy)
    loc = PadLoc(ref, cfg)
    rows = []
    t0 = time.time()
    for ch in chunks:
        res = loc.run_chunk(ch.frames, gauge_heading=ch.gauge_heading, prepass_frames=ch.prepass_frames, prepass_gauge_heading=ch.prepass_gauge_heading,
                            stream=True)
        m = chunk_metrics(res.poses_xy_m, ch.gt_xy)
        fz = chunk_metrics(res.frozen_poses_xy_m(), ch.gt_xy)
        rows.append({"scene": scene.name, "chunk_idx": ch.idx, "theta_deg": round(float(res.theta) * 180.0 / 3.141592653589793, 3), **m,
                     "frozen_rmse_last10_m": fz["rmse_last10_m"], "frozen_success_1_0m": fz["success_1_0m"]})
    log(f"    {scene.name}: {len(rows)} chunks in {time.time() - t0:.0f}s")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return rows


def cells(fix: Path, flt):
    out = []
    gib_root = fix / "gibson" / "Gibson Floorplan Localization Dataset" / "gibson_t"
    for T in GIB_T:
        for arm in ("mono", "comp", "unloc"):
            out.append(("gibson", arm, T, dict(kind="gibson", arm=arm, root=gib_root, T=T)))
    bonn_rows = {"lidar_vo": ("bonn", "laser_full", "planar_vo.tum", BONN_LIDAR),
                 "lidar_wheel": ("bonn", "laser_full", "wheel_odom_static0_framemotion.tum", BONN_LIDAR),
                 "depth90_vo": ("bonn", "depth90", "planar_vo.tum", BONN_LIDAR),
                 "rgbd_vo": ("bonn_cam", "camera_depth", "planar_vo_camera.tum", BONN_RGBD)}
    for row, (sub, src, tum, preset) in bonn_rows.items():
        for T in (300, 450):
            out.append(("bonn_modular", row, T, dict(kind="bonn", root=fix / sub / "bonn" / "prepared", seq="static_0", src=src, tum=fix / "odom" / tum, preset=preset, T=T)))
    for mot in ("wheel", "GT"):
        for seq in USEQS:
            for T in (150, 300, 450):
                tum = (fix / "odom" / f"wheel_cam_{seq}_framemotion.tum") if mot == "wheel" else None
                out.append(("bonn_unloc", f"{mot}:{seq}", T, dict(kind="bonn", root=fix / "bonn_struct_C" / "bonn" / "prepared", seq=seq, src="camera_depth", tum=tum, preset=BONN_STRUCTURAL, T=T)))
    if flt:
        out = [c for c in out if any(f in f"{c[0]}/{c[1]}/{c[2]}" for f in flt)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*")
    ap.add_argument("--fixtures", default=str(HERE.parent / "fixtures"))
    ap.add_argument("--out", default=str(HERE.parent / "runs"))
    ap.add_argument("--scenes", default=None, help="comma-separated Gibson scene subset (smoke tests)")
    args = ap.parse_args()
    fix, out = Path(args.fixtures), Path(args.out)
    expected = json.load((HERE.parent / "expected.json").open())
    scenes_filter = args.scenes.split(",") if args.scenes else None
    log = lambda s: print(s, flush=True)   # noqa: E731
    fmt = lambda p: "  ".join(f"{ro} SR={q['SR@1m']:.3f} R={q['meanRMSE_success']:.2f} sR={q['medRMSE_success']:.2f} "   # noqa: E731
                              f"mR={q['medRMSE_all']:.2f} SWA={q['SWA']:.3f}" for ro, q in p.items()) + f"  n={p['chunk_end']['n']}"
    results, cell_rows = {}, {}
    for table, grp, T, spec in cells(fix, args.filters):
        rows = []
        if spec["kind"] == "gibson":
            cfg = PadLocConfig(cell_m=0.10, lag_L=0, **GIBSON)
            for name in (scenes_filter or GIB_SCENES):
                scene = gibson.load_scene(spec["root"], name, cell_m=0.10)
                if spec["arm"] == "unloc":
                    depth = gibson.load_unloc_depth(scene, fix / "gibson_unloc_dump")
                    drop = cfg.bhat_drop_m
                else:
                    depth = gibson.load_f3loc_depth(scene, f"depth_learned_f3loc_{spec['arm']}.txt")
                    drop = None
                chunks = gibson.iter_chunks(scene, depth, traj_len=T, bhat_drop_m=drop)
                rows += run_scene(scene, depth, cfg, traj_len=T, chunks=chunks, out_csv=out / f"{table}__{grp}__T{T}" / f"{name}.csv", log=log)
        else:
            scene = bonn.load_scene(spec["root"], spec["seq"])
            cfg = PadLocConfig(cell_m=scene.cell_m, lag_L=0, **spec["preset"])
            depth = bonn.load_depth(scene, spec["src"], **(CAM if spec["src"] == "camera_depth" else {}))
            chunks = bonn.iter_chunks(scene, depth, traj_len=T, odometry_tum=spec["tum"], max_chunks=200)
            rows = run_scene(scene, depth, cfg, traj_len=T, chunks=chunks, out_csv=out / f"{table}__{grp}__T{T}" / f"{spec['seq']}.csv", log=log)
        p = {ro: pooled([{"success_1_0m": r[sk], "rmse_last10_m": r[rk]} for r in rows]) for ro, (sk, rk) in READOUTS.items()}
        results[(table, grp, str(T))] = p
        cell_rows[(table, grp, str(T))] = rows
        log(f"  {table}/{grp}/T{T}: " + fmt(p))
    # Bonn pooled over the four sequences (Table 4 pooled row and text, supplement Table S9)
    for mot in ("wheel", "GT"):
        for T in ("150", "300", "450"):
            keys = [("bonn_unloc", f"{mot}:{seq}", T) for seq in USEQS]
            if all(k in cell_rows for k in keys):
                rows = [r for k in keys for r in cell_rows[k]]
                log(f"  bonn_unloc/{mot}:POOLED/T{T}: " + fmt({ro: pooled([{"success_1_0m": r[sk], "rmse_last10_m": r[rk]} for r in rows])
                                                              for ro, (sk, rk) in READOUTS.items()}))
    # verdict: every published value of both readouts (null = not yet published)
    ok = total = pending = 0
    fails = []

    def chk(key, exp, got):
        nonlocal ok, total, pending
        if exp is None:
            pending += 1
            return
        total += 1
        if got is not None and abs(got - exp) <= TOL:
            ok += 1
        else:
            fails.append((key, exp, got))

    def sr(ro, key):
        c = results.get(key)
        return c[ro]["SR@1m"] if c else None

    for ro in READOUTS:
        ex = expected[ro]
        for arm, ts in ex["gibson"].items():
            for T, exp in ts.items():
                chk(f"{ro}/gibson/{arm}/T{T}", exp, sr(ro, ("gibson", arm, T)))
        for row, ts in ex["bonn_modular"].items():
            for T, exp in ts.items():
                chk(f"{ro}/bonn_modular/{row}/T{T}", exp, sr(ro, ("bonn_modular", row, T)))
        bu = ex["bonn_unloc"]
        for seq, ts in bu["wheel_perseq"].items():
            for T, exp in ts.items():
                chk(f"{ro}/bonn_unloc/wheel/{seq}/T{T}", exp, sr(ro, ("bonn_unloc", f"wheel:{seq}", T)))
        for mot, key in (("wheel", "wheel_pooled"), ("GT", "GT_pooled")):
            for T, exp in bu[key].items():
                cells_t = [results.get(("bonn_unloc", f"{mot}:{seq}", T)) for seq in USEQS]
                if all(c is not None for c in cells_t):
                    tot = sum(c[ro]["n"] for c in cells_t)
                    got = sum(c[ro]["SR@1m"] * c[ro]["n"] for c in cells_t) / tot if tot else None
                else:
                    got = None
                chk(f"{ro}/bonn_unloc/{mot}_pooled/T{T}", exp, got)
    if not args.filters and not scenes_filter:
        print(f"\n=== reproduced {ok}/{total} published cells (tol {TOL}; {pending} not yet published) ===")
        for key, exp, got in fails:
            print(f"FAIL {key:46s} paper={exp:.3f} current={'n/a' if got is None else f'{got:.3f}'}")


if __name__ == "__main__":
    main()
