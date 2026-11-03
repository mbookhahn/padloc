#!/usr/bin/env python3
"""Forward filters (UnLoc fwd, F3Loc filter) scored under the chunk-end readout.

UnLoc's evaluate.py and F3Loc's eval_filtering.py score each of the last ten frames with the
forward filter's MAP pose *at that frame* (per-frame readout; the competitor rows of the
paper's main tables). PadLoc delayed and UnLoc refined write the last ten poses after reading
the whole chunk. This scorer gives a forward filter the same chunk-end readout: its MAP pose at
the last frame T, backfilled to frames T-9..T-1 with the odometry the filter itself consumed,
rotated through the filter's own heading estimate as both filters' transition step does (a
hypothesis at bin angle o is displaced by R(o) @ rel, rel = ego-frame transition). The UnLoc
fwd and F3Loc filter rows of the supplement's chunk-end tables (S6, S7, S9) come from this
script; docs/competitor_readout.md documents the protocol, the dump patches and the validation.

Inputs: one NPZ per trial with pred_poses_map (forward MAP pose per frame in map pixels,
theta = orientation-bin angle), gt_poses_map (T,3), resolution (m/px), optional transitions
(T-1,3) = the ego-frame [dx,dy,dtheta] (m, rad) actually applied. UnLoc: evaluate.py
--dump-poses-dir (tools/unloc_gibson_fwd_dump.sh, tools/unloc_bonn_fwd_dump.sh), (T,3) poses.
F3Loc: eval_filtering.py --save_pred_dir with tools/f3loc_pose_dump_patch.py
(tools/f3loc_fwd_dump.sh), (T-3,3) poses (no estimate for the first 3 frames), resolution
0.01 m/px.
Backfill, 0-based, k from T-2 down to T-10:
    with dumped transitions:  th_k = wrap(th_{k+1} - dth_k);  pos_k = pos_{k+1} - R(th_k) @ (dx_k,dy_k)/res
    GT-derived (Gibson, Bonn published runs: transitions = get_rel_pose(gt_k, gt_{k+1})):
                              th_k = wrap(th_{k+1} - (gt_th[k+1]-gt_th[k]))
                              pos_k = pos_{k+1} - R(th_k - gt_th[k]) @ (gt_xy[k+1]-gt_xy[k])
Both agree when the transitions were GT-derived. The backfilled error differs from the
final-frame error only through the heading error at T ("final-frame only" = heading-free).
Validation: the per-frame rule recomputed from the dumps must reproduce the published
archive (sorted per-trial RMSE multiset identical); median |theta_T - gt| on successes must
sit inside one orientation bin; with dumped transitions the backfill must equal the GT formula.

Layouts: gibson  -> --dumps DIR/poses_T{T}/*.npz, --archive DIR/results_T{T}.npz
         bonn    -> --dumps ROOT/{seq}_T{T}/poses/*.npz, --archive ROOT/{seq}_T{T}/bonn/res0.1_orn72_T{T}_dt1/results.npz
         f3loc   -> --dumps ROOT/{d,comp_s}_T{T}/*.npz, --archive DIR with F3Loc's {d,comp_s}_T{T}.log
                    (per-chunk "last_errors [...]" = the per-frame rule)
Reference CSVs (--padloc, --competitors; columns table,group,T,sr,n) are optional: they fill
the PadLoc-delayed / main-table per-frame columns; without them those columns print "—".
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

TLENS = {"gibson": (15, 20, 35, 50, 100), "bonn": (150, 300, 450), "f3loc": (15, 20, 35, 50, 100)}
F3_ARMS = (("d", "f3loc_mono", "mono"), ("comp_s", "f3loc_comp", "comp"))  # (net_type, competitor group, PadLoc group)
BONN_SEQS = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
TAU = 1.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def backfill(P, G, res, trans):
    T = G.shape[0]  # frame count; P may hold fewer rows (F3Loc: first 3 frames unfiltered), P[-1] is always frame T-1
    pos = P[-1, :2].astype(np.float64).copy(); th = float(P[-1, 2])
    err = np.zeros(10); err[-1] = np.linalg.norm(pos - G[-1, :2]) * res
    for j, k in enumerate(range(T - 2, T - 11, -1)):
        if trans is not None:
            dx, dy, dth = trans[k]
            th = wrap(th - dth)
            pos = pos - rot(th) @ np.array([dx, dy]) / res
        else:
            th = wrap(th - wrap(G[k + 1, 2] - G[k, 2]))
            pos = pos - rot(th - G[k, 2]) @ (G[k + 1, :2] - G[k, :2])
        err[-2 - j] = np.linalg.norm(pos - G[k, :2]) * res
    return err


def score_trial(z):
    P, G, res = np.asarray(z["pred_poses_map"], float), np.asarray(z["gt_poses_map"], float), float(z["resolution"])
    trans = np.asarray(z["transitions"], float) if "transitions" in z.files else None
    err_pf = np.linalg.norm(P[-10:, :2] - G[-10:, :2], axis=1) * res
    bf = backfill(P, G, res, trans)
    bf_gt = backfill(P, G, res, None) if trans is not None else bf
    return dict(pf=err_pf, bf=bf, fin=np.full(10, err_pf[-1]), dth=wrap(P[-1, 2] - G[-1, 2]),
                trans_vs_gt=float(np.max(np.abs(bf - bf_gt))), has_trans=trans is not None)


def agg(errs):
    s = np.array([bool((e < TAU).all()) for e in errs]); r = np.array([float(np.sqrt(np.mean(e ** 2))) for e in errs])
    return dict(n=len(s), sr=float(s.mean()), R=float(r[s].mean()) if s.any() else np.nan,
                med=float(np.median(r)), swa=float(np.mean(s * (1 - np.minimum(r, TAU) / TAU)))), s, r


def archive_check(path, r_pf):
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".log"):  # F3Loc log: per-chunk "last_errors [e1 .. e10]" (m)
        import re
        errs = [np.array([float(x) for x in m.group(1).split()]) for m in re.finditer(r"last_errors \[([^\]]*)\]", open(path, errors="replace").read())]
        s = np.array([bool((e < TAU).all()) for e in errs]); r = np.array([float(np.sqrt(np.mean(e ** 2))) for e in errs])
        ok = len(r) == len(r_pf)
        return dict(archive_sr=float(s.mean()), archive_n=int(len(r)),
                    sorted_rmse_max_abs_diff=float(np.max(np.abs(np.sort(r) - np.sort(r_pf)))) if ok else np.nan)
    z = np.load(path, allow_pickle=True)
    i = int(np.argmin(np.abs(z["success_thresholds_m"] - 1.0)))
    ok = len(z["rmses"]) == len(r_pf)
    return dict(archive_sr=float(z["success_flags"][:, i].mean()), archive_n=int(len(z["rmses"])),
                sorted_rmse_max_abs_diff=float(np.max(np.abs(np.sort(z["rmses"]) - np.sort(r_pf)))) if ok else np.nan)


def ref_rows(path, table):
    out = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["table"] == table:
                out[(r["group"], int(r["T"]))] = r
    return out


def cell(files, arch, label, T, n_expect=None):
    tr = [score_trial(np.load(f, allow_pickle=True)) for f in files]
    m_pf, s_pf, r_pf = agg([t["pf"] for t in tr]); m_bf, s_bf, r_bf = agg([t["bf"] for t in tr]); m_fin, _, _ = agg([t["fin"] for t in tr])
    dth = np.array([t["dth"] for t in tr])
    return dict(cell=label, T=T, n=m_pf["n"], fwd_perframe_sr=m_pf["sr"], fwd_delayed_sr=m_bf["sr"], final_frame_sr=m_fin["sr"],
                gained=int((~s_pf & s_bf).sum()), lost=int((s_pf & ~s_bf).sum()),
                fwd_perframe_R=m_pf["R"], fwd_delayed_R=m_bf["R"], fwd_perframe_med=m_pf["med"], fwd_delayed_med=m_bf["med"],
                fwd_perframe_swa=m_pf["swa"], fwd_delayed_swa=m_bf["swa"],
                med_abs_heading_err_deg_succ=float(np.degrees(np.median(np.abs(dth[s_bf])))) if s_bf.any() else np.nan,
                dumped_transitions=all(t["has_trans"] for t in tr), trans_vs_gt_backfill_max_m=float(max(t["trans_vs_gt"] for t in tr)),
                succ_bf=s_bf, rmse_bf=r_bf, succ_pf=s_pf, rmse_pf=r_pf, **archive_check(arch, r_pf))


def pooled(cells_, label, T):
    s_pf = np.concatenate([c["succ_pf"] for c in cells_]); r_pf = np.concatenate([c["rmse_pf"] for c in cells_])
    s_bf = np.concatenate([c["succ_bf"] for c in cells_]); r_bf = np.concatenate([c["rmse_bf"] for c in cells_])
    return dict(cell=label, T=T, n=int(len(s_pf)), fwd_perframe_sr=float(s_pf.mean()), fwd_delayed_sr=float(s_bf.mean()),
                final_frame_sr=np.nan, gained=int((~s_pf & s_bf).sum()), lost=int((s_pf & ~s_bf).sum()),
                fwd_perframe_R=float(r_pf[s_pf].mean()) if s_pf.any() else np.nan, fwd_delayed_R=float(r_bf[s_bf].mean()) if s_bf.any() else np.nan,
                fwd_perframe_med=float(np.median(r_pf)), fwd_delayed_med=float(np.median(r_bf)),
                fwd_perframe_swa=float(np.mean(s_pf * (1 - np.minimum(r_pf, TAU) / TAU))), fwd_delayed_swa=float(np.mean(s_bf * (1 - np.minimum(r_bf, TAU) / TAU))),
                med_abs_heading_err_deg_succ=np.nan, dumped_transitions=all(c["dumped_transitions"] for c in cells_),
                trans_vs_gt_backfill_max_m=max(c["trans_vs_gt_backfill_max_m"] for c in cells_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("gibson", "bonn", "f3loc"), default="gibson")
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--archive", default=None)
    ap.add_argument("--padloc", default=None, help="optional reference CSV (table,group,T,sr,n): PadLoc delayed SR per cell")
    ap.add_argument("--competitors", default=None, help="optional reference CSV (table,group,T,sr,n): main-table per-frame SR per cell")
    ap.add_argument("--T", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Ts = a.T or list(TLENS[a.layout])
    rows = []
    if a.layout == "gibson":
        pad = ref_rows(a.padloc, "gibson"); comp = ref_rows(a.competitors, "gibson")
        for T in Ts:
            files = sorted(glob.glob(os.path.join(a.dumps, f"poses_T{T}", "*.npz")))
            if not files:
                print(f"T={T}: no dumps"); continue
            r = cell(files, os.path.join(a.archive, f"results_T{T}.npz") if a.archive else None, "gibson", T)
            r.update(padloc_sr=float(pad[("unloc", T)]["sr"]) if ("unloc", T) in pad else np.nan,
                     unloc_refined_sr=float(comp[("unloc_refined", T)]["sr"]) if ("unloc_refined", T) in comp else np.nan,
                     unloc_fwd_published_sr=float(comp[("unloc_fwd", T)]["sr"]) if ("unloc_fwd", T) in comp else np.nan)
            rows.append(r)
    elif a.layout == "f3loc":
        pad = ref_rows(a.padloc, "gibson"); comp = ref_rows(a.competitors, "gibson")
        for net, cg, pg in F3_ARMS:
            for T in Ts:
                files = sorted(glob.glob(os.path.join(a.dumps, f"{net}_T{T}", "*.npz")))
                if not files:
                    print(f"{net} T={T}: no dumps"); continue
                r = cell(files, os.path.join(a.archive, f"{net}_T{T}.log") if a.archive else None, cg, T)
                r.update(padloc_sr=float(pad[(pg, T)]["sr"]) if (pg, T) in pad else np.nan, unloc_refined_sr=np.nan,
                         unloc_fwd_published_sr=float(comp[(cg, T)]["sr"]) if (cg, T) in comp else np.nan)
                rows.append(r)
    else:
        pad = ref_rows(a.padloc, "bonn_unloc"); comp = ref_rows(a.competitors, "bonn_unloc")
        for T in Ts:
            percell = []
            for seq in BONN_SEQS:
                files = sorted(glob.glob(os.path.join(a.dumps, f"{seq}_T{T}", "poses", "*.npz")))
                if not files:
                    print(f"{seq} T={T}: no dumps"); continue
                arch = os.path.join(a.archive, f"{seq}_T{T}", "bonn", f"res0.1_orn72_T{T}_dt1", "results.npz") if a.archive else None
                r = cell(files, arch, seq, T)
                r.update(padloc_sr=float(pad[(f"wheel:{seq}", T)]["sr"]) if (f"wheel:{seq}", T) in pad else np.nan,
                         unloc_refined_sr=np.nan,
                         unloc_fwd_published_sr=float(comp[(f"unloc:{seq}", T)]["sr"]) if (f"unloc:{seq}", T) in comp else np.nan)
                rows.append(r); percell.append(r)
            if percell:
                p = pooled(percell, "POOLED", T)
                ns = [pad[(f"wheel:{s}", T)] for s in BONN_SEQS if (f"wheel:{s}", T) in pad]
                p.update(padloc_sr=sum(float(x["sr"]) * int(x["n"]) for x in ns) / sum(int(x["n"]) for x in ns) if ns else np.nan,
                         unloc_refined_sr=np.nan,
                         unloc_fwd_published_sr=float(comp[("unloc:POOLED", T)]["sr"]) if ("unloc:POOLED", T) in comp else np.nan,
                         **{k: np.nan for k in ("archive_sr", "sorted_rmse_max_abs_diff")}, archive_n=np.nan)
                rows.append(p)
    f3 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    md = ["| cell | T | n | fwd per-frame: main table / dump | archive max|Δ sorted RMSE| | **fwd CHUNK-END** | final-frame only | gained/lost | R pf / chunk-end | med pf / chunk-end | SWA pf / chunk-end | med |Δθ_T| succ (°) | trans. dumped / max|Δ| vs GT-backfill (m) | PadLoc delayed | UnLoc refined |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cell']} | {r['T']} | {r['n']} | {f3(r['unloc_fwd_published_sr'])} / {f3(r['fwd_perframe_sr'])} | {r.get('sorted_rmse_max_abs_diff', float('nan')):.1e} | **{f3(r['fwd_delayed_sr'])}** | {f3(r['final_frame_sr'])} | +{r['gained']} / −{r['lost']} | "
                  f"{r['fwd_perframe_R']:.2f} / {r['fwd_delayed_R']:.2f} | {r['fwd_perframe_med']:.2f} / {r['fwd_delayed_med']:.2f} | {r['fwd_perframe_swa']:.3f} / {r['fwd_delayed_swa']:.3f} | {r['med_abs_heading_err_deg_succ']:.1f} | "
                  f"{'yes' if r['dumped_transitions'] else 'no'} / {r['trans_vs_gt_backfill_max_m']:.3f} | {f3(r['padloc_sr'])} | {f3(r['unloc_refined_sr'])} |")
    print("\n".join(md))
    if a.out:
        keep = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows]
        open(a.out + ".md", "w").write("\n".join(md) + "\n")
        with open(a.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keep[0].keys())); w.writeheader(); w.writerows(keep)
        json.dump(keep, open(a.out + ".json", "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
