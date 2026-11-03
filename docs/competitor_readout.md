# Competitor readouts

`examples/reproduce_tables.py` reproduces the PadLoc rows. The competitor rows come from the
competitors' released evaluators; this note documents how the paper scores them, how their
chunk-end rows (supplement Sec. H) are produced, and how to validate them.

## Two readouts of the last ten frames

The headline metric is the last-10 success rate: a trial (a chunk of T frames) succeeds when
all ten of its final pose estimates lie within 1 m of ground truth.

- **Per-frame readout** (main Tables 1, 2, 4). Frame t is scored with the estimate available
  at frame t and never revised. F3Loc's `eval_filtering.py` and UnLoc's `evaluate.py
  --no-postprocess` (UnLoc fwd) score their forward filters this way; PadLoc *frozen* is the
  same rule for a method that emits one pose per 5-frame window (each frame keeps the latest
  pose emitted by its time, propagated by odometry).
- **Chunk-end readout** (supplement Tables S6, S7, S9). The last ten poses are written after
  the whole chunk has been read. PadLoc *delayed* writes x̂_{T−j} = a_T + c_{T−j},
  j = 0…9: the chunk-end anchor plus each frame's odometry offset. A forward filter gets the
  same readout from its MAP pose at the last frame, carried back through the transitions it
  consumed (below). UnLoc refined (`utils/postprocessing.py`) and PadLoc accum are batched and
  have a single readout.

## Backfill: the chunk-end readout of a forward filter

Both filters displace a hypothesis at orientation-bin angle o by R(o)·rel, where
rel = (dx, dy, dθ) is the ego-frame transition between consecutive frames
(`get_rel_pose(gt_k, gt_{k+1})` on Gibson; on Bonn UnLoc runs with its ground-truth
transitions, as in the paper). The chunk-end readout of the forward filter is its MAP pose at
the last frame, (pos_{T−1}, θ_{T−1}) in map pixels, carried back through those transitions
(0-based k = T−2 … T−10, res = metres per map pixel):

    θ_k   = wrap(θ_{k+1} − dθ_k)
    pos_k = pos_{k+1} − R(θ_k) · (dx_k, dy_k) / res

With GT-derived transitions this equals

    θ_k   = wrap(θ_{k+1} − (θ^gt_{k+1} − θ^gt_k))
    pos_k = pos_{k+1} − R(θ_k − θ^gt_k) · (xy^gt_{k+1} − xy^gt_k)

which the scorer uses when a dump carries no `transitions` field. The backfilled error differs
from the final-frame error only through the filter's heading error at T; the scorer also
reports a heading-free "final-frame only" column.

UnLoc refined starts from exactly this chain (`postprocess_trajectory` chains the last forward
argmax backwards through the transitions over the last ten frames), then optimizes one global
SE(2) correction of the chain against the floor plan (Adam, 100 steps from zero at
lr 1e-3 m / 1e-5 rad, so at most ≈0.1 m / 0.06°) and re-rounds to the DESDF grid. On Gibson
its success flags coincide trial by trial with the backfilled forward flags (694/694 trials
over T = 15…100), so UnLoc fwd equals UnLoc refined in SR at every T under the chunk-end
readout (main paper Sec. 4.2): UnLoc's refinement gain is the readout change, not the
optimization.

## Producing the dumps

`tools/competitor_chunk_end.py` reads one NPZ per trial:

| key | shape | meaning |
|---|---|---|
| `pred_poses_map` | (T, 3) UnLoc; (T−3, 3) F3Loc | forward MAP pose per frame: x_px, y_px, θ (orientation-bin angle) |
| `gt_poses_map` | (T, 3) | ground truth in the same map-pixel frame |
| `resolution` | () | metres per map pixel (F3Loc: 0.01) |
| `transitions` | (T−1, 3) | ego-frame (dx, dy, dθ) in m / rad actually passed to the transition step; optional |

**UnLoc** (`github.com/matthias-wueest/UnLoc`) has `--no-postprocess` but no pose dump. The
paper's runs add this block to `evaluate.py`, in the trial loop right after the last-10 RMSE
is computed (`transition_ls` is the list upstream `evaluate.py` already keeps for its
postprocess):

```python
        # --- per-trial pose dump ---
        if args.dump_poses_dir is not None:
            os.makedirs(args.dump_poses_dir, exist_ok=True)
            np.savez(
                os.path.join(args.dump_poses_dir, f"{scene}_chunk{idx_within_scene:02d}.npz"),
                pred_poses_map=pred_poses_map.astype(np.float32),  # (T,3) x_px, y_px, theta
                gt_poses_map=poses_map.astype(np.float32),         # (T,3) x_px, y_px, theta
                rmse_last10_m=np.float32(rmse),
                transitions=np.asarray(transition_ls, dtype=np.float32),
                resolution=np.float32(original_resolution),
                scene=str(scene),
                traj_len=np.int32(traj_l),
                chunk_idx=np.int32(idx_within_scene),
            )
```

The Gibson dumps behind the paper carry no `transitions` (they are GT-derived there, so the GT
formula is exact); the Bonn dumps carry them.

- `tools/unloc_gibson_fwd_dump.sh` — `evaluate.py --dataset gibson_t --no-postprocess
  --dump-poses-dir` in the paper's configuration (DESDF 0.1 m, 36 orientation bins, dt 1,
  `unloc_gibson_vitl.ckpt`), T = 15/20/35/50/100.
- `tools/unloc_bonn_fwd_dump.sh` — the paper's Bonn configuration (cameras-only DESDF,
  measured structural depth μ with UnLoc's network b̂, GT-motion transitions, DESDF 0.1 m,
  72 bins, dt 1), 4 sequences × T = 150/300/450. Its Bonn dataset path and measured-depth
  flags belong to the paper's extension of `evaluate.py`, which is not part of upstream UnLoc
  or of this repository.

**F3Loc** (`github.com/felix-ch/f3loc` @ `9e8027d`): `tools/f3loc_pose_dump_patch.py` adds
`--save_pred_dir` to `eval_filtering.py` (the filter and its per-chunk log are unchanged);
`tools/f3loc_fwd_dump.sh` runs the paper's command (`--net_type d` = mono and `comp_s`,
T = 15/20/35/50/100) with it. The filter emits no estimate for the first 3 frames of a chunk;
the last ten frames are always scored.

Scoring (set the environment variables at the top of each driver first):

```bash
python tools/competitor_chunk_end.py --layout gibson --dumps RUN --archive ARCHIVE --out out/unloc_gibson
#   RUN/poses_T{T}/*.npz; ARCHIVE/results_T{T}.npz = UnLoc's results.npz of the same configuration
python tools/competitor_chunk_end.py --layout bonn --dumps ROOT --archive ROOT_PUB --out out/unloc_bonn
#   ROOT/{seq}_T{T}/poses/*.npz; ROOT_PUB/{seq}_T{T}/bonn/res0.1_orn72_T{T}_dt1/results.npz
python tools/competitor_chunk_end.py --layout f3loc --dumps OUT --archive OUT --out out/f3loc
#   OUT/{d,comp_s}_T{T}/*.npz and OUT/{d,comp_s}_T{T}.log (F3Loc's per-chunk "last_errors")
```

`--padloc` / `--competitors` take optional reference CSVs (columns `table,group,T,sr,n`) that
fill the PadLoc-delayed and main-table columns; without them those columns print "—".
`--out` writes the table as `.md`, `.csv` and `.json`.

## Validation (all three hold in every cell)

1. **The dumps reproduce the evaluator's own result.** The per-frame rule recomputed from the
   dumps gives the evaluator's per-frame SR and an identical sorted per-trial RMSE multiset
   (column `archive max|Δ sorted RMSE|`: ≤ 1.5e-6 m in the 17 UnLoc cells, ≤ 3.3e-7 m in the
   10 F3Loc cells).
2. **Heading convention.** The median |θ_T − θ^gt_T| over chunk-end successes lies inside one
   orientation bin (10° Gibson, 5° Bonn): 2.4–3.1° UnLoc Gibson, 0.3–5.9° UnLoc Bonn,
   3.2–3.5° F3Loc.
3. **Transitions.** Where `transitions` are dumped (UnLoc Bonn, F3Loc), the backfill from them
   equals the GT-derived formula (column `max|Δ| vs GT-backfill`: 0.000 m in every cell).

## Results

Competitor columns: `tools/competitor_chunk_end.py` on the paper's dumps. PadLoc columns:
`expected.json` (`frozen` = main tables, `chunk_end` = supplement Sec. H). Gained / lost =
trials that the chunk-end readout turns into successes / failures.

Gibson, UnLoc μ depth (main Table 1 / supplement Table S6), 9 scenes:

| T | n | UnLoc fwd per-frame | UnLoc fwd chunk-end | gained / lost | UnLoc refined | PadLoc frozen | PadLoc delayed |
|---|---|---|---|---|---|---|---|
| 15 | 268 | 0.660 | 0.810 | +40 / −0 | 0.810 | 0.731 | 0.843 |
| 20 | 200 | 0.710 | 0.865 | +32 / −1 | 0.865 | 0.840 | 0.900 |
| 35 | 111 | 0.874 | 0.919 | +5 / −0 | 0.919 | 0.919 | 0.937 |
| 50 | 78 | 0.923 | 0.949 | +2 / −0 | 0.949 | 1.000 | 1.000 |
| 100 | 37 | 0.973 | 0.973 | +0 / −0 | 0.973 | 1.000 | 1.000 |

Gibson, F3Loc depth (main Table 1 / supplement Table S6), 9 scenes:

| depth | T | F3Loc filter per-frame | F3Loc filter chunk-end | gained / lost | PadLoc frozen | PadLoc delayed |
|---|---|---|---|---|---|---|
| mono | 15 | 0.291 | 0.511 | +61 / −2 | 0.496 | 0.608 |
| mono | 20 | 0.345 | 0.530 | +37 / −0 | 0.585 | 0.695 |
| mono | 35 | 0.550 | 0.703 | +17 / −0 | 0.766 | 0.775 |
| mono | 50 | 0.705 | 0.808 | +8 / −0 | 0.821 | 0.846 |
| mono | 100 | 0.892 | 0.892 | +0 / −0 | 0.946 | 0.946 |
| comp_s | 15 | 0.366 | 0.567 | +54 / −0 | 0.563 | 0.675 |
| comp_s | 20 | 0.410 | 0.580 | +34 / −0 | 0.685 | 0.770 |
| comp_s | 35 | 0.622 | 0.757 | +15 / −0 | 0.883 | 0.910 |
| comp_s | 50 | 0.795 | 0.872 | +6 / −0 | 0.949 | 0.962 |
| comp_s | 100 | 0.946 | 0.946 | +0 / −0 | 0.946 | 0.946 |

Bonn (LiLocBench), four sequences (main Table 4 / supplement Table S9); UnLoc with its
ground-truth transitions, PadLoc with wheel odometry:

| cell | T | n | UnLoc fwd per-frame | UnLoc fwd chunk-end | gained / lost | PadLoc frozen | PadLoc delayed |
|---|---|---|---|---|---|---|---|
| static_0 | 150 | 59 | 0.322 | 0.373 | +3 / −0 | 0.59 | 0.59 |
| dynamics_0 | 150 | 15 | 0.333 | 0.333 | +0 / −0 | 0.67 | 0.67 |
| lt_changes_0 | 150 | 43 | 0.209 | 0.279 | +3 / −0 | 0.60 | 0.60 |
| lt_changes_dynamics_0 | 150 | 55 | 0.327 | 0.345 | +1 / −0 | 0.45 | 0.47 |
| **pooled** | 150 | 172 | 0.297 | 0.337 | +7 / −0 | 0.56 | 0.56 |
| static_0 | 300 | 29 | 0.586 | 0.586 | +0 / −0 | 0.79 | 0.79 |
| dynamics_0 | 300 | 7 | 0.429 | 0.429 | +0 / −0 | 0.71 | 0.71 |
| lt_changes_0 | 300 | 21 | 0.429 | 0.429 | +0 / −0 | 0.71 | 0.71 |
| lt_changes_dynamics_0 | 300 | 27 | 0.444 | 0.407 | +0 / −1 | 0.63 | 0.63 |
| **pooled** | 300 | 84 | 0.488 | 0.476 | +0 / −1 | 0.71 | 0.71 |
| static_0 | 450 | 19 | 0.684 | 0.737 | +1 / −0 | 0.89 | 0.89 |
| dynamics_0 | 450 | 5 | 0.600 | 0.600 | +0 / −0 | 0.60 | 0.60 |
| lt_changes_0 | 450 | 14 | 0.214 | 0.214 | +0 / −0 | 0.79 | 0.79 |
| lt_changes_dynamics_0 | 450 | 18 | 0.389 | 0.389 | +0 / −0 | 0.72 | 0.72 |
| **pooled** | 450 | 56 | 0.464 | 0.482 | +1 / −0 | 0.79 | 0.79 |
