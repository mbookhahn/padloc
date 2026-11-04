# PadLoc: Training-Free Floor-Plan Localization by Sparse Geometric Matching

Localizes a moving sensor in a known 2-D floor plan from per-frame **wall depth**
(any source: 2-D LiDAR, RGB-D, learned monocular depth) and **planar odometry**.
No training: a masked normalized cross-correlation over a few dominant wall
orientations, decoded online by a moving-pose Gaussian-mixture Bayes filter and a
consensus readout.

This repository is the reference implementation of the WACV 2027 paper
([arXiv:2611.05172](https://arxiv.org/abs/2611.05172)) and reproduces its PadLoc numbers on
Gibson and Bonn (LiLocBench).

## Install

```bash
pip install -e .            # numpy, scipy, opencv, concave-hull, pyyaml
pip install -e .[gpu]       # + torch: PADLOC_FFT_BACKEND=torch runs the FFT on the GPU
```

## Use

```python
from padloc import PadLoc, PadLocConfig, Reference, Frame

cfg = PadLocConfig(cell_m=0.10, lag_L=0)           # the paper's defaults; set the front-end knobs
ref = Reference.from_config(occupancy, cfg)        # 0 = wall raster at cell_m (once per plan)
loc = PadLoc(ref, cfg)

frames = [Frame(points=pts_xy, cum_dx_m=cx, cum_dy_m=cy, cum_dtheta_rad=ct) for ...]
res = loc.run_chunk(frames, gauge_heading=heading_of_frame0, stream=True)
res.theta, res.anchor_xy_m, res.poses_xy_m        # chunk-end readout (PadLoc delayed)
res.stream_xy_m, res.frozen_poses_xy_m()           # causal stream, frozen readout (PadLoc frozen)
```

Defaults are the paper's back-end: fusion and pre-pass windows W = S = 5 frames
(`BATCH30` = 30/30 for front-ends that reconstruct in 30-frame batches, as on Hilti),
the weighted masked NCC with the exact second moment (Eqs. 1–2), and the running
consensus sums (Sec. 3.4). `docs/config_mapping.md` maps every parameter of the paper
(supplement Table S1) to its `PadLocConfig` field; `ncc_d2_exact=False` and
`consensus_accum="resum"` select the two alternatives the supplement discusses.

`padloc.scan` turns depth columns (+ bearings) into `points`; `padloc.odometry`
integrates odometry; `padloc.datasets` has the Gibson/Bonn adapters used by the paper.

## Pipeline (one module per stage)

| stage | module | what |
|---|---|---|
| 1 | `reference` | wall raster → concave-hull interior mask → blurred distance field → angle histogram → FFT cache |
| 2 | `scan` | depth columns → local wall points (b̂ column drop, no-hit threshold) |
| 3 | `odometry` | increments → cumulative motion, gauge-invariant rotation |
| 4 | `heading` | causal Hough pre-pass, histogram cross-correlation, von-Mises tracker, mirror expansion; chunk-end Hough fallback |
| 5 | `raster`, `windows`, `ncc` | W/S window rasters (built once), weighted masked NCC (Padfield, exact second moment) via FFT per heading |
| 6 | `observe` | overlap-weighted heading selection; top-K NMS + Hessian covariance → Gaussian observations |
| 7 | `gmbf` | moving-pose GM-BF: predict, Mahalanobis gate, update, prune/merge/cap |
| 8 | `consensus` | heal, running consensus sums, eligibility, anchor; bounded-lag streaming and frozen readout |
|   | `pipeline` | `PadLoc.run_chunk` ties it together; one NCC per (heading, window), cached |

## Reproduce the paper's numbers

Download the fixture package (per-frame depth columns, odometry, floor plans and poses;
98 MB, 290 MB extracted) from the
[v1.0.0 release](https://github.com/mbookhahn/padloc/releases/tag/v1.0.0) and extract it
in the repository root:

```bash
curl -LO https://github.com/mbookhahn/padloc/releases/download/v1.0.0/padloc-fixtures-v1.0.0.tar.gz
shasum -a 256 padloc-fixtures-v1.0.0.tar.gz          # e2d5fb9a9311f9a441c26cc3b953dec9a97252a22745fb5649bbe185e5db89e7
tar xzf padloc-fixtures-v1.0.0.tar.gz                # -> fixtures/
PADLOC_FFT_BACKEND=torch python examples/reproduce_tables.py               # all 41 cells
PADLOC_FFT_BACKEND=torch python examples/reproduce_tables.py gibson/unloc  # cells by substring
```

Every chunk is scored under both readouts of the paper: **frozen** (main paper; each frame
keeps the latest pose emitted by its time) and **chunk-end** (supplement Sec. H, PadLoc
delayed; the last ten frames from the final anchor). Per cell and readout the runner prints
the last-10 SR@1 m, the mean (R) and median (sR) last-10 RMSE over successful chunks, the
all-trial median (mR) and SWA, the Bonn cells also pooled over the four sequences, and
then checks every SR against `expected.json` (41 cells × 2 readouts). Per-chunk CSVs
with both readouts go to `runs/`.

The printed values are these numbers of the paper:

| paper | PadLoc rows | runner cells |
|---|---|---|
| Table 1 | frozen, all three depth blocks: SR, R | `gibson/{mono,comp,unloc}`, frozen |
| Table 2 | frozen (10 cm): SWA, mR | `gibson/unloc`, frozen |
| Table 4 + Sec. 4.4 text | wheel per sequence and pooled: SR, sR; GT pooled SR; pooled mR and SWA | `bonn_unloc/{wheel,GT}:*`, frozen |
| Tables S6, S7, S9 + Sec. H text | delayed: the same rows under the chunk-end readout | the same cells, chunk_end |
| Table S10 (top) | deployed-σ modularity rows: SR, mR | `bonn_modular/*`, frozen |

Not reproduced here: PadLoc accum and the MapAnything block (Table 1), the 5 cm
configuration (Table 2), the ablations (Tables 3, S4, S8), the 16-ray control (Table 4),
Hilti (Table 5; its MapAnything reconstructions are not part of the fixtures) and the
supplement's sweeps.

**Backend.** The paper's numbers come from the torch backend on a CUDA GPU (verified on an
RTX 3090 with torch 2.12, CUDA 13: every chunk identical in both readouts; 2 h 23 min,
peak host memory 18 GB). The scipy backend computes the same formula but rounds differently
inside the FFT; about 1 % of chunks are near-ties that flip, so a CPU run can differ from
the published SR by a chunk or two per cell and from the RMSE / SWA columns in the last digit.

**Tests.** `pytest tests/` (no fixtures needed) checks the masked NCC against the direct
formula, the running sums against a re-sum, the frozen readout rule, the rasterizer gates and
the defaults.

**Competitor rows.** The F3Loc and UnLoc rows come from their released evaluators.
`docs/competitor_readout.md` documents the per-frame (main tables) and chunk-end (supplement)
readouts, the dump patches and drivers in `tools/`, the scorer
`tools/competitor_chunk_end.py`, its validation and the resulting numbers.

## Citation

```bibtex
@inproceedings{bookhahn2027padloc,
  title     = {{PadLoc}: Training-Free Floor-Plan Localization by Sparse Geometric Matching},
  author    = {Bookhahn, Marian and Neumann, Frank},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

MIT (code). The fixture package redistributes inputs derived from the Gibson
`gibson_t` split of F3Loc, from F3Loc's and UnLoc's depth networks and from Bonn
LiLocBench; their own licenses apply to that data.
