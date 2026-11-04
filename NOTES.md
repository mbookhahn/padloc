# Release notes

## v1.0.1

The Gibson floor plans and poses are no longer redistributed; the numbers are unchanged.

- **Fixture package** `padloc-fixtures-v1.0.1.tar.gz` (96 MB, 288 MB extracted, 99 files):
  v1.0.0 without F3Loc's `map.png` and `poses.txt` of the nine Gibson scenes and without the
  unused `gt_poses_world` array of the UnLoc dumps, plus `fixtures/DATA_LICENSE.md`. Every other
  file and array is identical.
- `examples/install_gibson_inputs.py` installs the Gibson floor plans and poses from F3Loc's
  per-scene zips (or extracted scenes) and checks their SHA-256; `reproduce_tables.py` points to
  it when they are missing.
- `DATA_LICENSE.md`: the fixtures and `docs/padloc_method.gif` are CC BY-NC-SA 4.0, with the
  sources, our changes and the Gibson Database of Spaces terms.

```bash
tar xzf padloc-fixtures-v1.0.1.tar.gz                              # -> fixtures/ in the repo root
python examples/install_gibson_inputs.py f3loc                     # F3Loc's 9 scene zips (README)
PADLOC_FFT_BACKEND=torch python examples/reproduce_tables.py       # all cells
```

sha256: `e46589a400c57a707be661295aa917c0e8bb483f0001b0d21bc4aeb2c3621cdc`

## v1.0.0

Reference implementation of PadLoc (WACV 2027) and the reproduction of the paper's PadLoc
numbers on Gibson and Bonn (LiLocBench).

- `padloc/`: the back-end of the paper — W = S = 5 fusion and pre-pass windows (`BATCH30`
  for 30-frame batch front-ends), weighted masked NCC with the exact second moment, moving-pose
  GM-BF, running consensus sums, causal (frozen) and chunk-end readouts.
- `examples/reproduce_tables.py`: every PadLoc consensus cell of main Tables 1, 2 and 4 and
  supplement Tables S6, S7, S9 and S10, under both readouts; the SR of all 41 cells × 2
  readouts is checked against `expected.json`.
- `tools/`: the competitor readouts — dump patch / drivers for UnLoc and F3Loc and
  `competitor_chunk_end.py`, which scores their forward filters under the chunk-end readout
  (`docs/competitor_readout.md`).
- `tests/`: fixture-free unit tests.

**Fixture package** (`padloc-fixtures-v1.0.0.tar.gz`, 98 MB, 290 MB extracted; withdrawn,
replaced by v1.0.1): the inputs of the reproduction — per-frame depth columns (F3Loc mono /
comp_s, UnLoc DepthAnything-V2 per-column μ / b̂), Bonn 2-D LiDAR / D455 columns, odometry
trajectories, floor-plan rasters and poses for the 9 Gibson scenes and 4 Bonn LiLocBench
sequences. Depth networks are not shipped; their outputs are the fixtures.

```bash
tar xzf padloc-fixtures-v1.0.0.tar.gz                              # -> fixtures/ in the repo root
PADLOC_FFT_BACKEND=torch python examples/reproduce_tables.py       # all cells
```

sha256: `e2d5fb9a9311f9a441c26cc3b953dec9a97252a22745fb5649bbe185e5db89e7`
