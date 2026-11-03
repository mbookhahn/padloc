# Release notes

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

**Fixture package** (`padloc-fixtures-v1.0.0.tar.gz`, 98 MB, 290 MB extracted): the inputs of
the reproduction — per-frame depth columns (F3Loc mono / comp_s, UnLoc DepthAnything-V2
per-column μ / b̂), Bonn 2-D LiDAR / D455 columns, odometry trajectories, floor-plan rasters
and poses for the 9 Gibson scenes and 4 Bonn LiLocBench sequences. Depth networks are not
shipped; their outputs are the fixtures.

```bash
tar xzf padloc-fixtures-v1.0.0.tar.gz                              # -> fixtures/ in the repo root
PADLOC_FFT_BACKEND=torch python examples/reproduce_tables.py       # all cells
```

sha256: `e2d5fb9a9311f9a441c26cc3b953dec9a97252a22745fb5649bbe185e5db89e7`
