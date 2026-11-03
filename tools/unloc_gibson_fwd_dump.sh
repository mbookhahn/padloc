#!/bin/bash
# UnLoc FORWARD-ONLY run on Gibson with per-trial pose dumps (reference driver; the paper's
# runs used it on the authors' GPU host). Runs UnLoc's own evaluate.py with --no-postprocess
# (forward Bayes filter, no SE(2) refinement = "UnLoc fwd") on gibson_t for each T in the
# published configuration (DESDF 0.1 m, 36 orientation bins, dt 1), writing one NPZ per trial
# (pred + GT x/y/theta in map px, last-10 RMSE, transitions) to $OUT/poses_T<T>/ through the
# --dump-poses-dir block described in docs/competitor_readout.md.
# Score with: python tools/competitor_chunk_end.py --layout gibson --dumps $OUT --archive <published results_T<T>.npz dir>
# Sanity: the forward SR@1m grepped per T must reproduce UnLoc's published forward numbers.
set -euo pipefail
UNLOC=${UNLOC:?"UnLoc checkout (evaluate.py with the --dump-poses-dir block)"}
PYTHON=${PYTHON:-python}                                   # interpreter of UnLoc's environment
DATASET_PATH=${DATASET_PATH:?"the 'Gibson Floorplan Localization Dataset' directory"}
CKPT=${CKPT:-$UNLOC/logs/unloc_gibson_vitl.ckpt}
read -ra TLENS <<< "${TLENS:-15 20 35 50 100}"
OUT=${OUT:?"output root, e.g. runs/unloc_fwd_dump_<date>"}
mkdir -p "$OUT"
cd "$UNLOC"
export PYTHONPATH="$UNLOC:${PYTHONPATH:-}"
echo "OUT=$OUT  T=${TLENS[*]}  ckpt=$CKPT"
for T in "${TLENS[@]}"; do
  echo "=== T=$T (forward only) ==="
  "$PYTHON" -u evaluate.py \
    --dataset_path "$DATASET_PATH" \
    --checkpoint_path "$CKPT" \
    --dataset gibson_t \
    --traj_len "$T" \
    --no-postprocess \
    --no-save_results \
    --dump-poses-dir "$OUT/poses_T${T}" \
    --output_dir "$OUT" > "$OUT/T${T}.log" 2>&1
  echo "T=$T done -> $(grep -c . "$OUT/poses_T${T}"/*.npz 2>/dev/null | wc -l) trials"
  grep -E "Success @  1.0|Success @  0.5|RMSE \(all\)|Test set size" "$OUT/T${T}.log" || true
done
touch "$OUT/DONE"
echo "ALL_DONE $OUT"
