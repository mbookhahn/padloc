#!/bin/bash
# UnLoc on Bonn (LiLocBench), the paper's PUBLISHED configuration (cameras-only DESDF, measured
# structural depth mu with UnLoc's network b-hat, GT-motion transitions, DESDF 0.1 m, 72
# orientation bins, dt 1), re-run FORWARD-ONLY with per-trial pose dumps (--dump-poses-dir) so
# the forward filter can be scored under the chunk-end readout (tools/competitor_chunk_end.py
# --layout bonn). Reference driver: the Bonn dataset path and the --use-measured-depth /
# --measured-depth-uncertainty-* flags belong to the paper's extension of UnLoc's evaluate.py
# (not in upstream UnLoc, not part of this release); see docs/competitor_readout.md.
# Writes to a NEW results root; the published results root is untouched.
# Optionally waits for a marker file first (WAIT_FOR=...), to chain behind another GPU run.
set -uo pipefail
UNLOC=${UNLOC:?"UnLoc checkout with the paper's Bonn extension of evaluate.py"}
PYTHON=${PYTHON:-python}                                   # interpreter of UnLoc's environment
SU=${SU:?"prepared Bonn tree (<SU>/<seq>/..., <SU>/metadata.json)"}
DESDF=${DESDF:?"cameras-only DESDF .npy shared by all sequences"}
CKPT=${CKPT:-logs/unloc_gibson_vitl.ckpt}                  # relative to $UNLOC
TMPROOT=${TMPROOT:-/tmp}                                   # per-sequence symlink trees
read -ra SEQS  <<< "${SEQS:-static_0 dynamics_0 lt_changes_0 lt_changes_dynamics_0}"
read -ra TLENS <<< "${TLENS:-150 300 450}"
JOBS=${JOBS:-2}
ROOT_OUT=${ROOT_OUT:-results/bonn/unloc_C_bhat_network_fwd_dump}   # relative to $UNLOC
WAIT_FOR=${WAIT_FOR:-}
cd "$UNLOC"
if [ -n "$WAIT_FOR" ]; then
  echo "waiting for $WAIT_FOR ..."; while [ ! -f "$WAIT_FOR" ]; do sleep 60; done; echo "marker present $(date -Is)"
fi
mkdir -p "$ROOT_OUT"; echo "start $(date -Is) SEQS=${SEQS[*]} TLENS=${TLENS[*]} JOBS=$JOBS" >> "$ROOT_OUT/launch.log"
declare -A TREE
for s in "${SEQS[@]}"; do
  TR=$TMPROOT/sU_Cbhat_dump_${s}; rm -rf "$TR"; mkdir -p "$TR/bonn/prepared/$s"
  for f in "$SU/$s"/*; do bn=$(basename "$f"); [ "$bn" = "desdf.npy" ] && continue
    ln -sf "$f" "$TR/bonn/prepared/$s/$bn"; done
  ln -sf "$DESDF" "$TR/bonn/prepared/$s/desdf.npy"
  ln -sf "$SU/metadata.json" "$TR/bonn/prepared/metadata.json"
  printf "test:\n- %s\n" "$s" > "$TR/bonn/prepared/split.yaml"
  TREE[$s]="$TR"
done
run_one () {  # $1=scene $2=T
  local s=$1 T=$2 OUT
  OUT=$ROOT_OUT/${s}_T${T}; [ -s "$OUT/run.log" ] && grep -q "Success @" "$OUT/run.log" && { echo "skip $s T$T"; return; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  "$PYTHON" -u evaluate.py --dataset_path "${TREE[$s]}" --dataset bonn \
    --checkpoint_path "$CKPT" --traj_len "$T" --dt 1 \
    --desdf_resolution 0.1 --orn_slice 72 \
    --use-measured-depth --measured-depth-uncertainty-m 0.05 \
    --measured-depth-uncertainty-mode network \
    --no-postprocess --dump-poses-dir "$OUT/poses" \
    --output_dir "$OUT" > "$OUT/run.log" 2>&1 \
    && echo "done $s T$T $(date -Is)" >> "$ROOT_OUT/launch.log" || { echo "FAIL $s T$T" >> "$ROOT_OUT/launch.log"; tail -4 "$OUT/run.log" >> "$ROOT_OUT/launch.log"; }
}
for T in "${TLENS[@]}"; do for s in "${SEQS[@]}"; do
  run_one "$s" "$T" &
  while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do wait -n; done
done; done
wait
for T in "${TLENS[@]}"; do for s in "${SEQS[@]}"; do
  L=$ROOT_OUT/${s}_T${T}/run.log
  echo "$s T$T: n=$(grep -m1 'Test set size' "$L" 2>/dev/null | awk '{print $NF}') fwd@1m=$(grep -m1 'Success @  1.0' "$L" 2>/dev/null | awk '{print $NF}')" >> "$ROOT_OUT/launch.log"
done; done
touch "$ROOT_OUT/DONE"; echo "ALL_DONE $ROOT_OUT $(date -Is)" >> "$ROOT_OUT/launch.log"
