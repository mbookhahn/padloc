#!/bin/bash
# F3Loc native filter on gibson_t with per-chunk pose dumps (reference driver; the paper's runs used
# it on the authors' GPU host). Runs F3Loc's own eval_filtering.py with the paper's command
# (--net_type d = mono, comp_s; T = 15/20/35/50/100) plus --save_pred_dir, so the forward filter can
# be scored under the chunk-end readout:
#   python tools/competitor_chunk_end.py --layout f3loc --dumps $OUT --archive $OUT
# (the per-chunk logs $OUT/{net}_T{T}.log are the per-frame archive the scorer validates against).
# Requires tools/f3loc_pose_dump_patch.py applied to $F3LOC/eval_filtering.py.
set -uo pipefail
F3LOC=${F3LOC:?"F3Loc checkout (github.com/felix-ch/f3loc) with data/ and logs/ (mono, mv, comp checkpoints)"}
PYTHON=${PYTHON:-python}                                   # interpreter of F3Loc's environment
DATASET_PATH=${DATASET_PATH:-./data/Gibson Floorplan Localization Dataset}   # relative to $F3LOC
OUT=${OUT:?"output root, e.g. runs/f3loc_fwd_dump_<date>"}
read -ra NETS  <<< "${NETS:-d comp_s}"
read -ra TLENS <<< "${TLENS:-15 20 35 50 100}"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$F3LOC"
for net in "${NETS[@]}"; do for T in "${TLENS[@]}"; do
  [ -s "$OUT/${net}_T${T}.done" ] && continue
  "$PYTHON" -u eval_filtering.py --net_type "$net" --traj_len "$T" \
    --dataset_path "$DATASET_PATH" --ckpt_path ./logs \
    --save_pred_dir "$OUT/${net}_T${T}" \
    > "$OUT/${net}_T${T}.log" 2>&1
  echo "exit=$? $(date -Is)" > "$OUT/${net}_T${T}.done"
done; done
echo "ALL_DONE $(date -Is)" | tee "$OUT/ALL_DONE"
