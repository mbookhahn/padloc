"""Add --save_pred_dir to F3Loc's eval_filtering.py (github.com/felix-ch/f3loc @ 9e8027d).

    python tools/f3loc_pose_dump_patch.py /path/to/f3loc/eval_filtering.py

Per chunk, after the t-loop, the patched script writes <save_pred_dir>/<scene>_chunk<NN>.npz with
    pred_poses_map (traj_l-L, 3)  forward MAP pose per filtered frame, map px / rad
    gt_poses_map, poses_map (traj_l, 3)  GT in map px / rad (same array under two keys)
    transitions (traj_l-1, 3)  ego-frame GT transition k->k+1 via the filter's own get_rel_pose (m, rad)
    resolution  0.01 m/px, L  3 (the filter emits no pose for the first L frames)
Scored by tools/competitor_chunk_end.py --layout f3loc. The patch only adds the dump; the
filter and its per-frame log are unchanged. Idempotent: a marker line guards re-patching.
"""
import argparse, sys
from pathlib import Path

MARKER = "# --- PadLoc pose-dump patch (chunk-end readout of the F3Loc filter) ---"

ARG_ANCHOR = '    args = parser.parse_args()\n'
ARG_BLOCK = '''    %s
    parser.add_argument(
        "--save_pred_dir", type=str, default=None,
        help="If set, dump per-chunk pred_poses_map / gt poses_map / GT transitions as "
             "<scene>_chunk<NN>.npz (chunk-end readout scoring).",
    )
    args = parser.parse_args()
''' % MARKER

DUMP_ANCHOR = '        if log_error:\n            pred_poses_map = np.stack(pred_poses_map)\n'
DUMP_BLOCK = '''        %s
        if args.save_pred_dir is not None:
            import os as _os
            _os.makedirs(args.save_pred_dir, exist_ok=True)
            _p = poses[0]
            _tr = torch.stack([get_rel_pose(_p[_k], _p[_k + 1]) for _k in range(_p.shape[0] - 1)])
            np.savez(
                _os.path.join(args.save_pred_dir, f"{scene}_chunk{idx_within_scene:02d}.npz"),
                pred_poses_map=np.stack(pred_poses_map).astype(np.float32),
                gt_poses_map=np.asarray(poses_map, dtype=np.float32),
                poses_map=np.asarray(poses_map, dtype=np.float32),
                transitions=_tr.detach().cpu().numpy().astype(np.float32),
                resolution=np.float32(0.01), L=np.int64(L),
            )
        if log_error:
            pred_poses_map = np.stack(pred_poses_map)
''' % MARKER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    a = ap.parse_args()
    src = a.path.read_text()
    if MARKER in src:
        print("already patched"); return 0
    for anchor in (ARG_ANCHOR, DUMP_ANCHOR):
        if src.count(anchor) != 1:
            print(f"anchor not unique ({src.count(anchor)}): {anchor!r}"); return 1
    src = src.replace(ARG_ANCHOR, ARG_BLOCK).replace(DUMP_ANCHOR, DUMP_BLOCK)
    a.path.write_text(src)
    print("patched", a.path); return 0


if __name__ == "__main__":
    sys.exit(main())
