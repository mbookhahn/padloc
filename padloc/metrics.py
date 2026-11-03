"""Evaluation metrics used in the paper."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def last10_errors(pred_xy: Sequence[Tuple[float, float]], gt_xy: Sequence[Tuple[float, float]]) -> np.ndarray:
    p, g = pred_xy[-10:], gt_xy[-10:]
    return np.asarray([math.hypot(px - gx, py - gy) for (px, py), (gx, gy) in zip(p, g)], dtype=np.float64)


def chunk_metrics(pred_xy: Sequence[Tuple[float, float]], gt_xy: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """last-10 RMSE and success@{1.0,0.5,0.3,0.2} m (all last-10 frames within threshold)."""
    errs = last10_errors(pred_xy, gt_xy)
    out = {"rmse_last10_m": float(np.sqrt(np.mean(errs ** 2))), "n_frames": len(pred_xy)}
    for thr in (1.0, 0.5, 0.3, 0.2):
        out[f"success_{thr:.1f}m".replace(".", "_")] = int(bool((errs < thr).all()))
    return out


def pooled(rows: Sequence[Dict[str, float]], tau_m: float = 1.0) -> Dict[str, float]:
    """Pooled last-10 SR@tau; last-10 RMSE as mean and median over successes and median over all; SWA."""
    if not rows:
        return {"n": 0}
    sr = np.asarray([r["success_1_0m"] for r in rows], dtype=np.float64)
    rm = np.asarray([r["rmse_last10_m"] for r in rows], dtype=np.float64)
    swa = float(np.mean(sr * np.clip(1.0 - rm / tau_m, 0.0, None)))
    return {"n": len(rows), "SR@1m": float(sr.mean()), "medRMSE_all": float(np.median(rm)),
            "meanRMSE_success": float(np.mean(rm[sr > 0])) if (sr > 0).any() else float("nan"),
            "medRMSE_success": float(np.median(rm[sr > 0])) if (sr > 0).any() else float("nan"), "SWA": swa}


def online_success(errors_m: Sequence[float], tau_m: float) -> float:
    e = np.asarray(errors_m, dtype=np.float64)
    return float((e < tau_m).mean()) if e.size else float("nan")
