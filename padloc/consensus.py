"""Stage 8b — consensus readout: heal, cyclic shift-and-sum, eligibility, anchor.

The filter's MAP sequence x_hat sets only the *relative* cyclic shifts between
windows (healed back to odometry where a step departs from it by more than tau);
the raw centred fields are shifted into the target window's frame and summed. Only
cells covered by at least half the windows may win. The peak places the target
window's centroid; subtracting its odometry gives the chunk anchor.

Every window votes on the same frame-0 anchor, so the sum is kept incrementally
(``accum="running"``, the published readout): with g_w the grid cell of x_hat_w,
window w is shifted once by -g_w into one anchor-space running sum (and count), a
read shifts the sum by +g_N of its target window N, and a bounded past subtracts
expired windows. On the cyclic grid this equals re-summing every window with
offsets g_w - g_N, bit for bit. ``accum="resum"`` re-sums every window at each read
with offsets round((x_hat_w - x_hat_N) / cell), the re-summed consensus the supplement
compares against (one rounding per offset, within half a cell; g_w - g_N is within one).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Field = Tuple[np.ndarray, np.ndarray]   # (centroid-indexed NCC field, valid mask)
Peak = Tuple[int, int, float, float]    # (row, col, score, psr)
ToCell = Callable[[float, float], Tuple[float, float]]   # world (x, y) m -> fractional (row, col)


def heal(xh: Sequence[Tuple[float, float]], cens: Sequence[Tuple[float, float]], max_jump_m: float) -> List[Tuple[float, float]]:
    """Rebuild positions from increments, replacing any increment that departs from the
    odometry increment by more than ``max_jump_m`` with the odometry increment (Eq. heal)."""
    if max_jump_m <= 0.0 or len(xh) <= 1:
        return [(float(a), float(b)) for a, b in xh]
    out = [(float(xh[0][0]), float(xh[0][1]))]
    for j in range(1, len(xh)):
        dxe = (float(xh[j][0]) - float(xh[j - 1][0]), float(xh[j][1]) - float(xh[j - 1][1]))
        dvo = (float(cens[j][0]) - float(cens[j - 1][0]), float(cens[j][1]) - float(cens[j - 1][1]))
        d = dvo if float(np.hypot(dxe[0] - dvo[0], dxe[1] - dvo[1])) > max_jump_m else dxe
        out.append((out[-1][0] + d[0], out[-1][1] + d[1]))
    return out


def _centred(field: Field, h_ref: int, w_ref: int) -> np.ndarray:
    imc, vc = field
    c = np.zeros((h_ref, w_ref), dtype=np.float32)
    c[vc] = imc[vc] - float(imc[vc].mean())
    return c


def _select(S: np.ndarray, cnt: np.ndarray, n_windows: int) -> Optional[Peak]:
    """Coverage-gated argmax: (row, col, score, psr) of the eligible peak, None if none."""
    ok = cnt >= max(1, int(round(0.5 * n_windows)))
    if not ok.any():
        return None
    sv = S[ok]
    score = float(sv.max())
    psr = score / (float(sv.std()) + 1e-06)
    i = int(np.argmax(np.where(ok, S, -np.inf)))
    pr, pc = divmod(i, S.shape[1])
    return pr, pc, score, psr


class RunningConsensus:
    """Anchor-space running consensus sum: ``add`` is one shift-and-add per window,
    ``read`` one shift of the sum onto the target window. ``remove`` subtracts a window
    added earlier by recomputing its shifted field (float32 rounding may then differ
    from a re-sum in the last bits)."""

    def __init__(self, ref_shape: Tuple[int, int], to_cell: ToCell) -> None:
        self.h_ref, self.w_ref = ref_shape
        self.to_cell = to_cell
        self.A = np.zeros(ref_shape, dtype=np.float32)
        self.C = np.zeros(ref_shape, dtype=np.int32)
        self.n = 0

    def cell(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        r, c = self.to_cell(float(xy[0]), float(xy[1]))
        return int(round(r)), int(round(c))

    def _shifted(self, field: Field, xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        g = self.cell(xy)
        cr = np.roll(_centred(field, self.h_ref, self.w_ref), shift=(-g[0], -g[1]), axis=(0, 1))
        mr = np.roll(field[1], shift=(-g[0], -g[1]), axis=(0, 1)).astype(np.int32)
        return cr, mr

    def add(self, field: Field, xy: Tuple[float, float]) -> None:
        cr, mr = self._shifted(field, xy)
        self.A += cr
        self.C += mr
        self.n += 1

    def remove(self, field: Field, xy: Tuple[float, float]) -> None:
        cr, mr = self._shifted(field, xy)
        self.A -= cr
        self.C -= mr
        self.n -= 1

    def read(self, xy: Tuple[float, float]) -> Optional[Peak]:
        g = self.cell(xy)
        return _select(np.roll(self.A, g, axis=(0, 1)), np.roll(self.C, g, axis=(0, 1)), self.n)


def _resum_peak(xh: Sequence[Tuple[float, float]], fields: Sequence[Field], *, cell_m: float, ref_shape: Tuple[int, int],
                target_idx: int) -> Optional[Peak]:
    h_ref, w_ref = ref_shape
    xl = xh[target_idx]
    S = cnt = None
    for f, (mx, my) in zip(fields, xh):
        orow = int(round((my - xl[1]) / float(cell_m)))
        ocol = int(round((mx - xl[0]) / float(cell_m)))
        cr = np.roll(_centred(f, h_ref, w_ref), shift=(-orow, -ocol), axis=(0, 1))
        mr = np.roll(f[1], shift=(-orow, -ocol), axis=(0, 1))
        S = cr if S is None else S + cr
        cnt = mr.astype(np.int32) if cnt is None else cnt + mr.astype(np.int32)
    return _select(S, cnt, len(fields))


def consensus_peak(xh: Sequence[Tuple[float, float]], fields: Sequence[Field], *, cell_m: float, ref_shape: Tuple[int, int],
                   to_cell: ToCell, accum: str = "running", target_idx: int = -1) -> Optional[Peak]:
    """Shift every field so its x_hat coincides with x_hat[target_idx], sum the centred
    fields, and return (row, col, score, psr) of the eligible peak (None if nothing eligible)."""
    if not xh:
        return None
    if accum == "resum":
        return _resum_peak(xh, fields, cell_m=cell_m, ref_shape=ref_shape, target_idx=target_idx)
    if accum != "running":
        raise ValueError(f"consensus accum {accum!r}: running|resum")
    rc = RunningConsensus(ref_shape, to_cell)
    for f, xy in zip(fields, xh):
        rc.add(f, xy)
    return rc.read(xh[target_idx])


def streaming_peaks(xh: Sequence[Tuple[float, float]], fields: Sequence[Field], *, lag_L: int, back_windows: int,
                    cell_m: float, ref_shape: Tuple[int, int], to_cell: ToCell,
                    accum: str = "running") -> List[Optional[Peak]]:
    """Bounded-lag readout: window w is decoded from windows [w - back, w + L] (-1 = unbounded)."""
    n = len(xh)
    out: List[Optional[Peak]] = []
    rc = RunningConsensus(ref_shape, to_cell) if accum == "running" else None
    n_in = n_out = 0   # windows added to / removed from the running sum
    for w in range(n):
        lo = 0 if back_windows < 0 else max(0, w - back_windows)
        hi = n if lag_L < 0 else min(n, w + lag_L + 1)
        if rc is None:
            out.append(consensus_peak(xh[lo:hi], fields[lo:hi], cell_m=cell_m, ref_shape=ref_shape, to_cell=to_cell,
                                      accum=accum, target_idx=w - lo))
            continue
        while n_in < hi:
            rc.add(fields[n_in], xh[n_in])
            n_in += 1
        while n_out < lo:
            rc.remove(fields[n_out], xh[n_out])
            n_out += 1
        out.append(rc.read(xh[w]))
    return out
