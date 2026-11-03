"""Stage 5 — Padfield masked normalized cross-correlation via FFT.

One reference (R, M_R), many targets (T, M): every term of the masked NCC is one
FFT correlation, so a full translation field costs O(HW log HW). The reference
spectra are precomputed once per target size (``precompute_reference_cache``).

Implemented score (paper Eqs. 1-2). R is the premasked reference (the blurred
distance field, zero outside its interior mask M_R); the target pair is T (1 at
wall hits) and M (``wall_weight`` at wall hits, 1 in observed free space, 0
unobserved), and the target enters premasked, T~ = M * T. With corr(a, c) the
FFT cross-correlation, the field at shift p is

    NCC = [ corr(R, T~) - corr(R, M) corr(M_R, T~) / N ] / sqrt(D1 D2)
    N   = corr(M_R, M)                                 weighted overlap mass
    D1  = corr(R^2, M)    - corr(R, M)^2 / N           reference side
    D2  = corr(M_R, M T^2) - corr(M_R, T~)^2 / N       target side

D2 uses the exact weighted second moment M T^2 (``d2_exact=True``, the default and
the paper's score). ``d2_exact=False`` forms it from the squared premasked target,
T~^2 = M^2 T^2 (``gm * gm``), i.e. Padfield's Algorithm 1 applied unchanged to the
premasked target (supplement, "Weighted masked NCC"). Shifts with N < min_overlap * max N are rejected; N carries a
+0.001 floor and each variance a 1e-6 N floor (score 0 below it).

Backends: ``scipy`` (default on CPU) and ``torch`` (GPU; identical formula; also
batched over many targets). Select with ``PADLOC_FFT_BACKEND=scipy|torch|auto``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2

_BACKEND = os.environ.get("PADLOC_FFT_BACKEND", "auto").lower()
try:
    import torch
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAVE_TORCH = False
if _BACKEND == "auto":
    _BACKEND = "torch" if (_HAVE_TORCH and torch.cuda.is_available()) else "scipy"
if _BACKEND == "torch" and not _HAVE_TORCH:
    _BACKEND = "scipy"


def backend() -> str:
    return _BACKEND


def _workers() -> int:
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        return int(n) if n is not None else -1
    except ValueError:
        return -1


def _pad_to(x: np.ndarray, shape: Tuple[int, int], dtype) -> np.ndarray:
    out = np.zeros(shape, dtype=dtype)
    out[: x.shape[0], : x.shape[1]] = x
    return out


# ---------------------------------------------------------------------------- numpy / scipy
@dataclass(frozen=True)
class ReferenceCache:
    F1: object          # rfft2(R * M_R)
    M1: object          # rfft2(M_R)
    F1M: object         # rfft2((R * M_R)^2)
    padded_shape: Tuple[int, int]
    f_shape: Tuple[int, int]
    backend: str


def _cache_numpy(f: np.ndarray, mf: np.ndarray, g_shape: Tuple[int, int], dtype=np.float32) -> ReferenceCache:
    f = np.asarray(f, dtype=dtype)
    mf = np.asarray(mf, dtype=dtype)
    H = next_fast_len(f.shape[0] + g_shape[0] - 1)
    W = next_fast_len(f.shape[1] + g_shape[1] - 1)
    fm = f * mf
    w = _workers()
    return ReferenceCache(F1=rfft2(_pad_to(fm, (H, W), dtype), workers=w),
                          M1=rfft2(_pad_to(mf, (H, W), dtype), workers=w),
                          F1M=rfft2(_pad_to(fm * fm, (H, W), dtype), workers=w),
                          padded_shape=(H, W), f_shape=tuple(f.shape), backend="scipy")


def _ncc_numpy(cache: ReferenceCache, g: np.ndarray, mg: np.ndarray, *, min_overlap: float,
               return_overlap: bool, d2_exact: bool = True, dtype=np.float32):
    H, W = cache.padded_shape
    F1, M1, F1M = cache.F1, cache.M1, cache.F1M
    w = _workers()
    g_arr = np.asarray(g, dtype=dtype)
    mg_arr = np.asarray(mg, dtype=dtype)
    gm = _pad_to(g_arr * mg_arr, (H, W), dtype)
    mgp = _pad_to(mg_arr, (H, W), dtype)
    F2 = np.conj(rfft2(gm, workers=w))
    M2 = np.conj(rfft2(mgp, workers=w))
    F2M = np.conj(rfft2(_pad_to(g_arr * g_arr * mg_arr, (H, W), dtype) if d2_exact else gm * gm, workers=w))
    D = np.real(irfft2(M1 * M2, s=(H, W), workers=w)) + 0.001
    valid = D >= min_overlap * np.max(D)
    F1M2 = np.real(irfft2(F1 * M2, s=(H, W), workers=w))
    M1F2 = np.real(irfft2(M1 * F2, s=(H, W), workers=w))
    D_v, F1M2_v, M1F2_v = D[valid], F1M2[valid], M1F2[valid]
    den1 = np.clip(np.real(irfft2(F1M * M2, s=(H, W), workers=w))[valid] - F1M2_v ** 2 / D_v, 0, None)
    den2 = np.clip(np.real(irfft2(M1 * F2M, s=(H, W), workers=w))[valid] - M1F2_v ** 2 / D_v, 0, None)
    num_v = np.real(irfft2(F1 * F2, s=(H, W), workers=w))[valid] - F1M2_v * M1F2_v / D_v
    den_v = np.sqrt(den1 * den2)
    var_floor = 1e-06 * D_v
    well = (den1 > var_floor) & (den2 > var_floor)
    ncc = np.zeros_like(D)
    ncc[valid] = np.where(well, num_v / np.maximum(den_v, 1e-12), 0.0)
    return (ncc, D - 0.001) if return_overlap else ncc


# ---------------------------------------------------------------------------- torch
def _torch_fast_len(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _cache_torch(f: np.ndarray, mf: np.ndarray, g_shape: Tuple[int, int]) -> ReferenceCache:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    H = _torch_fast_len(f.shape[0] + g_shape[0] - 1)
    W = _torch_fast_len(f.shape[1] + g_shape[1] - 1)
    f_t = torch.from_numpy(np.ascontiguousarray(f)).to(device=device, dtype=dt)
    mf_t = torch.from_numpy(np.ascontiguousarray(mf)).to(device=device, dtype=dt)
    fm = f_t * mf_t

    def pad(x):
        out = torch.zeros((H, W), dtype=dt, device=device)
        out[: x.shape[0], : x.shape[1]] = x
        return out

    return ReferenceCache(F1=torch.fft.rfft2(pad(fm)), M1=torch.fft.rfft2(pad(mf_t)),
                          F1M=torch.fft.rfft2(pad(fm * fm)), padded_shape=(H, W),
                          f_shape=tuple(int(s) for s in f.shape), backend="torch")


def _ncc_torch_single(cache: ReferenceCache, g: np.ndarray, mg: np.ndarray, *, min_overlap: float,
                      return_overlap: bool, d2_exact: bool = True):
    """Single-target torch path — op-for-op the implementation the published tables used."""
    H, W = cache.padded_shape
    F1, M1, F1M = cache.F1, cache.M1, cache.F1M
    device, dt = F1.device, torch.float32

    def pad(x):
        out = torch.zeros((H, W), dtype=dt, device=device)
        out[: x.shape[0], : x.shape[1]] = x
        return out

    g_t = torch.from_numpy(np.ascontiguousarray(g)).to(device=device, dtype=dt)
    mg_t = torch.from_numpy(np.ascontiguousarray(mg)).to(device=device, dtype=dt)
    gm = g_t * mg_t
    F2 = torch.conj(torch.fft.rfft2(pad(gm)))
    M2 = torch.conj(torch.fft.rfft2(pad(mg_t)))
    F2M = torch.conj(torch.fft.rfft2(pad((mg_t * g_t * g_t) if d2_exact else (gm * gm))))
    D = torch.real(torch.fft.irfft2(M1 * M2, s=(H, W))) + 0.001
    valid = D >= float(min_overlap) * torch.max(D)
    F1M2 = torch.real(torch.fft.irfft2(F1 * M2, s=(H, W)))
    M1F2 = torch.real(torch.fft.irfft2(M1 * F2, s=(H, W)))
    den1 = torch.real(torch.fft.irfft2(F1M * M2, s=(H, W))) - F1M2 * F1M2 / D
    den2 = torch.real(torch.fft.irfft2(M1 * F2M, s=(H, W))) - M1F2 * M1F2 / D
    den1 = torch.clamp(den1, min=0.0)
    den2 = torch.clamp(den2, min=0.0)
    num = torch.real(torch.fft.irfft2(F1 * F2, s=(H, W))) - F1M2 * M1F2 / D
    den = torch.sqrt(den1 * den2)
    var_floor = 1e-06 * D
    well = (den1 > var_floor) & (den2 > var_floor)
    ncc = torch.where(valid & well, num / torch.clamp(den, min=1e-12), torch.zeros_like(D))
    if return_overlap:
        return ncc.cpu().numpy(), (D - 0.001).cpu().numpy()
    return ncc.cpu().numpy()


def _ncc_torch_batched(cache: ReferenceCache, grids: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                       *, min_overlap: float, return_overlap: bool, d2_exact: bool = True):
    H, W = cache.padded_shape
    F1, M1, F1M = cache.F1, cache.M1, cache.F1M
    device, dt = F1.device, torch.float32
    K = len(grids)
    G = torch.zeros((K, H, W), dtype=dt, device=device)
    MG = torch.zeros((K, H, W), dtype=dt, device=device)
    GW = torch.zeros((K, H, W), dtype=dt, device=device) if d2_exact else None
    for i in range(K):
        g_t = torch.from_numpy(np.ascontiguousarray(grids[i])).to(device=device, dtype=dt)
        m_t = torch.from_numpy(np.ascontiguousarray(masks[i])).to(device=device, dtype=dt)
        hh, ww = g_t.shape
        G[i, :hh, :ww] = g_t * m_t
        MG[i, :hh, :ww] = m_t
        if GW is not None:
            GW[i, :hh, :ww] = m_t * g_t * g_t
    F2 = torch.conj(torch.fft.rfft2(G))
    M2 = torch.conj(torch.fft.rfft2(MG))
    F2M = torch.conj(torch.fft.rfft2(GW if GW is not None else G * G))
    D = torch.real(torch.fft.irfft2(M1 * M2, s=(H, W))) + 0.001
    valid = D >= float(min_overlap) * D.amax(dim=(-2, -1), keepdim=True)
    F1M2 = torch.real(torch.fft.irfft2(F1 * M2, s=(H, W)))
    M1F2 = torch.real(torch.fft.irfft2(M1 * F2, s=(H, W)))
    den1 = torch.clamp(torch.real(torch.fft.irfft2(F1M * M2, s=(H, W))) - F1M2 * F1M2 / D, min=0.0)
    den2 = torch.clamp(torch.real(torch.fft.irfft2(M1 * F2M, s=(H, W))) - M1F2 * M1F2 / D, min=0.0)
    num = torch.real(torch.fft.irfft2(F1 * F2, s=(H, W))) - F1M2 * M1F2 / D
    den = torch.sqrt(den1 * den2)
    var_floor = 1e-06 * D
    well = (den1 > var_floor) & (den2 > var_floor)
    ncc = torch.where(valid & well, num / torch.clamp(den, min=1e-12), torch.zeros_like(D))
    out = ncc.cpu().numpy()
    if return_overlap:
        return out, (D - 0.001).cpu().numpy()
    return out


# ---------------------------------------------------------------------------- public API
def precompute_reference_cache(field: np.ndarray, mask: np.ndarray, target_shape: Tuple[int, int]) -> ReferenceCache:
    if _BACKEND == "torch":
        return _cache_torch(field, mask, target_shape)
    return _cache_numpy(field, mask, target_shape)


def masked_ncc(cache: ReferenceCache, target: np.ndarray, mask: np.ndarray, *, min_overlap: float,
               return_overlap: bool = False, d2_exact: bool = True):
    """Masked NCC field of one target against the cached reference.

    Returns the (H, W)-padded correlation field (and the overlap mass N(p) if asked);
    shift (0, 0) is at index (0, 0), i.e. the target's top-left on the reference's
    top-left — see ``windows.place_field`` for re-indexing to the map frame.
    """
    if cache.backend == "torch":
        return _ncc_torch_single(cache, target, mask, min_overlap=min_overlap, return_overlap=return_overlap, d2_exact=d2_exact)
    return _ncc_numpy(cache, target, mask, min_overlap=min_overlap, return_overlap=return_overlap, d2_exact=d2_exact)


def masked_ncc_batched(cache: ReferenceCache, targets: Sequence[np.ndarray], masks: Sequence[np.ndarray], *,
                       min_overlap: float, return_overlap: bool = False, d2_exact: bool = True) -> List:
    """Masked NCC of many targets; one batched FFT on torch, a loop on scipy."""
    if cache.backend == "torch":
        out = _ncc_torch_batched(cache, targets, masks, min_overlap=min_overlap, return_overlap=return_overlap, d2_exact=d2_exact)
        if return_overlap:
            return [(out[0][i], out[1][i]) for i in range(len(targets))]
        return [out[i] for i in range(len(targets))]
    return [_ncc_numpy(cache, t, m, min_overlap=min_overlap, return_overlap=return_overlap, d2_exact=d2_exact)
            for t, m in zip(targets, masks)]
