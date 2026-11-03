"""Stage 8a — moving-pose Gaussian-mixture Bayes filter over the window centroid.

State: a bounded mixture {(log w, m, P)} of the current window centroid's absolute
map position. Per window: predict by the odometry centroid displacement with
isotropic process noise Q = (sigma * sqrt(stride) * cell)^2 I; gate observations
by squared Mahalanobis distance to the MAP component (P only — not P + Sigma, so
flat wrong-basin peaks do not become easier to accept); Gaussian-product update;
prune / merge / cap. The MAP mean is the window estimate x_hat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from .observe import Observation


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, x - mean)
    return -0.5 * (float(y @ y) + 2.0 * float(np.sum(np.log(np.diag(L)))) + 2.0 * np.log(2.0 * np.pi))


@dataclass
class Component:
    log_weight: float
    mean: np.ndarray
    cov: np.ndarray


def _normalize(tracks: List[Component]) -> None:
    if not tracks:
        return
    lse = float(logsumexp(np.array([t.log_weight for t in tracks])))
    if not np.isfinite(lse):
        u = -float(np.log(len(tracks)))
        for t in tracks:
            t.log_weight = u
        return
    for t in tracks:
        t.log_weight -= lse


class Mixture:
    """Bounded Gaussian mixture with the published predict / gated update / prune / merge / cap."""

    def __init__(self, tracks: Optional[List[Component]] = None) -> None:
        self.tracks: List[Component] = list(tracks) if tracks else []
        self.last_update_all_gated = False

    @classmethod
    def from_observations(cls, obs: Sequence[Observation]) -> "Mixture":
        m = cls([Component(float(lw), np.asarray(mu, dtype=np.float64).copy(), np.asarray(cov, dtype=np.float64).copy()) for lw, mu, cov in obs])
        _normalize(m.tracks)
        return m

    def predict(self, delta_xy: np.ndarray, Q: np.ndarray) -> None:
        for t in self.tracks:
            t.mean = t.mean + delta_xy
            t.cov = t.cov + Q

    def update(self, obs: Sequence[Observation], *, gate_d2: float, birth_log_c_threshold: float = -10.0,
               birth_log_weight: float = -4.6) -> None:
        self.last_update_all_gated = False
        if gate_d2 > 0.0 and self.tracks and obs:
            map_t = max(self.tracks, key=lambda t: t.log_weight)
            kept = []
            for ob in obs:
                diff = np.asarray(ob[1], dtype=np.float64) - map_t.mean
                try:
                    d2 = float(diff @ np.linalg.solve(map_t.cov, diff))
                except np.linalg.LinAlgError:
                    d2 = 0.0
                if d2 <= gate_d2:
                    kept.append(ob)
            self.last_update_all_gated = not kept
            obs = kept
        n_prev, n_obs = len(self.tracks), len(obs)
        assoc = np.full((n_prev, n_obs), -np.inf)
        for hi, h in enumerate(self.tracks):
            for ki, (_, mu_k, Sigma_k) in enumerate(obs):
                try:
                    assoc[hi, ki] = _mvn_logpdf(np.asarray(mu_k, dtype=np.float64), h.mean, h.cov + np.asarray(Sigma_k, dtype=np.float64))
                except np.linalg.LinAlgError:
                    pass
        new: List[Component] = []
        eye2 = np.eye(2)
        for hi, h in enumerate(self.tracks):
            for ki, (log_w_k, mu_k, Sigma_k) in enumerate(obs):
                log_c = float(assoc[hi, ki])
                if not np.isfinite(log_c):
                    continue
                mu_k = np.asarray(mu_k, dtype=np.float64)
                S = h.cov + np.asarray(Sigma_k, dtype=np.float64)
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    continue
                K = h.cov @ S_inv
                m_new = h.mean + K @ (mu_k - h.mean)
                P_new = (eye2 - K) @ h.cov
                P_new = 0.5 * (P_new + P_new.T)
                new.append(Component(h.log_weight + float(log_w_k) + log_c, m_new, P_new))
        if n_obs > 0:   # births for observations no component explains
            max_per_obs = assoc.max(axis=0) if n_prev > 0 else np.full(n_obs, -np.inf)
            for ki, (log_w_k, mu_k, Sigma_k) in enumerate(obs):
                if max_per_obs[ki] < birth_log_c_threshold:
                    new.append(Component(float(birth_log_weight) + float(log_w_k), np.asarray(mu_k, dtype=np.float64).copy(), np.asarray(Sigma_k, dtype=np.float64).copy()))
        if not new:
            return
        _normalize(new)
        self.tracks = new

    def prune(self, min_log_weight: float = -15.0) -> None:
        kept = [t for t in self.tracks if t.log_weight >= min_log_weight]
        if kept:
            _normalize(kept)
            self.tracks = kept

    def merge(self, d2_thresh: float = 4.0) -> None:
        while len(self.tracks) > 1:
            n = len(self.tracks)
            # Pair scoring is one batched solve; row-major upper-triangle order and
            # first-minimum argmin reproduce the paper runs' per-pair pick order exactly.
            iu, ju = np.triu_indices(n, k=1)
            means = np.stack([t.mean for t in self.tracks])
            covs = np.stack([t.cov for t in self.tracks])
            diffs = means[iu] - means[ju]
            try:
                sol = np.linalg.solve(covs[iu] + covs[ju], diffs[:, :, None])[:, :, 0]
                d2 = np.einsum("ki,ki->k", diffs, sol)
            except np.linalg.LinAlgError:   # a singular pair: score one by one, skipping it
                d2 = np.full(iu.shape[0], np.inf)
                for k in range(iu.shape[0]):
                    try:
                        d2[k] = float(diffs[k] @ np.linalg.solve(covs[iu[k]] + covs[ju[k]], diffs[k]))
                    except np.linalg.LinAlgError:
                        pass
            d2 = np.where(np.isfinite(d2), d2, np.inf)
            k = int(np.argmin(d2))
            if float(d2[k]) >= d2_thresh:
                break
            i, j = int(iu[k]), int(ju[k])
            a, b = self.tracks[i], self.tracks[j]
            lse = float(logsumexp(np.array([a.log_weight, b.log_weight])))
            w_a, w_b = float(np.exp(a.log_weight - lse)), float(np.exp(b.log_weight - lse))
            m_new = w_a * a.mean + w_b * b.mean
            da, db = a.mean - m_new, b.mean - m_new
            P_new = w_a * (a.cov + np.outer(da, da)) + w_b * (b.cov + np.outer(db, db))
            self.tracks = [t for k, t in enumerate(self.tracks) if k not in (i, j)] + [Component(lse, m_new, 0.5 * (P_new + P_new.T))]

    def cap(self, k_max: int) -> None:
        if len(self.tracks) > k_max:
            self.tracks.sort(key=lambda t: -t.log_weight)
            self.tracks = self.tracks[:k_max]
            _normalize(self.tracks)

    def map_estimate(self) -> Component:
        return max(self.tracks, key=lambda t: t.log_weight)

    def __len__(self) -> int:
        return len(self.tracks)


class WindowFilter:
    """Per-chunk moving-pose GM-BF over successive window centroids (cold start per chunk)."""

    def __init__(self, *, sigma_motion_cells: float, cell_m: float, stride: int, n_frames: int, gate_d2: float, k_max: int) -> None:
        # numerics of the paper's runs: sigma * sqrt(T + 1) * sqrt(S / (T + 1)) * cell  (== sigma * sqrt(S) * cell)
        sigma_m = sigma_motion_cells * math.sqrt(n_frames + 1.0) * math.sqrt(max(int(stride), 1) / float(n_frames + 1.0)) * cell_m
        self.Q = sigma_m ** 2 * np.eye(2)
        self.gate_d2 = float(gate_d2)
        self.k_max = int(k_max)
        self.mix: Optional[Mixture] = None
        self.last_xy: Optional[np.ndarray] = None

    def step(self, centroid_xy_m: Tuple[float, float], obs: Sequence[Observation], fallback_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Advance by one window; returns the MAP centroid position (or the field argmax when empty)."""
        cur = np.asarray(centroid_xy_m, dtype=np.float64)
        if self.mix is None:
            self.last_xy = cur
            if not obs:
                self.mix = Mixture()
                return fallback_xy
            self.mix = Mixture.from_observations(obs)
        else:
            self.mix.predict(cur - self.last_xy, self.Q)
            self.mix.update(obs, gate_d2=self.gate_d2)
            self.mix.prune(-15.0)
            self.mix.merge(4.0)
            self.mix.cap(self.k_max)
            self.last_xy = cur
        if len(self.mix) == 0:
            return fallback_xy
        m = self.mix.map_estimate().mean
        return float(m[0]), float(m[1])
