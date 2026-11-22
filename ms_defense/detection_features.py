"""Per-account sliding-window behavioral features for stealing detection.

Each feature is one signal in the literature on model-extraction detection
(boundary over-sampling, out-of-distribution probing, high diversity / low
redundancy, label flipping, extreme-tail coverage, novelty vs. the known
benign manifold, bursty request rates).
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial.distance import cdist

from . import config


class WindowFeatureExtractor:
    def __init__(self, benign_pool: np.ndarray, boundary_band: float = config.BOUNDARY_BAND,
                 tail_quantile: float = config.TAIL_QUANTILE):
        self.benign_pool = np.atleast_2d(benign_pool)
        self.boundary_band = boundary_band
        # reference quantiles of the benign manifold, per feature
        self.lo_q = np.quantile(self.benign_pool, tail_quantile, axis=0)
        self.hi_q = np.quantile(self.benign_pool, 1 - tail_quantile, axis=0)
        self.benign_min = self.benign_pool.min(axis=0)
        self.benign_max = self.benign_pool.max(axis=0)
        self._bin_edges = np.linspace(
            self.benign_min, self.benign_max, 9
        )  # 8 bins per feature

    # -- individual signals -------------------------------------------------
    def boundary_fraction(self, scores: np.ndarray) -> float:
        return float(np.mean(np.abs(scores - 0.5) <= self.boundary_band))

    def tail_fraction(self, X: np.ndarray) -> float:
        in_tail = (X < self.lo_q) | (X > self.hi_q)
        return float(np.mean(in_tail))

    def novelty(self, X: np.ndarray, k: int = 5) -> float:
        """Mean distance to the k nearest benign-manifold points."""
        d = cdist(X, self.benign_pool)
        knn = np.sort(d, axis=1)[:, :k].mean(axis=1)
        return float(knn.mean())

    def pair_diversity(self, X: np.ndarray) -> float:
        """Mean pairwise distance — extraction sets probe broadly, benign
        operational traffic stays on a narrow manifold."""
        if len(X) < 2:
            return 0.0
        d = cdist(X, X)
        iu = np.triu_indices(len(X), k=1)
        return float(d[iu].mean())

    def dedup_ratio(self, X: np.ndarray) -> float:
        keys = np.round(X, 2)
        uniq = np.unique(keys, axis=0).shape[0]
        return uniq / len(X)

    def label_flip_rate(self, labels: np.ndarray) -> float:
        if len(labels) < 2:
            return 0.0
        return float(np.mean(labels[1:] != labels[:-1]))

    def score_entropy(self, scores: np.ndarray, bins: int = 10) -> float:
        hist, _ = np.histogram(scores, bins=bins, range=(0, 1), density=True)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(bins))

    def score_range(self, scores: np.ndarray) -> float:
        return float(scores.max() - scores.min())

    def rate_burst(self, ts: np.ndarray) -> float:
        """High frequency with ultra-low gap variance = machine-driven sweep."""
        if len(ts) < 3:
            return 0.0
        gaps = np.diff(ts)
        cv = gaps.std() / (gaps.mean() + 1e-9)
        qps = 1.0 / (gaps.mean() + 1e-9)
        # large qps AND near-constant cadence
        return float(min(qps / 10.0, 3.0) * np.exp(-cv))

    def sign_flip_rate(self, X: np.ndarray) -> float:
        """Fraction of consecutive queries moving in opposite directions in
        feature space. Bisection walks drive monotonically toward the
        boundary (low flip rate); benign operational rows are independent
        draws (flip ~0.5). Suspicion = monotonicity, so return 1 - flip."""
        if len(X) < 3:
            return 0.0
        deltas = np.diff(X, axis=0)
        s = np.sign(deltas)
        flips = (s[1:] * s[:-1] < 0).mean()
        return float(1.0 - flips)

    def coord_concentration(self, X: np.ndarray) -> float:
        """Bisection perturbs one/few coordinates at a time: the mean L1/L2
        ratio of per-step deltas is ~1 for concentrated moves and ~sqrt(k)
        for diffuse ones. Normalized to ~[0, 1]."""
        if len(X) < 2:
            return 0.0
        deltas = np.diff(X, axis=0)
        l1 = np.abs(deltas).sum(axis=1)
        l2 = np.linalg.norm(deltas, axis=1) + 1e-12
        ratio = (l1 / l2).mean()
        return float(np.clip(ratio / np.sqrt(X.shape[1]), 0, 1))

    def hashed_bin_signatures(self, X: np.ndarray) -> np.ndarray:
        """Stable per-query signature used for cross-account overlap analysis."""
        binned = np.dstack([
            np.digitize(X[:, j], self._bin_edges[:, j])
            for j in range(X.shape[1])
        ]).squeeze()
        # polynomial rolling hash over bin ids
        h = np.zeros(len(X), dtype=np.int64)
        for j in range(binned.shape[1]):
            h = (h * 131 + binned[:, j]) % (2**61 - 1)
        return h

    def coarse_signature(self, X: np.ndarray, n_bins: int = 4) -> int:
        """Multi-resolution locality signature: each feature is quantized
        against the *benign* range into 4 coarse bins and the bin-id tuple is
        hashed. Two accounts probing the same broad region of feature space
        share signatures even when their exact points never coincide — this
        survives the attacker's per-account region splitting."""
        edges = np.linspace(self.benign_min, self.benign_max, n_bins + 1)
        binned = np.dstack([
            np.clip(np.digitize(X[:, j], edges[:, j]), 0, n_bins - 1)
            for j in range(X.shape[1])
        ]).squeeze()
        # histogram of coarse cell visits -> order-independent account profile
        cells = np.zeros(len(X), dtype=np.int64)
        for j in range(binned.shape[1]):
            cells = (cells * n_bins + binned[:, j]) % (2**61 - 1)
        vals, counts = np.unique(cells, return_counts=True)
        profile = dict(zip(vals.tolist(), counts.tolist()))
        h = hashlib.blake2b(digest_size=8)
        for v in sorted(profile):
            h.update(str(v).encode()); h.update(b":"); h.update(str(profile[v]).encode()); h.update(b"|")
        return int.from_bytes(h.digest(), "little")

    def coarse_cell_set(self, X: np.ndarray, n_bins: int = 4) -> set[int]:
        """Set of coarse cells an account has touched (Jaccard input)."""
        edges = np.linspace(self.benign_min, self.benign_max, n_bins + 1)
        binned = np.dstack([
            np.clip(np.digitize(X[:, j], edges[:, j]), 0, n_bins - 1)
            for j in range(X.shape[1])
        ]).squeeze()
        cells = np.zeros(len(X), dtype=np.int64)
        for j in range(binned.shape[1]):
            cells = (cells * n_bins + binned[:, j]) % (2**61 - 1)
        return set(cells.tolist())

    # -- full vector --------------------------------------------------------
    FEATURE_NAMES = [
        "boundary_fraction", "tail_fraction", "novelty", "pair_diversity",
        "dedup_ratio", "label_flip_rate", "score_entropy", "score_range",
        "rate_burst", "sign_flip_rate", "coord_concentration", "span_coverage",
    ]

    def extract(self, X: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                ts: np.ndarray) -> np.ndarray:
        span = float(np.mean([
            np.ptp(X[:, j]) / (self.benign_max[j] - self.benign_min[j] + 1e-9)
            for j in range(X.shape[1])
        ]))
        return np.array([
            self.boundary_fraction(scores),
            self.tail_fraction(X),
            self.novelty(X),
            self.pair_diversity(X),
            self.dedup_ratio(X),
            self.label_flip_rate(labels),
            self.score_entropy(scores),
            self.score_range(scores),
            self.rate_burst(ts),
            self.sign_flip_rate(X),
            self.coord_concentration(X),
            np.clip(span, 0, 5),
        ])
